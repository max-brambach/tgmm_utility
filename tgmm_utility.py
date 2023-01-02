import glob
import os
import datetime
import subprocess
from multiprocessing import Pool
import numpy as np
import SegmentationIO
from pathlib import Path
import glob
import pandas as pd
from scipy.spatial import KDTree

class RunTGMM(object):
    """
    Call executables from TGMM software (McDole et al. 2018) to segment nuclei in SPIM images.
    
    Default parameter set is from the repository that was published alongside with the paper.
    """
    def __init__(self,
                 name,
                 output_directory,
                 local_executables,
                 parameters):
        self.name = name
        self.loc_exec = local_executables
        self.output_directory = os.path.join(output_directory, name)
        default_params={
            'imgFilePattern': None,
            'debugPathPrefix': self.output_directory,
            'anisotropyZ':1,
            'backgroundThreshold':None,
            'persistanceSegmentationTau':None,
            'betaPercentageOfN_k':0.01,
            'nuPercentageOfN_k':1.0,
            'alphaPercentage':0.8,
            'maxIterEM':100,
            'tolLikelihood':1e-6,
            'regularizePrecisionMatrixConstants_lambdaMin':0.01,
            'regularizePrecisionMatrixConstants_lambdaMax':0.1,
            'regularizePrecisionMatrixConstants_maxExcentricity':9.0,
            'temporalWindowForLogicalRules':5,
            'thrBackgroundDetectorHigh':1.1,
            'thrBackgroundDetectorLow':0.2,
            'SLD_lengthTMthr':5,
            'radiusMedianFilter':2,
            'minTau':2,
            'conn3D':26,
            'estimateOpticalFlow':0,
            'maxDistPartitionNeigh':80.0,
            'deathThrOpticalFlow':-1,
            'minNucleiSize':150,
            'maxNucleiSize':4000,
            'maxPercentileTrimSV':0.4,
            'conn3DsvTrim':6,
            'maxNumKNNsupervoxel':10,
            'maxDistKNNsupervoxel':41.0,
            'thrSplitScore':-1.0,
            'thrCellDivisionPlaneDistance':12.403,
            'thrCellDivisionWithTemporalWindow':0.456,
        }
        self.parameters = default_params
        self.update_parameters(parameters)
        self.generate_output_folder()
    
    def generate_output_folder(self):
        """
        Generate folder at self.output_directory named self.name.
        
        This is done to keep all segmentation related files together and accessible by self.name.
        Otherwise, the segmentation would be in a weired folder called GMEM... 
        """
        Path(self.output_directory).mkdir(parents=True, exist_ok=True)
    
    def update_parameters(self, new_parameters):
        """
        Update the used parameter set. Unspecified parameters remain unchanged.
        """
        self.parameters.update(new_parameters)
        
    def generate_tgmm_config(self):
        """
        Generate tgmm config file as .txt and save it at self.output_directory.
        """
        savepath = os.path.join(self.output_directory, 'cf-tgmm-' + self.name + '.txt' )
        self.config_file = savepath
        with open(savepath, 'w') as file:
            file.write('# TGMM config file\n# generated {}\n'.format(datetime.datetime.now()))
            for param, value in self.parameters.items():
                file.write('{}={}\n'.format(param, value))
                
    def run_watershed_segmentation(self):
        """
        Run the first stage of the segmentation; hierarchical watershed segmentation.
        """
        #print(self.loc_exec)
        #print(self.config_file)
        call_str = os.path.join(self.loc_exec, 'ProcessStack.exe') + ' ' + self.config_file + ' 0'
        #print(call_str)
        return subprocess.call(call_str, shell=False)
    
    def run_tgmm(self):
        """
        Run the second stage of the segmentation; gaussian mixture.
        """
        call_str = os.path.join(self.loc_exec, 'TGMM.exe') + ' ' + self.config_file + ' 0 0'
        return subprocess.call(call_str, shell=False)
    
    def get_segmentation_as_df(self):
        """
        Summarise the segmentation .xml files as one pd.DataFrame.
        
        The df is returned and stored in self.df. 
        Columns are:
        * x, y, z: coordinates of the segmented nuclei
        * sample: name of the sample (self.name)
        * run: 0,1,2,...,n_xmls, number to distinguish the different TGMM runs (i.e. individual GMEM.. folders/.xml files).
        """
        folders = glob.glob(os.path.join(self.output_directory, 'GMEM*'))
        df_list = []
        for i, f in enumerate(folders):
            file = os.path.join(f, 'XML_finalResult_lht', 'GMEMfinalResult_frame0000.xml')
            sdf = SegmentationIO.read_tgmm(file)
            sdf = sdf[['x', 'y', 'z', 'cell id']]
            sdf['run'] = i
            sdf['sample'] = self.name
            df_list.append(sdf)
        self.df = pd.concat(df_list)
        return self.df
    
    def combine_segmentations(self, distance=5):
        """
        Combine the segmentations of multiple runs based on a distance criterion.
        
        Only segmented cells that are outside `distance` are combined.
        """
        for run in self.df['run'].unique():
            if run == 0:
                df = self.df.loc[self.df['run'] == run].copy()
                continue
            current_df = self.df.loc[self.df['run'] == run]
            df = merge_pointclouds(df, current_df, distance)
        self.df = df
        return self.df
            
    
def merge_pointclouds(df_a, df_b, min_dist):
    """
    Merge two point clouds based on a distnce criterion.
    
    All points in df_a are used; points in df_b are used if their distance to the closest point in df_b is lower than min_dist.
    """
    Xa = df_a[list('xyz')].values
    Xb = df_b[list('xyz')].values
    tree = KDTree(Xa)
    dist, idx = tree.query(Xb)
    df_b = df_b.loc[dist>=min_dist]
    return pd.concat([df_a, df_b])
                
            
