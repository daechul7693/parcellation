import os
import numpy as np
import pandas as pd
import torch
import torch.functional as F
from scipy.io import loadmat, savemat

import pandas as pd

def save_cov_as_csv(file_prefix, cov_mat, dim):
    """
    Save the covariance matrix and dim in CSV form.
    Creates two files: one for the matrix, one for dim.
    """
    df_cov = pd.DataFrame(cov_mat)
    df_cov.to_csv(file_prefix + "_cov.csv", index=False)

    with open(file_prefix + "_dim.txt", "w") as f:
        f.write(str(dim))

def CBIG_gwMRF_build_prod_matrix(lh_input_filename, rh_input_filename, output_path, lh_output_filename, rh_output_filename):
   """ 
    This function precomputes the products of timeseries data.
    The time series data can be created with alex_build_time_matrix
    
    Input
       - lh_input_filename = left hemisphere matlab timeseries file, created by CBIG_create_time_matrix
       - rh_input_filename = right hemisphere matlab timeseries file, created by CBIG_create_time_matrix
       - output_path = folder to save the product timeseries matrix
       - lh_output_filename = filename for left hemisphere product matrix
       - rh_output_filename = filename for right hemisphere product matrix
     
    Output
       - lh_output_file = left hemisphere precomputed matrix file
       - rh_output_file = right hemisphere precomputed matrix file
       - dim = dimension of the matrix file
   """
   lh_data = torch.load(lh_input_filename, weights_only=False)
   
   lh_time_mat = lh_data['time_mat']
   # shape of mat: (V, 1200 * num_scan) 
   lh_time_mat = lh_time_mat - lh_time_mat.mean(dim=1, keepdim=True)
   # print(f'{lh_time_mat.mean(dim=1, keepdim=True).shape}')
   lh_norm = torch.norm(lh_time_mat, p=2, dim=1, keepdim=True)
   lh_time_mat = lh_time_mat / lh_norm
   lh_cov = torch.matmul(lh_time_mat, lh_time_mat.T)
   dim = lh_time_mat.shape[1]
   lh_output_file = os.path.join(output_path, lh_output_filename)
   # print(f'This is the shape of cov_mat: {lh_cov.shape} and this is how it looks like: \n{lh_cov}')
   torch.save({'cov_mat': lh_cov, 'dim': dim}, lh_output_file)
   
   # Similarly for right hemisphere
   rh_data = torch.load(rh_input_filename, weights_only=False)
   rh_time_mat = rh_data['time_mat']
   rh_time_mat = rh_time_mat - rh_time_mat.mean(dim=1, keepdim=True)
   rh_norm = torch.norm(rh_time_mat, p=2, dim=1, keepdim=True)
   rh_time_mat = rh_time_mat / rh_norm
   rh_cov = torch.matmul(rh_time_mat, rh_time_mat.T)
   dim = rh_time_mat.shape[1]
   rh_output_file = os.path.join(output_path, rh_output_filename)
   torch.save({'cov_mat': rh_cov, 'dim': dim}, rh_output_file)
   
   return lh_output_file, rh_output_file, dim