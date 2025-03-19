import os
import argparse
import pandas as pd
from time_mat import *
from train import *
from prod_mat import *
from download import *
from set_prams import *
import torch

def CBIG_gwMRF_build_data_and_perform_clustering(input_fullpaths, output_path, start_idx, end_idx,
                                                   num_left_cluster, num_right_cluster,
                                                   smoothcost, num_iterations, num_runs,
                                                   start_gamma, exponential, iter_reduce_gamma=None):
    """
    Reads in the data, computes the multiplication matrix and performs clustering.
    If the multiplication matrix files already exist, those steps are skipped.
    
    Args:
        input_fullpaths (str): Path to a file containing full paths to subjects' surf data.
        output_path (str): Output directory.
        start_idx (int): Starting index (also used as the seed).
        end_idx (int): Ending index.
        num_left_cluster (int): Number of clusters for the left hemisphere.
        num_right_cluster (int): Number of clusters for the right hemisphere.
        smoothcost (float): Weight for the smoothness cost in the MRF.
        num_iterations (int): Number of iterations per random initialization.
        num_runs (int): Number of random initializations.
        start_gamma (int): Starting gamma value.
        exponential (float): Exponential parameter.
        iter_reduce_gamma (int, optional): Parameter for reducing gamma per iteration.
                                           Defaults to 300 if not provided.
    
    Returns:
        prams (dict): Dictionary containing parameters used in clustering.
        results (dict): Dictionary containing the clustering results.
    """
    if iter_reduce_gamma is None:
        iter_reduce_gamma = 300

    fsaverage = 'fsaverage6'
    os.makedirs(output_path, exist_ok=True)
    time_data_dir = os.path.join(output_path, 'time_data')
    mult_mat_dir = os.path.join(output_path, 'mult_mat')
    clustering_dir = os.path.join(output_path, 'clustering')
    os.makedirs(time_data_dir, exist_ok=True)
    os.makedirs(mult_mat_dir, exist_ok=True)
    os.makedirs(clustering_dir, exist_ok=True)

    rh_mult_matrix_file = os.path.join(mult_mat_dir, 'rh_mult_matrix.pt')
    
    if not os.path.exists(rh_mult_matrix_file):
        lh_output_file, rh_output_file = CBIG_gwMRF_build_time_matrix(
            input_fullpaths,
            time_data_dir,
            start_idx,
            end_idx,
            fsaverage,
            'lh_time_matrix.pt',
            'rh_time_matrix.pt'
        )

        # Build the product (multiplication) matrix.
        lh_output_mult_mat_file, rh_output_mult_mat_file, dim = CBIG_gwMRF_build_prod_matrix(
            lh_output_file,
            rh_output_file,
            mult_mat_dir,
            'lh_mult_matrix.pt',
            'rh_mult_matrix.pt'
        )
    else:
        lh_output_mult_mat_file = os.path.join(mult_mat_dir, 'lh_mult_matrix.pt')
        rh_output_mult_mat_file = rh_mult_matrix_file
        loaded = torch.load(rh_output_mult_mat_file)
        dim = loaded.get('dim')
        
    prams = CBIG_gwMRF_set_prams()
    results = CBIG_gwMRF_graph_cut_clustering_iter_split(prams)
    
    print('Done Process')
    return prams, results

def main():
    os.makedirs('data', exist_ok=True)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    if not os.path.exists(current_dir + '/data/downloaded'):
        download()
    
    out_L_path = os.path.join(current_dir, "data", "downloaded", "100610", "hcp1200", "rest", "rfMRI_REST1_LR", "rfMRI_REST1_LR_Atlas.L.func.gii")
    out_R_path = os.path.join(current_dir, "data", "downloaded", "100610", "hcp1200", "rest", "rfMRI_REST1_LR", "rfMRI_REST1_LR_Atlas.R.func.gii")
    input_path = pd.DataFrame([out_L_path, out_R_path])
    
    prams, results = CBIG_gwMRF_build_data_and_perform_clustering(
        input_fullpaths= input_path,
        output_path='results',
        start_idx=0,
        end_idx=1,
        num_left_cluster=7,
        num_right_cluster=7,
        smoothcost=5000,
        num_iterations=7,
        num_runs=2,
        start_gamma=50000000,
        exponential=15,
        iter_reduce_gamma=300
    )
    
    print("Clustering completed.")


if __name__ == '__main__':
    main()
