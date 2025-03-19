import torch
import nibabel as nib
import numpy as np
import pandas as pd
import os
from typing import Tuple, List
from readMesh import *

def gifti_to_data(gifit):
    return torch.tensor(np.stack([d.data for d in gifit.darrays], axis=1), dtype=torch.float32)

def CBIG_gwMRF_build_time_matrix(
    input_fullpaths,
    output_path,
    start_idx,
    end_idx,
    fsaverage,
    lh_output_file,
    rh_output_file
):
    """
    This function concatenates timeseries data from several subjects
    
    Args:
        input_fullpaths: File containing full paths to all subjects' surf data
        output_path: Path to where output files are written
        start_idx: Start index for selecting subsets in the subject list
        end_idx: End index for selecting subsets in the subject list
        fsaverage: Which fsaverage resolution to use
        lh_output_file: Filename for left hemisphere product matrix
        rh_output_file: Filename for right hemisphere product matrix

    Returns:
        Tuple containing paths to:
        - lh_output_file: Matrix containing the concatenated timeseries data of left hemisphere
        - rh_output_file: Matrix containing the concatenated timeseries data of right hemisphere
    """
    if isinstance(start_idx, str):
        start_idx = int(start_idx)
    if isinstance(end_idx, str):
        end_idx = int(end_idx)
    
    data = input_fullpaths
    full_paths = data.values.tolist()
    num_subs, num_scans = data.shape

    lh_avg_mesh, _ = CBIG_ReadNCAvgMesh(prams=None , hemi='lh', mesh_name=fsaverage, surf_type='inflated', label = 'cortex')
    rh_avg_mesh, _ = CBIG_ReadNCAvgMesh(prams=None, hemi='rh', mesh_name=fsaverage, surf_type='inflated', label = 'cortex')

    matrix_number_of_scans = 0
    
    for k in range(start_idx, end_idx + 1):
        for i in range(num_scans):
            if pd.notna(full_paths[k][i]) and full_paths[k][i].strip():
                matrix_number_of_scans += 1
                
    first_scan = nib.load(full_paths[0][0])
    first_scan_data = gifti_to_data(first_scan)
    length_of_time = first_scan_data.shape[-1]
    lh_mask = (lh_avg_mesh['MARS_label'] == 2)
    rh_mask = (rh_avg_mesh['MARS_label'] == 2)
    
    lh_time_mat = torch.zeros((lh_mask.sum(), length_of_time * matrix_number_of_scans), dtype=torch.float32)
    rh_time_mat = torch.zeros((rh_mask.sum(), length_of_time * matrix_number_of_scans), dtype=torch.float32)
    print(f'initial time matrix shape: {lh_time_mat.shape}')
    matrix_number_of_scans = 0
    scans = []
    files_used_lh = []
    files_used_rh = []

    for k in range(start_idx, end_idx + 1):
        print(f'Processing subject number {k}')
        sub_num_scans = sum(1 for x in full_paths[k] if isinstance(x, str) and len(x) > 4)
        scans.append(sub_num_scans)

        for i in range(num_scans):
            if pd.notna(full_paths[k][i]) and full_paths[k][i].strip():
                lh_input = full_paths[k][i]

                if '.L.func.gii' in lh_input:
                    rh_input = lh_input.replace('.L.func.gii', '.R.func.gii')
                elif '.R.func.gii' in lh_input:
                    rh_input = lh_input
                    lh_input = lh_input.replace('.R.func.gii', '.L.func.gii')
                else:
                    raise ValueError("Filename does not contain 'L' or 'R'")
                files_used_lh.append(lh_input)
                files_used_rh.append(rh_input)

                for j, (input_file, mask, time_mat) in enumerate([
                    (lh_input, lh_mask, lh_time_mat),
                    (rh_input, rh_mask, rh_time_mat)
                ]):
                    hemi_data = nib.load(input_file)
                    hemi_data_tensor = gifti_to_data(hemi_data)
                    vol = torch.tensor(hemi_data_tensor.reshape(-1, hemi_data_tensor.shape[-1]))[mask] # shape of (1172, 1200) for left hemisphere and (870, 1200) for right hemisphere
                    ## Shape of (N, 1200)
                    vol = vol - vol.mean(dim=1, keepdim=True)
                    vol = vol / vol.std(dim=1, keepdim=True) ## Normalize each subject
                    start_idx = matrix_number_of_scans * length_of_time
                    end_idx = (matrix_number_of_scans + 1) * length_of_time
                    time_mat[:, start_idx:end_idx] = vol

                matrix_number_of_scans += 1

    # Save results
    lh_output_path = os.path.join(output_path, lh_output_file)
    rh_output_path = os.path.join(output_path, rh_output_file)
        
    torch.save({
        'time_mat': lh_time_mat,
        'scans': scans,
        'files_used': files_used_lh
    }, lh_output_path)
    
    torch.save({
        'time_mat': rh_time_mat,
        'scans': scans,
        'files_used': files_used_rh
    }, rh_output_path)
    
    return lh_output_path, rh_output_path

