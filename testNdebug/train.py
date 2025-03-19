import torch
import nibabel as nib
import numpy as np
import os
from scipy.io import loadmat
from scipy.sparse import coo_matrix, lil_matrix, csc_matrix
import scipy.sparse as sp
import pygco
from readMesh import *
from scipy.special import ive


def CBIG_gwMRF_graph_cut_clustering_iter_split(prams):
    """
    This function loads the data, normalizes it and calls the clustering function 

    Args:
        prams (dict): a dictionary containing various input parameters
        
    Return:
        results (dict): A dictionary containing the labels and various additional results
    """
    lh_avg_mesh, lh_prams = CBIG_ReadNCAvgMesh(prams=prams, hemi='lh', mesh_name=prams['fsaverage'], surf_type='inflated', label= 'cortex')
    l1 = torch.where(lh_avg_mesh['MARS_label'] == 2)[0]
    rh_avg_mesh, rh_prams = CBIG_ReadNCAvgMesh(prams=prams, hemi='rh', mesh_name=prams['fsaverage'], surf_type='inflated', label= 'cortex')
    r1 = torch.where(lh_avg_mesh['MARS_label'] == 2)[0]
    print(l1)
    lh_vol, rh_vol = None, None
    if lh_prams['grad_prior'] == 'gordon_water':
        CBIG_build_LogOdds(lh_avg_mesh, rh_avg_mesh, prams)
    
    if lh_prams['separate_hemispheres'] == 1 or lh_prams['local_concentration'] > 0.00:
        if lh_prams['skip_left'] == 0:
            print(f"Processing left hemisphere: {lh_prams['lh_avg_file']}")
            if lh_prams['pca'] == 0:
                lh_vol = torch.load(lh_prams['lh_avg_file'])['lh_vol']
                ## If we have volume data and it has more index then mesh index, we will use only index amount of mesh
                if lh_vol.shape[0] > len(l1):
                    lh_vol = lh_vol[l1, :]
            else:
                lh_vol = torch.load(lh_prams['lh_avg_file'], weights_only=False)['cov_mat']     

            torch.manual_seed(5489)
            print("Computing parameters in split brains")

            lh_prams['cluster'] = lh_prams['left_cluster']

            if prams['pca'] == 1:
                likelihod, results = CBIG_gwMRF_graph_cut_clustering_split_newkappa_prod(torch.tensor(lh_vol, dtype=torch.float32), lh_prams, 'lh')
            
            print(results['full_label'])
            results['lh_label'] = results['full_label']
            results['lh_final_likeli'] = results['final_likeli']
            results['lh_likeli_pos'] = results['likeli_pos']
        
        else:
            results = ...

        # Right Hemisphere
        print('\n right hemisphere')
        if rh_prams['skip_right'] == 0:
            if rh_prams['pca'] == 0:
                rh_vol = loadmat(rh_prams['rh_avg_file'])['rh_vol']
                if rh_vol.shape[0] > len(r1):
                    rh_vol = rh_vol[r1, :]
            else:
                rh_vol = torch.load(rh_prams['rh_avg_file'], weights_only=False)['cov_mat']
                
            torch.manual_seed(5489)
            prams['cluster'] = rh_prams['right_cluster']

            if prams['pca'] == 1:
                likelihood, results_rh = CBIG_gwMRF_graph_cut_clustering_split_newkappa_prod(torch.tensor(rh_vol), rh_prams, 'rh')
            
            results['rh_label'] = results_rh['full_label']
        else:
            results_rh = ...
            
        results['D'] = results_rh['D'] + results['D']
        results['S'] = results_rh['S'] + results['S']
        results['E'] = results_rh['E'] + results['E']
        results['UnormalizedE'] = results_rh['UnormalizedE'] + results['UnormalizedE']
        results['gamma'] = np.concatenate([results['gamma'], results_rh['gamma']])
        results['kappa'] = np.concatenate([results['kappa'], results_rh['kappa']])
    
    return results

def CBIG_gwMRF_graph_cut_clustering_split_newkappa_prod(x, prams, hemisphere):
    """
    Perform actual clustering

    Args:
        x (torch.tensor): input MRI data of shape (n, features), where n is the number of vertices. This data should be lh_avg_file or rh_avg_file extracting from prams dictionary
        prams (dict): a dictionary containing various input parameters
        hemisphere (str): a string indicating hemisphere 'lh' or 'rh'
    """
    avg_mesh, prams = CBIG_ReadNCAvgMesh(prams, hemi=hemisphere ,mesh_name=prams['fsaverage'], surf_type='inflated', label='cortex')  
    
    cortex_vertices = np.count_nonzero([avg_mesh['MARS_label'] == 2])  #1172 (N)
    if not prams['potts']:
        if hemisphere == 'lh':
            neighborhood = CBIG_build_sparse_gradient(avg_mesh, prams['lh_grad_file'], prams)  # (1172, 1172) (N, N)  Neighborhood from gradient file 
        elif hemisphere == 'rh':
            neighborhood = CBIG_build_sparse_gradient(avg_mesh, prams['rh_grad_file'], prams)
        prams['fileID'].write("with gradient \n")

        
    else: # potts' model
        neighborhood = CBIG_build_sparse_neighborhood(avg_mesh)
        prams['fileID'].write('with potts \n')
        
            
    idx_cortex_vertices = torch.where(avg_mesh['MARS_label'] == 2)[0]
    grad_file = None 
    
    if hemisphere == 'lh':
        grad_file = loadmat(prams['lh_grad_file'])
    else:
        grad_file = loadmat(prams['rh_grad_file'])
    grad_matrix = grad_file['border_matrix'][:, idx_cortex_vertices]  ## shape  (6, 1172) = (6, V)
    grad_matrix = torch.tensor(grad_matrix, dtype=torch.float32)
    torch.manual_seed(prams['seed'])
    prams['fileID'].write("some log message\n")
    
    likeli, initial_assigned, max_max_likeli = CBIG_initialize_cluster(prams['cluster'], prams['dim'], x, grad_matrix) # likelihood of each cluster (N, L) -> (1172, 7[manually assigned])
    prams['graphCutIterations'] = 10
    gamma = torch.zeros((1, prams['cluster']))[0] + prams['start_gamma'] #initialize tau(or gamma) parameters May be tau?
    label, new_energy, D, S = CBIG_compute_labels(cortex_vertices, likeli, neighborhood, prams) # Operating first graph cut 
    # label: (N, )
    label, gamma = CBIG_assign_empty_cluster(label, prams['cluster'], grad_matrix, gamma, prams)
    likelihood, results, label = CBIG_gwMRF_graph_cut_clustering_split_standard(x, prams, hemisphere, label, 
                                                                                neighborhood, gamma, 1, grad_matrix)   #initially compute labels with given formula, do we update x
    initial_label = label
    for i in range(0,5,2):
        for j in range(1,1001):
            gamma_head = CBIG_UpdateGamma(gamma, label, prams, hemisphere)
            if (j > 1) and (gamma == gamma_head).all():
                break
            else:
                gamma = gamma_head
                prams['graphCutIterations'] = 10 ** (i + 2)
                # It only use updated label and keep using x for mu_times_x
                likelihood, results, label = CBIG_gwMRF_graph_cut_clustering_split_standard(x, prams, hemisphere,  
                                                                                            label, neighborhood, gamma, 1, grad_matrix, j)   #keep updating labels 
                gamma = results['gamma']
    results['initial_full_label'] = torch.zeros(avg_mesh['vertices'].shape[0], dtype=torch.long)
    results['initial_full_label'][avg_mesh['MARS_label'] == 2] = torch.tensor(initial_label[:cortex_vertices], dtype=torch.long)
    if prams['reduce_gamma'] == 1:
        likelihood, results, label = CBIG_gwMRF_graph_cut_clustering_split_reduce(x, prams, hemisphere, label, neighborhood, 
                                                                                  gamma, grad_matrix, avg_mesh)
        
    return likelihood, results

def CBIG_SaveIntermediate(label, prams, avg_mesh, cortex_vertices, hemisphere, k, gamma):
    """
    Save optimized parameters 

    Args:
        label (torch.tensor): Shape of (N, ) which contains information about each vertex's label and its value could be 1 through L
        prams (dict): a dictionary containing various input parameters
        avg_mesh (dict): Preprocessed mesh data dictionary containing MARS_label 
        cortex_vertices (int): _description_
        hemisphere (str): a string indicating hemisphere 'lh' or 'rh'
        k (int): Number of clusters 
        gamma (torch.tensor): Concentration parameters of shape (L,). This parameter is used to compute label for spatial term. Gamma variables which is tau in the paper and its shape is (L, ) where L is the number of clutsers
    """
    current_label = np.zeros(avg_mesh['vertices'].shape[0], dtype=int)
    current_label[avg_mesh['MARS_label'] == 2] = np.array(label[:cortex_vertices])

    # Create the in-between results directory if it doesn't exist
    inbetween_results_path = os.path.join(prams['output_folder'], 'inbetween_results')
    os.makedirs(inbetween_results_path, exist_ok=True)

    # Construct the file path for saving
    file_path = os.path.join(
        inbetween_results_path,
        f"{prams['output_name']}_seed_{prams['seed']}_{hemisphere}_reduction_iteration_{k}.npz"
    )
    # Save current_label and gamma using NumPy
    np.savez(file_path, current_label=current_label, gamma=np.array(gamma))
    torch.save({'current_label': torch.tensor(current_label, dtype=torch.int32), 
                'gamma': gamma}, file_path)

def CBIG_gwMRF_graph_cut_clustering_split_reduce(x, prams, hemisphere, label, neighborhood, gamma, grad_matrix, avg_mesh):
    """
    MAP2: This is for doing graph cut but reducing gamma till convergence as it is mentioned in 

    Args:
        x (torch.tensor): input MRI data of shape (n, features), where n is the number of vertices.
        prams (dcit): a dictionary containing various input parameters
        hemisphere (str):a string indicating hemisphere 'lh' or 'rh'
        label (torch.tensor): Shape of (N, ) which contains information about each vertex's label and its value could be 1 through L
        neighborhood (torch.tensor): _description_
        gamma (torch.tensor): Concentration parameters of shape (L,). This parameter is used to compute label for spatial term. Gamma variables which is tau in the paper and its shape is (L, ) where L is the number of clutsers
        grad_matrix (torch.tensor): _description_
        avg_mesh (torch.tensor): Preprocessed mesh data dictionary containing MARS_label 
    Returns:
        _type_: _description_
    """
    for k in range(prams['iter_reduce_gamma']):
        gamma_bar = gamma.clone()
        gamma /= prams['reduce_speed']
        gamma = torch.clamp(gamma, min=0, max=1000)
        
        likelihood, results, label = CBIG_gwMRF_graph_cut_clustering_split_standard(x, prams, hemisphere, label, neighborhood, gamma, 1, grad_matrix)
        for i in range(0, 5, 2):
            for j in range(1000):
                gamma_head = CBIG_UpdateGamma(gamma, label, prams, hemisphere)
                if (j > 1) and torch.allclose(gamma, gamma_head, atol=1e-3):
                    break
                else:
                    gamma = gamma_head
                    i = i 
                    prams['graphCutIterations'] = 10 ** (i + 2)
                    likelihood, results, label = CBIG_gwMRF_graph_cut_clustering_split_standard(x, prams, hemisphere, label, 
                                                                                                neighborhood, gamma, 10**(-i), grad_matrix)
                    gamma = results['gamma']
     
        prams['fileID'].write(f"reduction iteration: {k}\n")
        gh = ' '.join(map(str, gamma_bar))
        prams['fileID'].write(f'gamma bar {gh}\n')
        g = ' '.join(map(str, gamma))
        prams['fileID'].write(f'gamma     {g}\n')
        if (k > 1) and torch.mean(gamma) >= torch.mean(gamma_bar):
            results['gamma'] = gamma_bar
            break
        else: 
            cortex_vertices = np.count_nonzero([avg_mesh['MARS_label'] == 2])
            CBIG_SaveIntermediate(label, prams, avg_mesh, cortex_vertices, hemisphere, k, gamma)
    return likelihood, results, label

def CBIG_UpdateGamma(gamma, label, prams, hemisphere):
    """"
    By generating components and using them, we update the value of gamma
    
    Args:
        gamma (torch.tensor): Concentration parameters of shape (L,). This parameter is used to compute label for spatial term. Gamma variables which is tau in the paper and its shape is (L, ) where L is the number of clutsers
        label (torch.tensor): Shape of (N, ) which contains information about each vertex's label and its value could be 1 through L
        prams (dict):a dictionary containing various input parameters
        hemisphere (str): a string indicating hemisphere 'lh' or 'rh'

    Returns:
        gamma_head(torch.tensor): Updated gamma
    """
    avg_mesh, _ = CBIG_ReadNCAvgMesh(prams, hemisphere, prams['fsaverage'], surf_type='inflated', label='cortex')
    cortex_mask = (avg_mesh['MARS_label'] == 2) # 1172, -> (1172, 1172)
    # cortex_vertices = cortex_mask.sum().item()
    cortex_vertices = np.count_nonzero([avg_mesh['MARS_label'] == 2])
    full_label = torch.zeros(avg_mesh['vertices'].shape[0], dtype=torch.long)
    
    full_label[cortex_mask] = torch.tensor(label[:cortex_vertices], dtype=torch.long)

    # Generate components
    lh_ci, _, _, _ = CBIG_gwMRF_generate_components(avg_mesh, avg_mesh, full_label, full_label)
    
    # Restrict lh_ci to cortex vertices only (fix indexing)
    lh_ci = lh_ci[cortex_mask]

    # Identify clusters with multiple connected components
    bin_vector = torch.zeros(prams['cluster'], dtype=torch.long)
    for l in range(prams['cluster']):
        idx = (label == l)
        unique_val = np.unique(lh_ci[idx])
        bin_vector[l] = 1 if len(unique_val) > 1 else 0

    # Update gamma (corrected logic)
    gamma_head = gamma.clone()
    gamma_head[(bin_vector == 1) & (gamma == 0)] = 1000
    gamma_head[(bin_vector == 1) & (gamma != 0)] *= prams['reduce_speed']
    
    return gamma_head


def CBIG_gwMRF_graph_cut_clustering_split_standard(x, prams, hemisphere, label, neighborhood, gamma, termination, grad_matrix, i =0):
    """
    Step 2 of MAP1: Using label to infer the other parameters 
    This function is used in iterations until gamma is converged
    Args:
        x (torch.tensor): input MRI data of shape (V, V), where n is the number of vertices.
        prams (dict): a dictionary containing various input parameters
        hemisphere (str): a string indicating hemisphere 'lh' or 'rh'
        label (torch.tensor): Shape of (N, ) which contains information about each vertex's label and its value could be 1 through L
        neighborhood (torch.tensor): _description_
        gamma (torch.tensor): Concentration parameters of shape (L,). This parameter is used to compute label for spatial term. Gamma variables which is tau in the paper and its shape is (L, ) where L is the number of clutsers
        termination (int): _description_
        grad_matrix (torch.tensor): _description_

    Returns:
        likelihood (torch.tensor): _description_
        result (dict): 
        label (torch.tensor): Shape of (N, ) which contains information about each vertex's label and its value could be 1 through L
    """
    new_energy = float('inf')
    # Read Mesh data
    avg_mesh, prams = CBIG_ReadNCAvgMesh(prams, hemi=hemisphere, mesh_name= prams['fsaverage'], surf_type='inflated', label= 'cortex')
    # cortex_vertices = torch.sum(avg_mesh['mesh_data'][avg_mesh['MARS_label'] == 2]).item()
    cortex_vertices = np.count_nonzero([avg_mesh['MARS_label'] == 2])
    likeli= 0 
    regularize = False
    for j in range(prams['iterations']):
        old_energy = new_energy
        
        # Part for U_global 
        if prams['kappa_vector'] == 1:
            likeli, max_max_likeli, kappa = CBIG_compute_likelihood_vector_of_kappa(x, label, prams['cluster'], prams['dim'], regularize)
        else:
            likeli, max_max_likeli, kappa = CBIG_compute_likelihood(x, label, prams['cluster'], prams['dim'])
        
        # Part for U_spatial
        likeli_pos, max_max_likeli_local = CBIG_get_position(prams, label, hemisphere, gamma, regularize)
        
        label, new_energy, E_current_D, E_current_S = CBIG_compute_labels_spatial(cortex_vertices, likeli, neighborhood, prams, likeli_pos, regularize)
        
        # if len(np.unique(label)) <= 2 or new_energy < 0:
        #     regularize = True
        #     print('Do regularize')
        # else: 
        #     regularize = False
        print(f'unique value of label spatial: {np.unique(label)}')
        
        new_energy -= max_max_likeli * cortex_vertices
        E_current_D -= max_max_likeli * cortex_vertices
        
        if new_energy < old_energy:
            print('it is reduced')
        elif new_energy == old_energy:
            print('same')
        else:
            print('increased')
            
        prams['fileID'].write(f"Improvement after {j} iterations: {(old_energy / new_energy - 1) * 100:.2f}%\n")
        prams['fileID'].write(f"Smoothcost: {E_current_S:.2f}, DataCost: {E_current_D:.2f}, Energy: {new_energy:.2f}\n")

        print(f"Eold: {old_energy}, Enew: {new_energy}, Abs Ratio: {abs(old_energy / new_energy - 1):.2f}")
        if len(np.unique(label)) < prams['cluster']:
            prams['fileID'].write(f'empty cluster \n')
            label, gamma = CBIG_assign_empty_cluster(label, prams['cluster'], grad_matrix, gamma, prams)
            
        elif abs(old_energy/new_energy -1)*100 < termination:
            prams['fileID'].write(f'hit termination threshold of {termination}')
            break
        
        else:
            print(f'Missed termination threshold of {termination}')
        print(f'Iteration {j + 1} completed')
    likeli += torch.round(max_max_likeli).to(torch.int32).numpy() #max_max_likeli
    likeli_pos += torch.round(max_max_likeli_local).to(torch.int32).numpy() # max_max_likeli_local
    likelihood = torch.mean(likeli[label.T])
    likeli = torch.tensor(likeli)
    # likelihood = torch.mean(likeli.float()[torch.tensor(label, dtype=torch.int64)])
    
    idx = torch.arange(np.size(label))

    final_likeli = likeli[idx, label.astype(np.int64)]

    results = {
        'D': E_current_D,
        'S': E_current_S,
        'UnormalizedE': new_energy,
        'final_likeli': final_likeli,
        'kappa': kappa,
        'E': CBIG_ComputeNormalizedEnergy(final_likeli, E_current_S),
        'gamma': gamma,
        'likeli_pos': likeli_pos[label.astype(np.int64)]
    }
    
    results['full_label'] = torch.zeros(avg_mesh['vertices'].shape[0], dtype=torch.long)
    results['full_label'][avg_mesh['MARS_label'] == 2] = torch.tensor(label[:cortex_vertices], dtype=torch.long)
    import pyvista as pv

    # plotter = pv.Plotter(off_screen=True)
    # plotter.add_mesh(avg_mesh['vis_mesh_data'], scalars = label,  cmap="tab10", show_edges=True)
    # plotter.show(screenshot=f'mesh_visualiztion_{i}_th_iteration.png')
    return likelihood, results, label

def CBIG_initialize_cluster(k, d, x, grad):
    """
    Random initialize likelihood 
    This calculates initial kappa * miu_times_x which is U_global. This part is articulated as below
    we use the current estimates of {𝜇1:L, 𝜅1:L, 𝜈1:L} to estimate labels l1:N
    
    Args:
        k (int): Number of clusters we want to assign
        d (int): Dimensionality 
        x (torch.tensor): input MRI data of shape (n, features), where n is the number of vertices.
        grad (torch.tensor): gradient file including gradient between vertices

    Returns:
        likeli (torch.tensor): likelihood of each cluster 1 to L
    """
    grad = torch.mean(grad, dim = 0)  # shape of (6, 1172)
    # print(grad.shape)
    low_grad_idx = torch.where((grad < 0.05))[0]
    
    indices = np.random.choice(range(len(low_grad_idx)), size = k, replace = False) # pick 7 indices
    initial_assigned_idx = low_grad_idx[indices] # choose 7 indices which will be initially assigned
    
    mu_times_x = torch.zeros((len(initial_assigned_idx),x.shape[1])) ## Shape of (L, )
    for i, idx in enumerate(initial_assigned_idx):
        mu_times_x[i, :] = x[idx, :] ## All parts have negative values 
        # initial mu_times_x is technically similarity(correlation) between specific vertex and all other vertices 
    if 2000 < d < 10000:
        kappa = 1800
    elif d >= 10000:
        kappa = 12500
    else :
        kappa = 500
    # print(mu_times_x)
    ## Some mu_times_x have negative values, is it possible
    ## From here this is not a probability anymore
    likeli = kappa * mu_times_x ## Here kappa is initially scalar
    
    max_max_likeli = torch.max(likeli)
     
    likeli = likeli - max_max_likeli ## shape of (L, V) Why do this?
    # likeli is technically similarity(correlation) between specific vertex and all other vertices, but not probability any more
    return likeli.T, initial_assigned_idx, max_max_likeli

def CBIG_ComputeNormalizedEnergy(likeli, smooth_cost):
    """
    Compute normalized energy by summing likelihood and adding with smooth cost

    Args:
        likeli (torch.tensor): Shape of (N, L) which contains information about each vertex's label and its value could be 1 through L
        smooth_cost (int): _description_

    Returns:
        normalized_energy (int): _description_
    """
    datacost = likeli
    normalized_energy = torch.sum(datacost) + smooth_cost

    return normalized_energy


def CBIG_assign_empty_cluster(label, k, input_grad, gamma, prams):
    """
    Assign empty cluster to random vertices 

    Args:
        label (torch.tensor): Shape of (N, ) which contains information about each vertex's label and its value could be 1 through L
        k (int): number of clusters
        input_grad (torch.tensor): matrix from github
        gamma (torch.tensor): Concentration parameters of shape (L,). This parameter is used to compute label for spatial term. Gamma variables which is tau in the paper and its shape is (L, ) where L is the number of clutsers
        prams (dict): a dictionary containing various input parameters
    """
    grad = torch.mean(input_grad, dim = 0)
    low_grad_idx = torch.where(grad < 0.05)[0]
    empty = []
    # Loop for check empty cluster
    for i in range(k):
        idx = (label == i) # index for each label
        if sum(idx) == 0:
            empty.append(i) ## Check which labels are not assigned
    assigned_vertices = []
    if len(empty) != 0:
        assigned_vertices = np.random.choice(low_grad_idx, size = len(empty), replace=False)
        label[assigned_vertices] = torch.tensor(empty, dtype=torch.long)
    for i in empty:
        gamma[i] = prams['start_gamma']
        
    return label, gamma


def CBIG_get_position(prams, label, hemisphere, gamma, reg):
    """
    Compute the likelihood of the position after computing kappa, but it does not use kappa and separate from U_global. 
    This is for computing U_spatial which is using CBIG_compute_likelihood_given_concentration

    Args:
        prams (dict): a dictionary containing various input parameters
        label (torch.tensor): Shape of (N, ) which contains information about each vertex's label and its value could be 1 through L
        hemisphere (str): a string indicating hemisphere 'lh' or 'rh'
        gamma (torch.tensor): Concentration parameters of shape (L,). This parameter is used to compute label for spatial term. Gamma variables which is tau in the paper and its shape is (L, ) where L is the number of clutsers

    Returns:
        likeli: Likelihood toward each cluster with the shape of (N, L)
        max_max_likeli:  
    """
    avg_mesh, prams = CBIG_ReadNCAvgMesh(prams=prams, hemi= hemisphere, mesh_name=prams['fsaverage'] , surf_type='sphere', label = 'cortex')
    
    data = avg_mesh['vertices'][avg_mesh['mesh_data'][avg_mesh['MARS_label'] == 2], :] #(1172, 3) different from Matlab(3, 1172)
    data /= torch.sqrt(torch.sum(data ** 2, dim=1, keepdim=True))  # (1172, 3) Normalize each vertex
    likeli, max_max_likeli = CBIG_compute_likelihood_given_concentration(data, label, prams['cluster'], 3, gamma, reg)
    return likeli.T, max_max_likeli

def CBIG_compute_likelihood_given_concentration(x, label, k, d, gamma, reg):
    """
    Compute the log-likelihood given concentration parameters. Used only in CBIG_get_position function 
    This part is for actual computing U_spatial. Even though the paper talks about the parameter v, we do not define and initialize it in the function since v is mean spatial direction
    
    Args:
        x (torch.Tensor): input MRI spatial data of shape (n, 3), where n is the number of vertices.
        label (torch.Tensor): Shape of (N, ) which contains information about each vertex's label and its value could be 1 through L (interchangable with k)
        k (int): Number of clusters which is the same as L
        d (int): Dimensionality of the data.
        gamma (torch.Tensor): Concentration parameters of shape (L,). This parameter is used to compute label for spatial term. Gamma variables which is tau in the paper and its shape is (L, ) where L is the number of clutsers

    Returns:
        likeli (torch.Tensor): Log-likelihood values.
        max_max_likeli (float): Global offset of the likelihood for numerical stability.
    """
    n, features = x.shape
    binary_matrix = torch.zeros((n, k), dtype=torch.float32)
    binary_matrix[torch.arange(n), label] = 1  
        
    nu = (x.T @ binary_matrix)
    nu /= torch.sqrt(torch.sum(nu**2, dim=0, keepdim=True)) 
    nu = nu.T  

    gammas = torch.zeros_like(gamma, dtype=torch.float32)
    non_zero_gamma = gamma != 0
    gammas[non_zero_gamma] = CBIG_Cdln(gamma[non_zero_gamma], d) # compute von Mises-Fisher

    likeli = gammas[:, None] + gamma[:, None] * (nu @ x.T)  # Shape: (k, n)
    
    if likeli.shape != (k,n):
        raise ValueError(f'Should think about dimensionality')
    max_max_likeli = torch.max(likeli)
    likeli -= max_max_likeli
    
    if reg:
        print('Need regularization')
    
    return likeli, max_max_likeli

    
def CBIG_compute_likelihood(x, label, k, d):
    """
    Compute the log-likelihood for von Mises-Fisher distribution which can be applied for general terms both U_spatial and U_global. 
    Used only once in CBIG_gwMRF_graph_cut_clustering_split_standard function since that function is used in wrapper function iteratively. 
    This calculates the likelihood of kappa when we treat kappa as not vector

    Args:
        x (torch.Tensor): Data matrix of shape (V, V), where n is the number of vertices.
        label (torch.Tensor): Shape of (N, L) which contains information about each vertex's label and its value could be 1 through L
        k (int): Number of clusters.
        d (int): Dimensionality of the data.

    Returns:
        likeli (torch.Tensor): Log-likelihood values.
        max_max_likeli (float): Global offset of the likelihood for numerical stability.
        kappa (int): Concentration parameter for each cluster.
    """
    n = x.shape[0]  

    binary_matrix = torch.zeros((n, k), dtype=torch.float32,)
    binary_matrix[torch.arange(n), label - 1] = 1  # Adjust for 0-based indexing

    miu_times_x = torch.zeros((k, x.shape[1]), dtype=torch.float32)
    kappa = torch.zeros(k, dtype=torch.float32, device=x.device)

    for i in range(k):
        indexed_data = (label == (i + 1)).nonzero(as_tuple=True)[0]
        if len(indexed_data) == 0:
            continue
        cluster_mean = x[indexed_data].mean(dim=0)
        norm_of_miu = torch.norm(cluster_mean)
        miu_times_x[i, :] = cluster_mean / norm_of_miu

    r_bar = torch.sum(binary_matrix.T * (miu_times_x @ x.T)) / n

    kappa = CBIG_inv_Ad(d, r_bar.item())
    kappa = kappa * torch.ones(k, )
    cdln_kappa = CBIG_Cdln(kappa, d)
    likeli = cdln_kappa[:, None] + (kappa[:, None] * (miu_times_x @ x.T))

    max_max_likeli = torch.max(likeli)
    likeli -= max_max_likeli
    
    return likeli, max_max_likeli, kappa

def CBIG_compute_likelihood_vector_of_kappa(x, label, k, d, reg):
    """
    Compute the likelihood vector of kappa for von Mises-Fisher distribution. Used only once in CBIG_gwMRF_graph_cut_clustering_split_standard function since that function is used in wrapper function iteratively. 
    Since this is for U_global term, it does not require gamma. Even though the paper talks about the parameter v, we do not define that. Then we can initialize it as miu_times_x since miu is mean direction 
    Main problem
    Args:
        x (torch.tensor):  Data matrix of shape (V, V), where V is the number of vertices.
        label (torch.Tensor): Shape of (N, ) which contains information about each vertex's label and its value could be 1 through L
        k (int): Number of clusters which is also kappa
        d (int): Dimensionality.
        
    Returns:
        likeli (torch.tensor): Likelihood vector with the shape of (k, N)
        max_max_likeli (float): Maximum likelihood offset.
        kappa (torch.tensor): Concentration parameters for each cluster with the shape of (k, )
    """ 
    
    n_samples = len(label)
    mu_times_x = torch.zeros((k, n_samples), dtype=torch.float32)
    kappa = torch.zeros(k, dtype=torch.float32)
    
    if not isinstance(label, torch.Tensor):
        label = torch.tensor(label, dtype=torch.long)
    
    for i in range(k):
        indexed_data = torch.where(label == i)[0]
        norm_of_miu = torch.sqrt(torch.sum(x[indexed_data][:, indexed_data]))  ## indexing covariance matrix only for corresponding labels -> scalar, This refers to 
        ## extracting # of vertices corresponding to label i which representing correlation between indexed vertices and all other vertices. 
        ## Then take sum and divide by norm of its label 
        mu_times_x[i, :] = torch.sum(x[indexed_data, :], dim=0) / norm_of_miu 
        r_bar = norm_of_miu / len(indexed_data) ## for Mean direction which is mu in the paper
        if len(indexed_data) == 1 and r_bar >= 0.975:
            r_bar = 0.975
        kappa[i] = CBIG_inv_Ad(d, r_bar) # concentration parameter kappa for each cluster  

    likeli = CBIG_Cdln(kappa, d)[:, None] + (kappa[:, None] * mu_times_x) ## all negative values
    max_max_likeli = torch.max(likeli)
    likeli = likeli - max_max_likeli
    if reg:
        print('Need regularization')
    
    likeli = likeli.T
    return likeli, max_max_likeli, kappa


def CBIG_inv_Ad(D, rbar):
    """
    Compute the inverse of Ad for given D and rbar to calculate kappa. This is Z() term of von Mises Fisher distribution

    Args:
        D (int): Dimensionality.
        rbar (float): R-bar value.
    Returns:
        float: Out value.
    """
    return rbar*(D-rbar)/(1-rbar**2)
    
   
# def CBIG_Ad(kappa, D):
#     """
#     Compute Concentration parameter as besseli(D/2, kappa) / besseli(D/2-1, kappa) which is for inv_Ad and from supplement page 4

#     Args:
#         kappa (torch.Tensor): Kappa values.
#         D (int): Dimensionality.
        
#     Returns:
#         torch.tensor: Result of CBIG_Ad.
#     """
#     bessel_num = torch.tensor(iv(D / 2, kappa.numpy()), dtype=torch.float32) ## using scipy function for modified bessel 
#     bessel_den = torch.tensor(iv(D / 2 - 1, kappa.numpy()), dtype=torch.float32)
#     return bessel_num / bessel_den

def CBIG_Cdln(kappa, d, nGrids=1000):
    """
    Compute the logarithm of the partition of function of von Mises-Fisher distribution as function of kappa
    
    Args:
        kappa (torch.Tensor): Kappa values.
        d (int): Dimensionality.
        nGrids (int): Number of grid points for overflow computation.
    Returns:
        torch.Tensor: Log partition function.
    """
    kappa_np = kappa.numpy().flatten()
    sizek = kappa_np.shape 
    v = d / 2 - 1
    
    # Use the scaled Bessel function: log(besseli(v, kappa)) = log(ive(v, kappa)) + kappa
    out = v * np.log(kappa_np) - (np.log(ive(v, kappa_np)) + kappa_np)
    if d < 1000:
        k0 = 10
    elif d < 2000:
        k0 = 500
    elif d < 5000:
        k0 = 1600
    elif d < 30000:
        k0 = 1600  # (or adjust based on your needs)
    elif d < 40000:
        k0 = 12500
    elif d < 50000:
        k0 = 14400
    elif d < 160000:
        k0 = 51200
    elif d >= 300000:
        k0 = 102288
    else:
        k0 = 102288 
    fk0 = v * np.log(k0) - (np.log(ive(v, k0)) + k0)
    if np.isinf(fk0):
        k0 = 0.331 * d
        fk0 = v * np.log(k0) - (np.log(ive(v, k0)) + k0)
        
    maskof = np.logical_or(kappa_np > k0, np.isinf(out))

    if np.any(maskof): ## Kappa values higher than overflow
        kof = kappa_np[maskof] 
        ofintv = (kof - k0) / nGrids 
        tempcnt = np.arange(1, nGrids + 1) - 0.5
        ks = k0 + np.outer(ofintv, tempcnt) ## shape (6, 1000)
        CBIG_Adsum = np.sum(1 / (0.5 * (d - 1) / ks + np.sqrt(1 + (0.5 * (d - 1) / ks) ** 2)), axis=1)
        out[maskof] = fk0 - ofintv * CBIG_Adsum
        
    return torch.tensor(out.reshape(sizek), dtype=torch.float32)


def CBIG_build_sparse_neighborhood(avg_mesh, gradient, prams):
    """
    Constructing 
    
    Args:
        avg_mesh (dict): Preprocessed mesh data dictionary containing MARS_label 
        gradient (torch.tensor): _description_
        prams (dict): a dictionary containing various input parameters
        
    Returns:
        neighborhood (torch.tensor): _description_
    """
    mars_label = torch.tensor(avg_mesh['MARS_label'], dtype=torch.float32)
    vertex_nbors = torch.tensor(avg_mesh['vertexNbors'], dtype=torch.long)
    idx_cortex_vertices = (mars_label == 2).nonzero(as_tuple=True)[0]

    vertices = vertex_nbors.shape[1]
    gradient_data = nib.load(gradient).get_fdata()
    r = torch.arange(13, vertices + 1).repeat_interleave(6).numpy() - 1  # Convert to 0-based indexing

    c = vertex_nbors[:6, 12:].reshape(-1).numpy() - 1  # Convert to 0-based indexing

    values = np.ones_like(r, dtype=np.float32)

    neighborhood = coo_matrix((values, (r, c)), shape=(vertices, vertices))

    neighborhood = lil_matrix(neighborhood)

    for i in range(12):
        neighbors = vertex_nbors[:5, i] - 1  
        neighbors = neighbors[neighbors >= 0]  
        neighborhood[i, neighbors] = 1

    neighborhood = neighborhood[idx_cortex_vertices, :][:, idx_cortex_vertices]

    neighborhood = neighborhood.tocsr()

    return neighborhood

def CBIG_build_sparse_gradient(avg_mesh, gradient, prams):
    """
    Build a sparse gradient matrix for the first term (V_grad) using the mesh and
    border (gradient) data.
    
    Args:
        avg_mesh (dict): Preprocessed mesh data (output from CBIG_ReadNCAvgMesh).
            Must include:
                - 'MARS_label': a torch.tensor (shape [num_vertices]) indicating cortex membership.
                - 'vertices': a torch.tensor of vertex coordinates.
                - 'vertexNbors': a list (length=num_vertices) of numpy arrays with neighbor indices.
        gradient (str): Path to the gradient MAT file containing 'border_matrix'.
        prams (dict): Dictionary of parameters. Must include the key 'exponential'.
    
    Returns:
        neighborhood (scipy.sparse.coo_matrix): Sparse matrix of shape (num_vertices, num_vertices)
            representing the gradient-based neighborhood, restricted to cortex vertices.
    """
    mars_label = avg_mesh['MARS_label']  # shape: (V,) 
    vertex_nbors_list = avg_mesh['vertexNbors']  ## list for each vertices neighbors'. Mostly 6 neighbor
    num_vertices = len(vertex_nbors_list) # V
    
    max_neighbors = max(len(n) for n in vertex_nbors_list) # because some vertices have 5 nneighbors
    
    padded_vertex_nbors = [
        np.pad(n, (0, max_neighbors - len(n)), constant_values=-1)
        for n in vertex_nbors_list
    ] # put -1 value in neighbors if a vertex has only 5 neighbors
    vertex_nbors = torch.tensor(padded_vertex_nbors, dtype=torch.long) # shape of (V, 6)
    gradient_data = loadmat(gradient)['border_matrix']
    border_matrix = torch.tensor(gradient_data, dtype=torch.float32)
    
    threshold = 12 # only first 12 vertices have 5 neighbors
    
    if max_neighbors > threshold and num_vertices > threshold:
        r = (torch.arange(threshold, num_vertices) ).repeat_interleave(6).tolist()
        c = vertex_nbors[:6, threshold:].reshape(-1).tolist()
        stable_vals = CBIG_StableE(border_matrix[:, threshold:], prams['exponential'])
    else:
        r = torch.arange(0, num_vertices).repeat_interleave(6).tolist() ## From 0 to 32492 duplicate 6 for each index 
        c = vertex_nbors.reshape(-1).tolist() ## Flattening neighborlist
        stable_vals = CBIG_StableE(border_matrix, prams['exponential']) # why do StableE for border matrix? Border matrix has the shape of (6, 40962) It does not match to  
        ## StableE function is for V_grad but why used in this
    
    values = stable_vals.flatten().tolist()
    min_len = min(len(r), len(c), len(values))
    r, c, values = r[:min_len], c[:min_len], values[:min_len] ## Adjust to minimum length among row, column, 

    filtered = [(ri, ci, vi) for ri, ci, vi in zip(r, c, values) if ci >= 0]  ## Filtering for padded values
    if len(filtered) == 0:
        raise ValueError("No valid indices found after filtering negative indices.")
    r, c, values = zip(*filtered)
    r, c, values = list(r), list(c), list(values)
    
    neighborhood = coo_matrix((values, (r, c)), shape=(num_vertices, num_vertices))
    neighborhood = neighborhood.tolil() 
    
    for i in range(min(threshold, num_vertices)):
        neigh = vertex_nbors[i, :5]
        valid_neigh = neigh[neigh >= 0]
        new_vals = CBIG_StableE(border_matrix[:5, i], prams['exponential'])
        for j, col in enumerate(valid_neigh.tolist()):
            neighborhood[i, col] = new_vals[j].item()
    idx_cortex = (mars_label == 2).nonzero(as_tuple=True)[0].numpy()
    neighborhood = neighborhood[idx_cortex, :][:, idx_cortex]
    
    neighborhood = neighborhood.tocoo()
    return neighborhood

def CBIG_compute_labels(cortex_vertices, probs, neighborhood, prams):
    """
    Step 1 of MAP1
    Compute labels based on its vertices, neighborhood, and probabilities. This is for initial calculation of U_global term but it does not require kappa(should ask)
    This part is articulated as "we use the current estimates of {𝜇1:L, 𝜅1:L, 𝜈1:L} to estimate labels l1:N". 
    
    Args:
        cortex_vertices (int): the number of cortex vertices
        probs (torch.tensor): Likelihood about each vertex shape of (N, L)
        neighborhood (torch.tensor): Matrix representing neighborhood between vertices with the shape of (N, N)
        prams (dict): a dictionary containing various input parameters
    
    Returns:
        labels (torch.tensor): Shape of (N, ) which contains information about each vertex's label and its value could be 1 through L and it would be newly updated labels after graph cut algorithm
        new_energy (int): 
        D (int): Data cost calculated from graph cut
        S (int): Smooth cost calculated from graph cut
    """

    eps = np.finfo(float).eps
    N, L = probs.shape
    data_array = -((probs.numpy() + eps) / cortex_vertices) + eps
    data_cost = (data_array * 100).round().astype(np.int32)
    smooth_cost = np.ones((L, L), dtype=np.float32)
    np.fill_diagonal(smooth_cost, 0)
    pair_wise = (smooth_cost * (prams['smoothcost'] * (1.0 / cortex_vertices) * 100)).astype(np.int32)
    if isinstance(neighborhood, torch.Tensor):
        neighborhood = neighborhood.cpu().numpy()

    scale_factor = 1000
    if sp.isspmatrix_coo(neighborhood):
        row = neighborhood.row.astype(np.int32)
        col = neighborhood.col.astype(np.int32)
        data = neighborhood.data.astype(np.float32)

        edges = np.column_stack([row, col]).astype(np.int32)        
        edge_weights = (data * scale_factor).astype(np.int32).reshape(-1, 1)
        edges_full = np.hstack([edges, edge_weights])               
    else:
        raw_edges = np.nonzero(neighborhood)
        edges = np.column_stack(raw_edges).astype(np.int32)        
        data = neighborhood[raw_edges].astype(np.float32)
        edge_weights = (data * scale_factor).astype(np.int32).reshape(-1, 1)
        edges_full = np.hstack([edges, edge_weights])             

    edges_full = np.ascontiguousarray(edges_full, dtype=np.int32)
    data_cost = np.ascontiguousarray(data_cost, dtype=np.int32)
    pair_wise = np.ascontiguousarray(pair_wise, dtype=np.int32)
    n_iter = int(prams['graphCutIterations'])
    labels = pygco.cut_from_graph(edges_full, data_cost, pair_wise, n_iter, algorithm='expansion')

    D = data_cost[np.arange(N), labels].sum()

    S = 0
    for (node1, node2), w in zip(edges, edge_weights):
        if labels[node1] != labels[node2]:
            S += w[0]

    Enew = D + S

    Enew *= cortex_vertices
    D *= cortex_vertices
    S *= cortex_vertices
    return labels, float(Enew), float(D), float(S)

def CBIG_compute_labels_spatial(cortex_vertices, probs, neighborhood, prams, likeli_pos, reg):
    """
    Python version of CBIG_compute_labels_spatial. Uses PyGCO to perform a graph cut,
    combining 'probs' and 'likeli_pos' into a single unary cost.

    Args:
        cortex_vertices (int): Number of cortex vertices, e.g., 1172
        probs (np.ndarray or torch.Tensor): [N, L] base likelihood data
        neighborhood (coo_matrix or np.ndarray): adjacency; can be sparse or dense
        prams (dict):
        likeli_pos (np.ndarray or torch.Tensor): [N, L] or [L, N] positional likelihood

    Returns:
        labels (np.ndarray): shape [N, ], assigned cluster for each vertex
        Enew (float): final energy
        D (float): unary (data) cost
        S (float): smooth cost
    """
    probs = np.array(probs)
    likeli_pos = np.array(likeli_pos)
    eps = np.finfo(float).eps
    normalize = cortex_vertices  
    data_cost = ( -(1/normalize) * ((probs + eps)) + eps )
    pos_cost = ( -(1/normalize) * ((likeli_pos + eps)) + eps )
    combined_cost = data_cost + pos_cost 
    combined_cost = combined_cost.round().astype(np.int32)
    
    
    print(combined_cost)
    print(f'min of cost: {combined_cost.min()} and max of cost: {combined_cost.max()}')
    # if reg:
    #     combined_cost = rescale_costs(combined_cost, target_min=500, target_max=50000)
    
    L = prams['cluster']
    
    smooth_cost = np.ones((L, L), dtype=np.float32)
    np.fill_diagonal(smooth_cost, 0)
    
    smooth_cost *= prams['smoothcost'] * (1.0 / normalize) * 100
    smooth_cost = smooth_cost.round().astype(np.int32)
    scale_factor = 100
    
    if sp.isspmatrix_coo(neighborhood):
        row = neighborhood.row.astype(np.int32)
        col = neighborhood.col.astype(np.int32)
        data_vals = neighborhood.data.astype(np.float32)
    else:
        row, col = np.nonzero(neighborhood)
        data_vals = neighborhood[row, col].astype(np.float32)

    edges = np.column_stack([row, col]).astype(np.int32)
    edge_weights = (data_vals * scale_factor).round().astype(np.int32).reshape(-1, 1)
    edges_full = np.hstack([edges, edge_weights]) 
    
    combined_cost = np.ascontiguousarray(combined_cost, dtype=np.int32)
    smooth_cost = np.ascontiguousarray(smooth_cost, dtype=np.int32)
    edges_full = np.ascontiguousarray(edges_full, dtype=np.int32)

    n_iter = int(prams['graphCutIterations'])
    labels = pygco.cut_from_graph(edges_full, combined_cost, smooth_cost, n_iter, algorithm='expansion')
    # if prams.get('potts', False):
    #     labels = pygco.cut_from_graph(None, combined_cost, smooth_cost, n_iter, algorithm='expansion')
    # else:
    #     labels = pygco.cut_from_graph(edges_full, combined_cost, smooth_cost, n_iter, algorithm='expansion')

    N = combined_cost.shape[0]
    D = combined_cost[np.arange(N), labels].sum()

    S = 0
    for (n1, n2), w in zip(edges, edge_weights):
        if labels[n1] != labels[n2]:
            S += w[0]

    Enew = D + S

    Enew *= normalize
    D *= normalize
    S *= normalize

    return labels.astype(np.int64), float(Enew), float(D), float(S)

    
def CBIG_StableE(x,k):
    """
    Do make exponentialize for the first terms of the objective function which is V_grad
    
    Args:
        x (torch.tensor): input data
        k (int): tunable parameter

    Returns:
        x: x which are stablized and exponentialized according to its values
    """
    eps = torch.finfo(x.dtype).eps
    x = torch.clamp(x, 0, 1)
    x = CBIG_our_exp(x, k)

    condition1 = torch.isinf(x) & (x > 0)
    condition2 = (~torch.isreal(x)) & (torch.real(x) > 0)
    condition3 = torch.isinf(x) & (x < 0)
    condition4 = (~torch.isreal(x)) & (torch.real(x) < 0)
    
    x[condition1] = CBIG_our_exp(1 - eps, k)
    x[condition2] = CBIG_our_exp(1 - eps, k)
    x[condition3] = CBIG_our_exp(0 - eps, k)
    x[condition4] = CBIG_our_exp(0 + eps, k)

    return x

def CBIG_our_exp(p, k):
    """
    Args:
        p (int): gradient between two different vertices
        k (int): tunable parameter

    Returns:
        exponential term (int): return exponential term of V_grad
    """
    p = torch.tensor(p, dtype=torch.float32)
    k = torch.tensor(k, dtype=torch.float32)
    return torch.exp(-k * p) - torch.exp(-k)

def CBIG_build_LogOdds(lh_avg_mesh6,rh_avg_mesh6,prams):
    return 

from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

def CBIG_gwMRF_generate_components(lh_avg_mesh, rh_avg_mesh, lh_labels, rh_labels):
    """
    Computes connected components for a labeled vector.

    Args:
        lh_avg_mesh (dict): Left hemisphere mesh with 'vertices' and 'vertexNbors'.
        rh_avg_mesh (dict): Right hemisphere mesh with 'vertices' and 'vertexNbors'.
        lh_labels (torch.Tensor): Label vector for the left hemisphere.
        rh_labels (torch.Tensor): Label vector for the right hemisphere.

    Returns:
        tuple: (lh_ci, rh_ci, lh_sizes, rh_sizes)
            lh_ci: Connected components for the left hemisphere.
            rh_ci: Connected components for the right hemisphere.
            lh_sizes: Size of the corresponding components for the left hemisphere.
            rh_sizes: Size of the corresponding components for the right hemisphere.
    """
    def generate_components(avg_mesh, labels):
        n = avg_mesh['vertices'].shape[0]
        neighbors = min(len(nbors) for nbors in avg_mesh['vertexNbors'])  # usually 6
        
        b = np.zeros(n * neighbors, dtype=int)
        c = np.zeros(n * neighbors, dtype=int)
        d = np.zeros(n * neighbors, dtype=int)

        if isinstance(labels, torch.Tensor):
            labels_np = labels.numpy()
        else:
            labels_np = np.array(labels)
        
        for i in range(12):
            current_nbors = avg_mesh['vertexNbors'][i] 
            num = neighbors - 1 
            start = i * neighbors
            end = start + num
            
            b[start:end] = i
            c[start:end] = np.array(current_nbors)[:num]
            d[start:end] = (labels_np[np.array(current_nbors[:num])] == labels_np[i]).astype(int)
        
        for i in range(12, n):
            current_nbors = avg_mesh['vertexNbors'][i]
            num = neighbors  
            start = i * neighbors
            end = start + num
            
            b[start:end] = i
            c[start:end] = current_nbors[:num]
            d[start:end] = (labels_np[np.array(current_nbors[:num])] == labels_np[i]).astype(int)
        
        indices_to_remove = [i * neighbors for i in range(12)]
        b = np.delete(b, indices_to_remove)
        c = np.delete(c, indices_to_remove)
        d = np.delete(d, indices_to_remove)
        
        g = coo_matrix((d, (b, c)), shape=(n, n))
        num_components, comp_labels = connected_components(csgraph=g, directed=False, return_labels=True)
        comp_sizes = np.bincount(comp_labels)
        
        return comp_labels, comp_sizes
    lh_ci, lh_sizes = generate_components(lh_avg_mesh, lh_labels)
    rh_ci, rh_sizes = generate_components(rh_avg_mesh, rh_labels)

    return lh_ci, rh_ci, lh_sizes, rh_sizes
