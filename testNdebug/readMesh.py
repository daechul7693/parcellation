import os
import numpy as np
import nibabel as nib
import torch
import pyvista as pv

import scipy.sparse

def CBIG_ReadNCAvgMesh(prams, hemi, mesh_name, surf_type = 'white', label = 'cortex'):
    """
    Read the average mesh (from FREESURFER) of normal control people

    Args:
        hemi (str): 'lh' or 'rh'
        mesh_name (str): 'fsaverage6', 'fsaverage5', 'fsaverage4'
        surf_type (str): 'inflated', 'white', 'sphere'
        label (str): 'cortex', 'Yeo2011_7Networks_N1000.annot'
        parms (dict): dictionary containing information about the data

    Returns:
        avg_mesh (dict) 
    """    
    
    subject_path = "data/downloaded/100610/hcp1200/fsaverage_LR32k/100610"
    lr = '.L.' if hemi == 'lh' else '.R.'
    final_path = subject_path + lr + surf_type +'.32k_fs_LR.surf.gii'
    label_path = subject_path + lr + 'BA.32k_fs_LR.label.gii'
    # print(label_path, final_path)
    gii_data = nib.load(final_path)
    label_gii = nib.load(label_path)
    # if gii_data is not None:
    #     print('gii data is successfully loaded')
    # if label_gii is not None:
    #     print('label is also loaded successfully')
    vertices = torch.tensor(gii_data.darrays[0].data)
    faces = torch.tensor(gii_data.darrays[1].data)
    mars_label = torch.tensor(label_gii.darrays[0].data)
    num_vertices = vertices.shape[0]
    neighbors_dict = compute_vertex_neighbors(faces, num_vertices)
    vertex_nbors = [neighbors_dict[i] for i in range(num_vertices)]
    vertexDistSq2Nbors = compute_vertex_dist_sq(vertices, neighbors_dict)
    
    faceAreas = compute_face_areas(vertices, faces)
    mesh_data = torch.arange(num_vertices)
    faces = np.hstack([np.full((faces.shape[0], 1),3), faces])
    vis_mesh_data = pv.PolyData(np.array(vertices), faces)

    avg_mesh = {'MARS_label': mars_label,
                'vertices': vertices,
                'faces': faces,
                'vertexNbors': vertex_nbors,
                'mesh_data': mesh_data,
                'vis_mesh_data': vis_mesh_data,
                'faceAreas': faceAreas,
                'vertexDistSq2Nbors': vertexDistSq2Nbors}
    if prams is not None:
        prams['hemi'] = hemi
        prams['subject'] = mesh_name   
        prams['read_surface'] = ...
        prams['raidus'] = 1000
        prams['unfoldBool'] = 0
        prams['flipFaceBool'] = 1
        prams['surf_filename'] = 'sphere'
        prams['metric_surf_filename'] = surf_type
    
    
    return avg_mesh, prams 


def compute_vertex_neighbors(faces, num_vertices):
    neighbors = {i: set() for i in range(num_vertices)}
    
    for face in faces:
        for i in range(3):
            v1 = face[i].item()
            v2 = face[(i+1)%3].item()
            v3 = face[(i+2)%3].item()
            neighbors[v1].update([v2, v3])
            
    neighbors = {i: np.array(list(neigh)) for i, neigh in neighbors.items()}
    return neighbors

def compute_vertex_dist_sq(vertices, neighbors):
    # Compute squared distances from each vertex to each of its neighbors.
    # Here, we store the results in a dictionary.
    dist_sq = {}
    for i, neigh in neighbors.items():
        dists = torch.sum((vertices[neigh] - vertices[i])**2, axis=1)
        dist_sq[i] = dists
    return dist_sq


def construct_adjacency_matrix(surface_gii):
    """
    Construct an adjacency (neighborhood) matrix from a GIFTI surface file.

    Parameters:
        surface_gii: GIFTI object containing surface mesh data

    Returns:
        adjacency_matrix (scipy.sparse.csr_matrix): Sparse adjacency matrix
    """
    vertices = surface_gii.darrays[0].data  
    faces = surface_gii.darrays[1].data    

    num_vertices = vertices.shape[0]
    
    adjacency_matrix = scipy.sparse.lil_matrix((num_vertices, num_vertices), dtype=np.int8)

    for face in faces:
        i, j, k = face
        adjacency_matrix[i, j] = 1
        adjacency_matrix[j, i] = 1
        adjacency_matrix[j, k] = 1
        adjacency_matrix[k, j] = 1
        adjacency_matrix[k, i] = 1
        adjacency_matrix[i, k] = 1

    return adjacency_matrix.tocsr()  

def compute_face_areas(vertices, faces):
    v = torch.tensor(vertices, dtype=torch.float32)
    f = torch.tensor(faces, dtype=torch.long)
    
    v0 = v[f[:, 0]]
    v1 = v[f[:, 1]]
    v2 = v[f[:, 2]]
    
    edge1 = v1 - v0
    edge2 = v2 - v0
    
    cross_prod = torch.cross(edge1, edge2, dim=1)
    face_areas = 0.5 * torch.norm(cross_prod, dim=1)
    return face_areas.numpy()