import torch
import numpy as np
from scipy.spatial.distance import pdist, squareform

def compute_tour_length(x, tour): 
    """
    Compute the length of a batch of tours
    Inputs : x of size (bsz, nb_nodes, 2) batch of tsp tour instances
             tour of size (bsz, nb_nodes) batch of sequences (node indices) of tsp tours
    Output : L of size (bsz,)             batch of lengths of each tsp tour
    """
    bsz = x.shape[0]
    nb_nodes = x.shape[1]
    arange_vec = torch.arange(bsz, device=x.device)
    first_cities = x[arange_vec, tour[:,0], :] # size(first_cities)=(bsz,2)
    previous_cities = first_cities
    L = torch.zeros(bsz, device=x.device)
    with torch.no_grad():
        for i in range(1,nb_nodes):
            current_cities = x[arange_vec, tour[:,i], :] 
            L += torch.sum( (current_cities - previous_cities)**2 , dim=1 )**0.5 # dist(current, previous node) 
            previous_cities = current_cities
        L += torch.sum( (current_cities - first_cities)**2 , dim=1 )**0.5 # dist(last, first node)  
    return L

def translate_to_distance_matrix(x):
    # size, bsz, nb_nodes, 2
    bsz = x.shape[0]
    nodes = x.shape[1]
    
    W_val = np.zeros([bsz, nodes, nodes])
    for i in range(bsz):
        W_val[i] = squareform(pdist(x[i], metric='euclidean'))
        
    return W_val

def translate_to_distance_matrix_tensor(x):
    # size, bsz, nb_nodes, 2
    bsz = x.shape[0]
    nodes = x.shape[1]
    
    W_val = torch.zeros([bsz, nodes, nodes])
    for i in range(bsz):
        W_val[i] = torch.Tensor(squareform(pdist(x[i], metric='euclidean')))
        
    return W_val



def reordering_sort_tour(x):
    
    indices = x[:, :, 0].sort()[1]
    reorder_x = x[torch.arange(x.size(0)).unsqueeze(1), indices]
    
    return reorder_x

"""
for i in range(10):
    coord, dist = generate_tsp_instance(1, 50, 2, torch.device('cuda'), reordering=True)
    
    print(coord)
    print(dist)
"""

def generate_tsp_instance(batch_size, nb_nodes, dim_input_nodes, device, reordering=False, distance_matrix=False):
    x = torch.rand(batch_size, nb_nodes, dim_input_nodes)
    
    if reordering:
        x = reordering_sort_tour(x)
    
    if distance_matrix:
        x_distance = translate_to_distance_matrix(x)
        x_distance = torch.Tensor(x_distance).to(device)
    
    else:
        x_distance = None
    
    x = x.to(device)
    return x, x_distance


def get_one_batch(data, batch_size):
    keys = list(data.keys())
    data_size = data[keys[0]].shape[0]
    idx = torch.randperm(data_size)[:batch_size]
    
    batch = {}
    for key in keys:
        
        d = { key : data[key][idx] }
        batch.update(d)
    
    return batch