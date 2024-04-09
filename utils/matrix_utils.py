import torch
from scipy.spatial.distance import pdist, squareform
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from collections import deque
import copy

def translate_to_distance_matrix__(x):
    
    W_val = squareform(pdist(x, metric='euclidean'))
        
    return W_val


def translate_to_distance_matrix(x):
    
    '''example:
    >>> x = np.random.rand(128, 50, 2)
    >>> x.shape
    (128, 50, 2)
    >>> x = translate_to_distance_matrix(x)
    >>> x.shape
    (128, 50, 50)
    '''
    
    bsz = x.shape[0]
    nodes = x.shape[1]
    
    W_val = np.zeros([bsz, nodes, nodes])
    for i in range(bsz):
        W_val[i] = squareform(pdist(x[i], metric='euclidean'))
        
    return W_val

def simple_eulerian(_matrix): 
    matrix = _matrix.copy()
    matrix[matrix == 0] = np.nan

    source = 0
    vertex_stack = [source]
    last_vertex = None

    while vertex_stack:
        current_vertex = vertex_stack[-1]
        if np.nansum(matrix[current_vertex]) == 0:
            if last_vertex is not None:
                yield (last_vertex, current_vertex)
            last_vertex = current_vertex
            vertex_stack.pop()
        else:
            next_vertex = np.nanargmin(matrix[current_vertex]) 
            vertex_stack.append(next_vertex)
            matrix[current_vertex, next_vertex] = np.nan

def double_tree_tour(_matrix, add_inital_node=False):
    
    '''example:
    # matrix.shape = (N , N)
    >>> matrix = np.array([[0, 1, 2, 3],
                          [1, 0, 4, 5],
                          [2, 4, 0, 6],
                          [3, 5, 6, 0]])    
                          
    >>> tour  = double_tree_tour(md)
    >>> print(tour) # [0, 3, 2, 1, 0],
    '''
    
    # MST 
    _matrix = minimum_spanning_tree(_matrix).toarray()
    
    # double the edges
    _matrix = _matrix + _matrix.T

    
    tour = []  
    
    # edges = [] # this is the edges of eulerian tour
    for u, v in simple_eulerian(_matrix):
        # edges.append([u, v])
        if u not in tour:
            tour.append(u)
            
    # must be a tour
    assert len(tour) == len(_matrix)
    assert len(tour) == len(set(tour))
    
    if add_inital_node:
        tour.append(tour[0])
        
    
    return tour


def reordering_matrix(_matrix, method='double_tree'):
    
    '''example:
    
    >>> matrix = np.array([[0, 1, 2, 3],[1, 0, 4, 5],[2, 4, 0, 6],[3, 5, 6, 0]], dtype='float') 
    >>> matrix
    array([[0., 1., 2., 3.],
        [1., 0., 4., 5.],
        [2., 4., 0., 6.],
        [3., 5., 6., 0.]])
    >>> reordering_matrix(matrix)
    array([[0., 3., 2., 1.],
        [3., 0., 6., 5.],
        [2., 6., 0., 4.],
        [1., 5., 4., 0.]])
    '''
    
    
    if method == 'double_tree':
        tour = double_tree_tour(_matrix, add_inital_node=False)
        
    elif method == 'random':
        tour = list(range(_matrix.shape[0]))
        np.random.shuffle(tour)
        
    
    _reorder_mat = _matrix[tour][:, tour].copy()
    
    
    return _reorder_mat, tour

def reordering_matrix_batch(_matrix, method='double_tree'):
    
    bsz = _matrix.shape[0]
    nodes = _matrix.shape[1]
    
    np_matrix = np.zeros([bsz, nodes, nodes])
    np_tour = np.zeros([bsz, nodes])
    
    for i in range(bsz):
        np_matrix[i], np_tour[i] = reordering_matrix(_matrix[i], method=method)
        
        
    return np_matrix, np_tour