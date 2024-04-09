from utils.matrix_utils import *
import numpy as np
import random
import torch
import os 
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='tsp_experiment_parser')
parser.add_argument('-s', '--seed', required=True, type=int, default=0)
parser.add_argument('-b', '--batch_size', required=True, type=int, default=10)
parser.add_argument('-t', '--tsp_size', required=True, type=int, default=50)

args = parser.parse_args()
    
SEED = args.seed
BATCH_SIZE = args.batch_size
TSP_SIZE = args.tsp_size
DIM_INPUT_NODES = 2
DIR = "data_dist_more"

is_save =  True
save_one_file = True

def graph(data, path, idx = 0): 
    import networkx as nx
    
    
    x = data['x'][idx].numpy()
    dist = data['distance_mat'][idx].numpy()
    rord = data['reordered_mat_double_tree'][idx]
    tour = data['tour_double_tree'][idx]
    # G = nx.Graph()
    
    G = nx.from_numpy_matrix(dist)
    
    # pos = {i: (a, b ) for i, (a, b) in enumerate(x)}
    pos = {i: (a, b ) for i, (a, b) in enumerate(x[tour])}
    # print(pos)
    # nx.draw(G, pos=pos)
    # # plt.show()
    # plt.savefig(path[:-4] + 'nx.png')
    
    
    # change order by tour
    # print(tour)
    
    # pos = {i: (a, b ) for i, (a, b) in enumerate(x[tour])}
    
    
    node_labels = {i: i for i in range(len(x))}
    
    edge_labels = {}
    
    for i in tour:
        for j in tour:
            if dist[i][j] != 0:
                edge_labels[(i, j)] = dist[i][j]
    
    
    # nx.draw(G, pos=pos, with_labels=True, labels=node_labels)
    nx.draw(G, pos=pos, with_labels=True, labels=node_labels)
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels)
    # plt.show()
    plt.savefig(path[:-4] + 'nx_ro.png')
    
    
def plot(data, path, idx = 0):
    # return graph(data, path, idx)
    x = data['x']
    dist_mat = data['distance_mat']
    rord_mat = data['reordered_mat_double_tree']
    tour = data['tour_double_tree']
    
    # dist_mat_gt = torch.zeros_like(dist_mat[idx])
    # dist_mat_gt[3][4] = dist_mat_gt[4][3] = 1
    # dist_mat_gt[30][45] = dist_mat_gt[45][30] = 1
    
    
    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(131)
    ax.set_title('TSP Distance Matrix - Original')
    ax.imshow(dist_mat[idx], cmap='gray')
    # ax.imshow(dist_mat_gt, cmap='gray_r')
    
    
    # road_mat_gt = dist_mat_gt[ tour[idx]][: , tour[idx]]
    
    print(x[idx])
    x =x[idx][tour[idx]] 
    
    print(x)
    
    x =x[tour[idx]] 
    
    print(x)
    
    x =x[tour[idx]] 
    
    print(x)
    
    
    x = translate_to_distance_matrix__(x)
    print(x)
    
    
    road_mat_gt = x
    ax = fig.add_subplot(132)
    ax.set_title('TSP Distance Matrix - reordered by double tree')
    # ax.imshow(rord_mat[idx], cmap='gray')
    ax.imshow(road_mat_gt, cmap='gray')
    
    
    
    rord_check_mat = rord_mat[idx]
    ax = fig.add_subplot(133)
    ax.set_title('TSP Distance Matrix - restoration')
    ax.imshow(rord_check_mat, cmap='gray')
    
    plt.savefig(path[:-4] + '.png')
    plt.close()
    

def load():
    path = os.path.join(DIR, str(BATCH_SIZE) + "tsp" + str(TSP_SIZE) + "seed" + str(SEED) + ".pkl")
    data = torch.load(path)
    print(data.keys())
    
    # plot(data, path)

def save():
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    
    # METHOD = 'random'
    METHOD = 'double_tree'
    print("REORDERING METHOD: ", METHOD)
    
    x = torch.rand(BATCH_SIZE, TSP_SIZE, DIM_INPUT_NODES)
    
    if METHOD == 'random':
        # x is all same value
        print(f"{METHOD} is used for One-Instance-Train")
        
        base = x[0]
        for i in range(BATCH_SIZE):
            x[i] = base
            
        
    
    distance_mat = translate_to_distance_matrix(x)
    # reordered_mat_double_tree, tour_double_tree = reordering_matrix_batch(distance_mat, method='double_tree')
    reordered_mat_double_tree, tour_double_tree = reordering_matrix_batch(distance_mat, method=METHOD)
    
    # x_double_tree = x[:,:,tour_double_tree]
    
    
    # np array to torch tensor
    distance_mat = torch.from_numpy(distance_mat)
    reordered_mat_double_tree = torch.from_numpy(reordered_mat_double_tree)
    tour_double_tree = torch.from_numpy(tour_double_tree).long()
    
    xr = torch.zeros_like(x)
    for i in range(BATCH_SIZE):
        xr[i] = x[i][tour_double_tree[i]]
        
        
        
    data_dir = os.path.join(DIR)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        

    
    if save_one_file:
        
        if METHOD == 'random':
            torch.save({ 'x': x, 'distance_mat': distance_mat, 'reordered_mat_random': reordered_mat_double_tree, 'tour_random': tour_double_tree, 'x_random' : xr}, '{}.pkl'.format(data_dir + "/" + str(BATCH_SIZE) + "tsp" + str(TSP_SIZE) + "seed" + str(SEED) + "_random"))
        else:
            torch.save({ 'x': x, 'distance_mat': distance_mat, 'reordered_mat_double_tree': reordered_mat_double_tree, 'tour_double_tree': tour_double_tree, 'x_double_tree' : xr}, '{}.pkl'.format(data_dir + "/" + str(BATCH_SIZE) + "tsp" + str(TSP_SIZE) + "seed" + str(SEED)))
    
    
    else:    
        torch.save({ 'x': x, }, '{}.pkl'.format(data_dir + "/" + str(BATCH_SIZE) + "tsp" + str(TSP_SIZE) + "seed" + str(SEED) + "_coord"))
        torch.save({ 'x': distance_mat, }, '{}.pkl'.format(data_dir + "/" + str(BATCH_SIZE) + "tsp" + str(TSP_SIZE) + "seed" + str(SEED) + "_dist" ))
        torch.save({ 'x': reordered_mat_double_tree, }, '{}.pkl'.format(data_dir + "/" + str(BATCH_SIZE) + "tsp" + str(TSP_SIZE) + "seed" + str(SEED) + "_dist_double_tree"))
        torch.save({ 'x': tour_double_tree, }, '{}.pkl'.format(data_dir + "/" + str(BATCH_SIZE) + "tsp" + str(TSP_SIZE) + "seed" + str(SEED) + "_tour_double_tree"))


if __name__ == '__main__':
    
    if is_save:
        save()
    else:
        print("data already exists")
        load()