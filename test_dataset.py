import torch
import torch.nn as nn
import time
import argparse

import os
import datetime

from torch.distributions.categorical import Categorical

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try: 
    import networkx as nx
    from scipy.spatial.distance import pdist, squareform
    from concorde.tsp import TSPSolver # !pip install -e pyconcorde
except:
    pass
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import random
seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device = torch.device("cpu"); gpu_id = -1 # select CPU

gpu_id = '0' # select a single GPU  
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('GPU name: {:s}, gpu_id: {:s}'.format(torch.cuda.get_device_name(0),gpu_id))   
    
print(device)



def get_args():
    parser = argparse.ArgumentParser(description='tsp_experiment_script')
    
    parser.add_argument('--check_point', required=True, type=str, help='check_point_log' )
    parser.add_argument('--data_type', required=True, choices=['x', 'x_double_tree'], type=str, help='data type')
    # parser.add_argument('--param_type', required=True, choices=['default', 'dim102'], type=str, help='param type')
    parser.add_argument('--nb_nodes', required=True, choices=[20, 50, 100], type=int, help='nb_nodes')
    
    #  B = 100; args.bsz = 250; greedy = False; beamsearch = True 
    parser.add_argument('--beam_width', required=True, choices=[1,2,4,8,16,32,64,128,256,512,1024,2048, 100, 1000, 2500], type=int, help='beamsize width')
    parser.add_argument('--batch_size', required=True, choices=[1, 10, 25, 100, 250, 512, 10000], type=int, help='batch size')
    
    parser.add_argument('--greedy', action='store_true', help='greedy')
    parser.add_argument('--beamsearch', action='store_true', help='beamsearch')
    
    input_args = parser.parse_args()
    
    assert input_args.greedy and (not input_args.beamsearch) or  (not input_args.greedy) and input_args.beamsearch

    return input_args

###################
# Hyper-parameters
###################

def set_param_args(param_type):
    class DotDict(dict):
        def __init__(self, **kwds):
            self.update(kwds)
            self.__dict__ = self


            
    args = DotDict()

    args.nb_nodes = 20 # TSP20
    args.nb_nodes = 50 # TSP50
    # args.nb_nodes = 100 # TSP100
    args.bsz = 512 # TSP20 TSP50
    args.gpu_id = 0

    args.nb_epochs = 10000
    args.nb_batch_per_epoch = 2500
    args.nb_batch_eval = 20
    args.lr = 1e-4
    args.tol = 1e-3
    args.batchnorm = True  # if batchnorm=True  than batch norm is used
    #args.batchnorm = False # if batchnorm=False than layer norm is used
    args.max_len_PE = 1000

    if param_type == 'default':
        args.dim_emb = 128
        args.dim_ff = 512
        args.dim_input_nodes = 2
        args.nb_layers_encoder = 6
        args.nb_layers_decoder = 2
        args.nb_heads = 8
    else:
        args.dim_emb = 102 #128
        args.dim_ff = 512
        args.dim_input_nodes = 2
        args.nb_layers_encoder = 6
        args.nb_layers_decoder = 2
        args.nb_heads = 6 # 8
        
    return args
        

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

def model_select(model_name='transformer'):
    
    
    if model_name == 'transformer':
        from models_test.tsp_transformer import TSP_net
    elif model_name == 'transformer_pe':
        from models_test.tsp_transformer_penc import TSP_net
    elif model_name == 'transformer_matrix':
        from models_test.tsp_transformer_matrix import TSP_net
    elif model_name == 'transformer_matrix_pe':
        from models_test.tsp_transformer_matrix_penc import TSP_net
    else:
        raise Exception('Unknown model name')
    
    model_baseline = TSP_net(args.dim_input_nodes, args.dim_emb, args.dim_ff, 
              args.nb_layers_encoder, args.nb_layers_decoder, args.nb_heads, args.max_len_PE,
              batchnorm=args.batchnorm)
    
    model_baseline = model_baseline.to(device)
    model_baseline.eval()
    
    return model_baseline
    
def model_load(checkpoint_file=None):
    
    assert checkpoint_file is not None
    
    checkpoint = torch.load(checkpoint_file, map_location=device)
    epoch_ckpt = checkpoint['epoch'] + 1
    tot_time_ckpt = checkpoint['tot_time']
    plot_performance_train = checkpoint['plot_performance_train']
    plot_performance_baseline = checkpoint['plot_performance_baseline']
    # model_baseline.load_state_dict(checkpoint['model_baseline'])

    # NOTE: remove pos_embedding from checkpoint
    checkpoint_model_baseline = checkpoint['model_baseline']
    checkpoint_model_baseline.pop("pos_embedding", None)
    checkpoint_model_baseline.pop("start_placeholder_coord", None)
    # model_baseline.load_state_dict(checkpoint['model_baseline'])
    model_baseline.load_state_dict(checkpoint_model_baseline)
    print('Load checkpoint file={:s}\n  Checkpoint at epoch= {:d} and time={:.3f}min\n'.format(checkpoint_file,epoch_ckpt-1,tot_time_ckpt/60))
    

    mystring_min = 'Epoch: {:d}, tot_time_ckpt: {:.3f}day, L_train: {:.3f}, L_base: {:.3f}\n'.format(
        epoch_ckpt, tot_time_ckpt/3660/24, plot_performance_train[-1][1], plot_performance_baseline[-1][1]) 
    print(mystring_min) 
    
    return model_baseline


def run_test(model_baseline, nb_nodes, bsz, _beam_width, _greedy, _beamsearch):
        
    ###################   
    # Hyper-parameter for beam search
    ###################
    args.nb_nodes = nb_nodes
    args.bsz = bsz
    B = _beam_width
    greedy = _greedy
    beamsearch = _beamsearch
    
    
    
    args.nb_batch_eval = (10000+args.bsz-1)// args.bsz
    
    print('nb_nodes: {}, bsz: {}, B: {}, nb_batch_eval: {}, tot TSPs: {}\n'.format(args.nb_nodes, args.bsz, B, args.nb_batch_eval, args.nb_batch_eval* args.bsz))


    ###################   
    # Test set
    ###################
    if args.nb_nodes == 50:
        x_10k = torch.load(f'data/{TEST_DATA}.pt').to(device)
        x_10k_len = torch.load('data/10k_TSP50_len.pt').to(device)
        L_concorde = x_10k_len.mean().item()
    # if args.nb_nodes == 100:
    #     x_10k = torch.load('data/10k_TSP100.pt').to(device)
    #     x_10k_len = torch.load('data/10k_TSP100_len.pt').to(device)
    #     L_concorde = x_10k_len.mean().item()
    nb_TSPs = args.nb_batch_eval* args.bsz


        

        
        
    ###################   
    # Run beam search
    ###################
    start = time.time()
    mean_tour_length_greedy = 0
    mean_tour_length_beamsearch = 0
    mean_scores_greedy = 0
    mean_scores_beamsearch = 0
    gap_greedy = 0
    gap_beamsearch = 0
    
    
    results_length = torch.Tensor([]).to(device)
    
    for step in range(0,args.nb_batch_eval):
        print('batch index: {}, tot_time: {:.3f}min'.format(step, (time.time()-start)/60))
        # extract a batch of test tsp instances 
        x = x_10k[step*args.bsz:(step+1)*args.bsz,:,:]
        x_len_concorde = x_10k_len[step*args.bsz:(step+1)*args.bsz]
        # compute tour for model and baseline
        with torch.no_grad():
            tours_greedy, tours_beamsearch, scores_greedy, scores_beamsearch = model_baseline(x, B, greedy, beamsearch)
            # greedy
            if greedy:
                L_greedy = compute_tour_length(x, tours_greedy)
                mean_tour_length_greedy += L_greedy.mean().item()  
                mean_scores_greedy += scores_greedy.mean().item()  
                x_len_greedy = L_greedy
                gap_greedy += (x_len_greedy/ x_len_concorde - 1.0).sum()   
                # results_length.append(x_len_greedy)
                results_length = torch.cat((results_length, x_len_greedy), 0)
            # beamsearch
            if beamsearch:
                tours_beamsearch = tours_beamsearch.view(args.bsz*B, args.nb_nodes)
                x = x.repeat_interleave(B,dim=0)
                L_beamsearch = compute_tour_length(x, tours_beamsearch)
                tours_beamsearch = tours_beamsearch.view(args.bsz, B, args.nb_nodes)
                L_beamsearch = L_beamsearch.view(args.bsz, B)
                L_beamsearch_tmp = L_beamsearch
                L_beamsearch, idx_min = L_beamsearch.min(dim=1)
                mean_tour_length_beamsearch += L_beamsearch.mean().item()
                mean_scores_beamsearch += scores_beamsearch.mean().item()  
                x_len_beamsearch = L_beamsearch
                gap_beamsearch += (x_len_beamsearch/ x_len_concorde - 1.0).sum()
                
                # print('L_beamsearch_tmp: ', gap_beamsearch)
                # print(gap_beamsearch*100 /step)
        torch.cuda.empty_cache() # free GPU reserved memory 
        
        # break
    if greedy:
        mean_tour_length_greedy =  mean_tour_length_greedy/ args.nb_batch_eval
        mean_scores_greedy =  mean_scores_greedy/ args.nb_batch_eval
        gap_greedy = (gap_greedy/ nb_TSPs).item()
    if beamsearch:
        mean_tour_length_beamsearch =  mean_tour_length_beamsearch/ args.nb_batch_eval
        mean_scores_beamsearch =  mean_scores_beamsearch/ args.nb_batch_eval
        gap_beamsearch /= nb_TSPs
    tot_time = time.time()-start
        

        
    ###################   
    # Write result file
    ###################
    nb_TSPs = args.nb_batch_eval* args.bsz
    # file_name = "beamsearch-nb_nodes{}".format(args.nb_nodes) + "-nb_TSPs{}".format(nb_TSPs) + "-B{}".format(B) + ".txt"
    # file = open(file_name,"w",1) 
    mystring = '\nnb_nodes: {:d}, nb_TSPs: {:d}, B: {:d}, L_greedy: {:.6f}, L_concorde: {:.5f}, L_beamsearch: {:.5f}, \
    gap_greedy(%): {:.5f}, gap_beamsearch(%): {:.5f}, scores_greedy: {:.5f}, scores_beamsearch: {:.5f}, tot_time: {:.4f}min, \
    tot_time: {:.3f}hr, mean_time: {:.3f}sec'.format(args.nb_nodes, nb_TSPs, B, mean_tour_length_greedy, L_concorde, \
                                    mean_tour_length_beamsearch, 100*gap_greedy, 100*gap_beamsearch, mean_scores_greedy, \
                                    mean_scores_beamsearch, tot_time/60, tot_time/3600, tot_time/nb_TSPs)
    print(mystring)
    # file.write(mystring)
    # file.close()
    
    return results_length, mystring
    
if __name__ == "__main__":
    ###################
    # Hardware : CPU / GPU(s)
    ###################


    # INPUT
    input_args = get_args()
    
    CHECKPOINT_FILE = input_args.check_point
    CHECKPOINT_DIR, FILE_NAME = CHECKPOINT_FILE.split('/')
    PREFIX, GPUID, DATA_TYPE, MODEL_NAME, PARAM_TYPE = FILE_NAME.split('.')[0].split('-')

    # LOAD DEFAULT PARAMS
    args = set_param_args(param_type=PARAM_TYPE)

    # CHANGE PARAMS
    args.nb_nodes = input_args.nb_nodes
    B = input_args.beam_width
    args.bsz = input_args.batch_size
    greedy = input_args.greedy
    beamsearch = input_args.beamsearch
    
    if input_args.data_type == 'x':
        TEST_DATA = '10k_TSP50'
    elif input_args.data_type == 'x_double_tree':
        TEST_DATA = '10k_TSP50_double_tree'
    else:
        raise Exception('Unknown data type')
    
    print(f"B: {B}, greedy: {greedy}, beamsearch: {beamsearch}, batch_size: {args.bsz}, nb_nodes: {args.nb_nodes}, data_type: {TEST_DATA}, model_name: {MODEL_NAME}")
    
    
    model_baseline = model_select(model_name=MODEL_NAME)
    model_baseline = model_load(checkpoint_file=CHECKPOINT_FILE)


    results, result_text = run_test(model_baseline, nb_nodes=args.nb_nodes, bsz=args.bsz, _beam_width=B, _greedy=greedy, _beamsearch=beamsearch)
    length_data = results.cpu().numpy()
    print(length_data)
    
    
    # os.mkdir(f'results/{model_baseline}', exist_ok=True)
    os.makedirs(f'results_{CHECKPOINT_DIR}/{TEST_DATA}/{MODEL_NAME}_{DATA_TYPE}', exist_ok=True)
    
    if greedy:
        # np.save(f'results/{MODEL_NAME}/greedy.npy', length_data)
        torch.save(results, f'results_{CHECKPOINT_DIR}/{TEST_DATA}/{MODEL_NAME}_{DATA_TYPE}/greedy.pt')
        
        file = open(f'results_{CHECKPOINT_DIR}/{TEST_DATA}/{MODEL_NAME}_{DATA_TYPE}/greedy_result.txt',"w",1) 
        file.write(result_text)
        file.close()
        
    if beamsearch:
        # np.save(f'results/{MODEL_NAME}/beamsearch_{B}.npy', length_data)
        torch.save(results, f'results_{CHECKPOINT_DIR}/{TEST_DATA}/{MODEL_NAME}_{DATA_TYPE}/beamsearch_{B}.pt')
    
        file = open(f'results_{CHECKPOINT_DIR}/{TEST_DATA}/{MODEL_NAME}_{DATA_TYPE}/beamsearch_{B}_result.txt',"w",1) 
        file.write(result_text)
        file.close()