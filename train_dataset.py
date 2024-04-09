import time
import argparse
import os
import datetime
import torch
import torch.nn as nn
from models.tsp_transformer_tta import TSP_net
from utils.tsp_utils import *


#########################
import random
import numpy as np
seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
#########################

###################
# Instantiate a training network and a baseline network
###################


def main(args):
    
    model_train = TSP_net(args.embedding, args.nb_nodes, args.input_pe, args.dim_input_nodes, args.dim_emb, args.dim_ff, 
                args.nb_layers_encoder, args.nb_layers_decoder, args.nb_heads, args.max_len_PE,
                batchnorm=args.batchnorm)

    model_baseline = TSP_net(args.embedding, args.nb_nodes,  args.input_pe, args.dim_input_nodes, args.dim_emb, args.dim_ff, 
                args.nb_layers_encoder, args.nb_layers_decoder, args.nb_heads, args.max_len_PE,
                batchnorm=args.batchnorm)
    
    # uncomment these lines if trained with multiple GPUs
    print(torch.cuda.device_count())
    if torch.cuda.device_count()>1:
        model_train = nn.DataParallel(model_train)
        model_baseline = nn.DataParallel(model_baseline)
    # uncomment these lines if trained with multiple GPUs

    optimizer = torch.optim.Adam( model_train.parameters() , lr = args.lr ) 

    model_train = model_train.to(device)
    model_baseline = model_baseline.to(device)
    model_baseline.eval()

    print(args); print('')

    # Logs
    logdir_name = 'logs_100epoch'
    # os.system("mkdir logs")
    os.system("mkdir {}".format(logdir_name))
    time_stamp=datetime.datetime.now().strftime("%y-%m-%d--%H-%M-%S")
    file_name = logdir_name +'/'+time_stamp + "-n{}".format(args.nb_nodes) + "-gpu{}".format(args.gpu_id) + ".txt"
    # file_name = 'logs'+'/'+time_stamp + input("Enter file name: ") + "-n{}".format(args.nb_nodes) + "-gpu{}".format(args.gpu_id) + ".txt"
    experiment_status = "n{}".format(args.nb_nodes) + "-gpu{}".format(args.gpu_id) + f"-{args.data_type}-{args.model_type}-{args.param_type}" 
    file_name = logdir_name +'/'+ experiment_status + ".txt"
    
    
    file = open(file_name,"w",1) 
    file.write(time_stamp+'\n\n') 
    for arg in vars(args):
        file.write(arg)
        hyper_param_val="={}".format(getattr(args, arg))
        file.write(hyper_param_val)
        file.write('\n')
    file.write('\n\n') 
    plot_performance_train = []
    plot_performance_baseline = []
    all_strings = []
    epoch_ckpt = 0
    tot_time_ckpt = 0

    ###################
    # Main training loop 
    ###################
    start_training_time = time.time()

    for epoch in range(0,args.nb_epochs):
        
        # re-start training with saved checkpoint
        epoch += epoch_ckpt

        ###################
        # Train model for one epoch
        ###################
        start = time.time()
        model_train.train() 
        
        train_loss = 0
        train_L_train = 0
        train_L_baseline = 0

        for step in range(1,args.nb_batch_per_epoch+1):    
            if args.data_type == 'default':
                x = torch.rand(args.bsz, args.nb_nodes, args.dim_input_nodes, device=device)
            else:
                batch = get_one_batch(train_data, args.bsz)
                x = batch[args.data_type].to(device)


            # compute tours for model
            tour_train, sumLogProbOfActions = model_train(x, deterministic=False) # size(tour_train)=(bsz, nb_nodes), size(sumLogProbOfActions)=(bsz)
        
            # compute tours for baseline
            with torch.no_grad():
                tour_baseline, _ = model_baseline(x, deterministic=True)

            # get the lengths of the tours
            L_train = compute_tour_length(x, tour_train) # size(L_train)=(bsz)
            L_baseline = compute_tour_length(x, tour_baseline) # size(L_baseline)=(bsz)
            
            # backprop
            loss = torch.mean( (L_train - L_baseline)* sumLogProbOfActions )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss +=  loss
            train_L_train += torch.mean(L_train).item()
            train_L_baseline += torch.mean(L_baseline).item()
            
            print(f"({step}/{args.nb_batch_per_epoch+1}) loss : {loss}, L_train : {torch.mean(L_train).item()}, L_baseline : {torch.mean(L_baseline).item()}")
            
        time_one_epoch = time.time()-start
        time_tot = time.time()-start_training_time + tot_time_ckpt
        
        train_loss = train_loss / args.nb_batch_per_epoch
        train_L_train = train_L_train /  args.nb_batch_per_epoch
        train_L_baseline = train_L_baseline / args.nb_batch_per_epoch

        ###################
        # Evaluate train model and baseline on 10k random TSP instances
        ###################
        model_train.eval()
        mean_tour_length_train = 0
        mean_tour_length_baseline = 0
        for step in range(0,args.nb_batch_eval):
            
            if args.data_type == 'default':
                x = torch.rand(args.bsz, args.nb_nodes, args.dim_input_nodes, device=device) 
            else:
                batch = get_one_batch(val_data, args.bsz)
                x = batch[args.data_type].to(device)

            # compute tour for model and baseline
            with torch.no_grad():
                tour_train, _ = model_train(x, deterministic=True)
                tour_baseline, _ = model_baseline(x, deterministic=True)
                
            # get the lengths of the tours
            L_train = compute_tour_length(x, tour_train)
            L_baseline = compute_tour_length(x, tour_baseline)

            # L_tr and L_bl are tensors of shape (bsz,). Compute the mean tour length
            mean_tour_length_train += L_train.mean().item()
            mean_tour_length_baseline += L_baseline.mean().item()

        mean_tour_length_train =  mean_tour_length_train/ args.nb_batch_eval
        mean_tour_length_baseline =  mean_tour_length_baseline/ args.nb_batch_eval

        # evaluate train model and baseline and update if train model is better
        update_baseline = mean_tour_length_train+args.tol < mean_tour_length_baseline
        if update_baseline:
            model_baseline.load_state_dict( model_train.state_dict() )

        # Compute TSPs for small test set
        # Note : this can be removed
        with torch.no_grad():
            tour_baseline, _ = model_baseline(x_1000tsp, deterministic=True)
        mean_tour_length_test = compute_tour_length(x_1000tsp, tour_baseline).mean().item()
        
        # For checkpoint
        plot_performance_train.append([ (epoch+1), mean_tour_length_train])
        plot_performance_baseline.append([ (epoch+1), mean_tour_length_baseline])
            
        # Compute optimality gap
        if args.nb_nodes==50: gap_train = mean_tour_length_train/5.692- 1.0
        elif args.nb_nodes==100: gap_train = mean_tour_length_train/7.765- 1.0
        else: gap_train = -1.0
        
        # Print and save in txt file
        mystring_min = 'Epoch: {:d}, epoch time: {:.3f}min, tot time: {:.3f}day, L_train: {:.3f}, L_base: {:.3f}, L_test: {:.3f}, gap_train(%): {:.3f}, train_loss: {:f}, train_L_train: {:f}, train_L_base: {:f}, update: {}'.format(
            epoch, time_one_epoch/60, time_tot/86400, mean_tour_length_train, mean_tour_length_baseline, mean_tour_length_test, 100*gap_train, train_loss, train_L_train, train_L_baseline, update_baseline) 
        print(mystring_min) # Comment if plot display
        file.write(mystring_min+'\n')
    #     all_strings.append(mystring_min) # Uncomment if plot display
    #     for string in all_strings: 
    #         print(string)
        
        # Saving checkpoint
        # checkpoint_dir = os.path.join("checkpoint")
        checkpoint_dir = os.path.join("checkpoint_epoch100")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        torch.save({
            'epoch': epoch,
            'time': time_one_epoch,
            'tot_time': time_tot,
            'loss': loss.item(),
            'TSP_length': [torch.mean(L_train).item(), torch.mean(L_baseline).item(), mean_tour_length_test],
            'plot_performance_train': plot_performance_train,
            'plot_performance_baseline': plot_performance_baseline,
            'mean_tour_length_test': mean_tour_length_test,
            'model_baseline': model_baseline.state_dict(),
            'model_train': model_train.state_dict(),
            'optimizer': optimizer.state_dict(),
            }, '{}.pkl'.format(checkpoint_dir + "/checkpoint_" + experiment_status))

        
def get_args():
    parser = argparse.ArgumentParser(description='tsp_experiment_script')
    
    parser.add_argument('--model_type', required=True, choices=['transformer', 'transformer_pe', 'transformer_matrix', 'transformer_matrix_pe'], type=str, help='model type' )
    parser.add_argument('--data_type', required=True, choices=['x', 'x_double_tree'], type=str, help='data type')
    parser.add_argument('--param_type', required=True, choices=['default', 'dim102'], type=str, help='param type')
    parser.add_argument('--nb_nodes', required=True, choices=[20, 50, 100], type=int, help='Interger type')
    
    input_args = parser.parse_args()
    return input_args

if __name__ == '__main__':

    ###################
    # Hyper-parameters
    ###################

    class DotDict(dict):
        def __init__(self, **kwds):
            self.update(kwds)
            self.__dict__ = self
            
    args = DotDict()
    
    """
    args.model_type = 'transformer'
    # args.data_type = 'x_double_tree'
    args.data_type = 'x'
    args.param_type = 'default'
    """
    input_args = get_args()
    args.model_type = input_args.model_type
    
    if args.model_type == 'transformer':
        args.embedding = 'dim_input_nodes'
        args.input_pe = False
    elif args.model_type == 'transformer_pe':
        args.embedding = 'dim_input_nodes'
        args.input_pe = True
    elif  args.model_type == 'transformer_matrix':
        args.embedding = 'nb_nodes'
        args.input_pe = False
    elif args.model_type == 'transformer_matrix_pe':
        args.embedding = 'nb_nodes'
        args.input_pe = True
    
    args.data_type = input_args.data_type
    args.param_type = input_args.param_type
    args.nb_nodes = input_args.nb_nodes
    
    device = torch.device("cpu"); gpu_id = -1 # select CPU
    gpu_id = '0'
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('GPU name: {:s}, gpu_id: {:s}'.format(torch.cuda.get_device_name(0),gpu_id))   
    print(device)
    
    # args.nb_nodes = 20 # TSP20
    # args.nb_nodes = 50 # TSP50
    # args.nb_nodes = 100 # TSP100

    
    args.bsz = 512
    # args.bsz = 16
    args.nb_epochs = 100
    args.nb_batch_per_epoch = 2500
    
    args.nb_epochs = 5
    args.nb_batch_per_epoch = 5
    
    if args.param_type == 'dim102':
        args.dim_emb = 51 * 2 #128
        args.dim_ff = 512
        args.dim_input_nodes = 2
        args.nb_layers_encoder = 6
        args.nb_layers_decoder = 2
        args.nb_heads = 6 #8
        args.nb_batch_eval = 20
        args.gpu_id = gpu_id
        args.lr = 1e-4
        args.tol = 1e-3
        args.batchnorm = True  # if batchnorm=True  than batch norm is used
        #args.batchnorm = False # if batchnorm=False than layer norm is used
        args.max_len_PE = 1000
        
    if args.param_type == 'default':
        args.dim_emb =128
        args.dim_ff = 512
        args.dim_input_nodes = 2
        args.nb_layers_encoder = 6
        args.nb_layers_decoder = 2
        args.nb_heads = 8
        args.nb_batch_eval = 20
        args.gpu_id = gpu_id
        args.lr = 1e-4
        args.tol = 1e-3
        args.batchnorm = True  # if batchnorm=True  than batch norm is used
        #args.batchnorm = False # if batchnorm=False than layer norm is used
        args.max_len_PE = 1000
    
    
    if args.data_type == 'default':
        x_1000tsp = torch.rand(1000, args.nb_nodes, args.dim_input_nodes, device=device)
        train_data = None
        val_data = None
        test_data = None
    
    else:
        train_data = torch.load(f"data_dist/100000tsp{args.nb_nodes}seed0.pkl")
        print('train data load complete')
        
        val_data = torch.load(f"data_dist/1000tsp{args.nb_nodes}seed2222.pkl")
        print('val data load complete')
        
        test_data = torch.load(f"data_dist/1000tsp{args.nb_nodes}seed1111.pkl")
        print('test data load complete')
        
        
        x_1000tsp = test_data[args.data_type].to(device)
        
    print(args)
    main(args)