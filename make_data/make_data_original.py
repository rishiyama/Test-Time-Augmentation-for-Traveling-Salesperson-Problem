import torch

####
# set random seed
####

torch.manual_seed(12345)
batch_size = 10000
nb_nodes = 50


for d in [32]:
    x = torch.rand(batch_size, nb_nodes, d)
    torch.save(x, f'./data_additional/10k_TSP{nb_nodes}_d{d}.pt')
    
    
    # dist_mat = torch.cdist(x, x, p=2)