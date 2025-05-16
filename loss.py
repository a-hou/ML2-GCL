import torch
import torch.nn.functional as F
from weight import convert_to_weight_matrix
import torch.nn as nn



def sim(z1: torch.Tensor, z2: torch.Tensor):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())



def contrastive_loss(z, neighbors, weights, device, tau=0.4):
    loss = torch.tensor(0.0, device=device, requires_grad=True)  
    #z = Projection(z)
    N = z.shape[0]
    z = z.to(device)
    f = lambda x: torch.exp(x / tau)
    refl_sim = f(sim(z, z))
    weight_matrix = convert_to_weight_matrix(weights, neighbors, N, device)
    pos_refl_sim = refl_sim * weight_matrix
    #print(pos_refl_sim.sum(1))
    neg_weight_matrix = (weight_matrix == 0).float()
    row_sums = torch.sum(neg_weight_matrix, dim=1)
    normalized_matrix = neg_weight_matrix / row_sums[:, None]
    neg_refl_sim = refl_sim * normalized_matrix
    epsilon = 1e-8
    pos_refl_sim_sum = pos_refl_sim.sum(1) + epsilon
    neg_refl_sim_sum = neg_refl_sim.sum(1) + epsilon
    return -torch.log(pos_refl_sim_sum / (pos_refl_sim_sum + neg_refl_sim_sum)).mean()
