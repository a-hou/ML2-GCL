import torch
import torch.nn.functional as F

def get_neighbors_and_weights(z, edge_index, device, epsilon=1e-5):
    N = z.size(0)
    neighbors = {}  
    weights = {} 

    for i in range(N):
        neighbor_indices = edge_index[1][edge_index[0] == i].tolist()

        if not neighbor_indices:  
            neighbors[i] = torch.tensor([], device=device)
            weights[i] = torch.tensor([], device=device)
            continue

        neighbors[i] = torch.tensor(neighbor_indices, device=device)

        x_i = z[i]
        x_neighbors = z[neighbor_indices]

        G = x_neighbors @ x_neighbors.T  # Gram 矩阵
        b = x_neighbors @ x_i.unsqueeze(-1)
        G = G.to(device)
        b = b.to(device)

        eye_matrix = torch.eye(G.size(0), device=device)
        w = torch.linalg.solve(G + epsilon * eye_matrix, b)

        w = F.relu(w).squeeze()
        if w.sum() > 0:
            w = w / w.sum()
        else:
            w = torch.ones(len(neighbor_indices), device=device) / len(neighbor_indices)

        weights[i] = w

    return neighbors, weights




def convert_to_weight_matrix(weights_dict, neighbors_dict, N, device):
    """
    Convert weights and neighbor information into a weighted adjacency matrix format.
    :param weights_dict: Dictionary of weights for each node
    :param neighbors_dict: Dictionary of neighbors for each node
    :param N: Total number of nodes
    :param device: Device information (e.g., CPU or GPU)
    :return: Weight matrix of shape (N, N)
    """
    weight_matrix = torch.zeros((N, N), device=device)  

    for i, neighbors in neighbors_dict.items():
        if len(neighbors) > 0:  
            weights = weights_dict[i]
            weight_matrix[i, neighbors] = weights  

    return weight_matrix
