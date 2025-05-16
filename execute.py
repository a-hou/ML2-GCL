import os

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import process
from model import CombinedModel
import torch.optim as optim
import random

from process import align_weights
from loss import weighted_info_nce_loss
from logreg import LogReg
# 设置随机种子
seed = 1  # 你可以选择任何整数作为种子

# 设置 Python 内部的随机种子
random.seed(seed)

# 设置 numpy 的随机种子
np.random.seed(seed)

# 设置 PyTorch 的随机种子
torch.manual_seed(seed)

# 如果你使用 GPU (CUDA)，你也需要设置：
torch.cuda.manual_seed(seed)

dataset = 'cora'
#dataset = 'pubmed'
#dataset = 'citeseer'
# training params
hidden = 256
sparse = False
nonlinearity = 'prelu' # special name to separate parameters

adj, features, labels, idx_train, idx_val, idx_test = process.load_data(dataset)

#对特征进行行归一化
features, _ = process.preprocess_features(features)

nb_nodes = features.shape[0]
ft_size = features.shape[1]
nb_classes = labels.shape[1]

features = torch.FloatTensor(features[np.newaxis])
features = features.squeeze(0)  # 去掉第一个维度

labels = torch.FloatTensor(labels[np.newaxis])
idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

features = features.to(device)
edge_index = process.convert_adj_to_edge_index(adj).to(device)

W = torch.load('weights.pt')
# 对齐权重
weights = align_weights(edge_index, W, nb_nodes, device)  # (E,)

model = CombinedModel(ft_size, hidden, hidden).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-5)
# 训练循环

cnt_wait = 0
best_loss = 1e9
best_epoch = 0
num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    # 获取模型输出
    h = model(features, edge_index)  # 模型返回单一视图的表示
    
    # 计算损失
    loss = weighted_info_nce_loss(h, edge_index, weights, temperature=0.5)
    
    print('Loss:', loss)

    if loss < best_loss:
        best = loss
        best_t = epoch
        cnt_wait = 0
        torch.save(model.state_dict(), 'best_model.pkl')
    else:
        cnt_wait += 1

    if cnt_wait == 10:
        print('Early stopping!')
        break

    loss.backward(retain_graph=True)
    optimizer.step()

model.load_state_dict(torch.load('best_model.pkl'))

model.eval()
embeds = model(features, edge_index)  # 前向传播，获取嵌入

#embeds = F.normalize(embeds, p=2, dim=1)
embeds = embeds.detach().cpu()

train_embs = embeds[idx_train]
val_embs = embeds[idx_val]
test_embs = embeds[idx_test]

train_lbls = torch.argmax(labels[0, idx_train], dim=1)
val_lbls = torch.argmax(labels[0, idx_val], dim=1)
test_lbls = torch.argmax(labels[0, idx_test], dim=1)
xent = nn.CrossEntropyLoss()

tot = torch.zeros(1)
tot = tot.cuda()

accs = []

for _ in range(50):
    log = LogReg(hidden, nb_classes)
    log = log.cuda()  # Move the model to GPU

    train_embs = train_embs.cuda()  # Move embeddings to GPU
    val_embs = val_embs.cuda()      # Move embeddings to GPU
    test_embs = test_embs.cuda()    # Move embeddings to GPU
    train_lbls = train_lbls.cuda()  # Move labels to GPU
    test_lbls = test_lbls.cuda()    # Move test labels to GPU

    opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)

    pat_steps = 0
    best_acc = torch.zeros(1).cuda()

    for _ in range(100):
        log.train()
        opt.zero_grad()

        logits = log(train_embs)
        loss = xent(logits, train_lbls)
        
        loss.backward()
        opt.step()

    logits = log(test_embs)
    preds = torch.argmax(logits, dim=1)
    acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
    accs.append(acc * 100)
    print(acc)
    tot += acc

print('Average accuracy:', tot / 50)

accs = torch.stack(accs)
print(accs.mean())
print(accs.std())

'''
训练节点权重
'''

# # 将稀疏矩阵转换为密集矩阵
# adj_dense = adj.toarray()  # 转换为 NumPy 数组

# # 将密集矩阵转换为 PyTorch tensor
# Q_tensor = torch.tensor(adj_dense, dtype=torch.float32)

# features_tensor = features # 不需要梯度，避免重新创建 tensor

# # 权重初始化
# w = {}
# for i in range(Q_tensor.size(0)):
#     neighbors = torch.nonzero(Q_tensor[i]).view(-1)  # 找到节点 i 的邻居
#     # 直接创建 tensor，并且保持梯度
#     w[i] = torch.ones(len(neighbors), dtype=torch.float32, requires_grad=True, device=device)  
#     w[i] = w[i] / len(neighbors)  # 归一化，但不进行原地操作
#     w[i].retain_grad()  # 保持梯度

# # 只优化 w[i]，不优化 features_tensor
# optimizer = torch.optim.Adam([w[i] for i in w], lr=0.01)

# # 目标函数计算：L = ∑i || xi - Σj∈Q(i) w_ij xj ||_2^2
# def loss_function(features, Q_tensor, w):
#     loss = 0
#     for i in range(Q_tensor.size(0)):  # 遍历每个节点
#         x_i = features[i]  # 获取节点 i 的特征
#         neighbors = torch.nonzero(Q_tensor[i]).view(-1)  # 获取节点 i 的邻居节点
#         weighted_sum = torch.sum(w[i].view(-1, 1) * features[neighbors], dim=0)  # 计算加权邻居特征和
#         loss += torch.norm(x_i - weighted_sum, p=2) ** 2  # 计算 L2 范数的平方
#     return loss

# # 确保 features_tensor 和 Q_tensor 没有被提前分离计算图
# features_tensor = features_tensor.to(device)  # 确保 features_tensor 需要梯度
# Q_tensor = Q_tensor.to(device)  # 确保 Q_tensor 需要梯度

# # 训练过程
# num_epochs = 100
# for epoch in range(num_epochs):
#     optimizer.zero_grad()  # 清空梯度
#     loss = loss_function(features_tensor, Q_tensor, w)  # 计算损失
#     print(f"Epoch {epoch+1}, Loss: {loss.item()}")
    
#     # 反向传播，确保计算图正确
#     loss.backward(retain_graph=True)  # 反向传播，计算梯度
#     optimizer.step()  # 更新参数

# # 对每个节点的权重进行归一化，并限制范围 [0, 1]
# for i in w:
#     # 归一化：确保每个 w[i] 的和为 1
#     w[i] = w[i] / torch.sum(w[i])  # 归一化权重，使其和为 1
    
#     # 限制每个权重的值在 [0, 1] 范围内
#     w[i] = torch.clamp(w[i], min=0.0, max=1.0)

# # 输出优化后的权重
# for i in w:
#     print(f"Optimized weights for node {i}: {w[i].detach().cpu().numpy()}")

# torch.save(w, 'weights.pt')


