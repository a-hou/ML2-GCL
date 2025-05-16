import numpy as np
import scipy.sparse as sp
import torch
import process
from model import CombinedModel
import torch.optim as optim
import random
from process import align_weights
from loss import contrastive_loss
import argparse
from sklearn.metrics import accuracy_score
from torch_geometric.utils import remove_self_loops, add_self_loops
from utils import load_network_data, random_planetoid_splits, get_train_data
from sklearn.linear_model import LogisticRegression
from weight import get_neighbors_and_weights, convert_to_weight_matrix
from sklearn.preprocessing import StandardScaler
import warnings
from scipy.sparse import csr_matrix
from torch_geometric.datasets import Planetoid


warnings.simplefilter(action='ignore', category=FutureWarning)


def train():
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # #adj, features, labels, idx_train, idx_val, idx_test = process.load_data(dataset)
    A , X , Y = load_network_data(args.dataset)
    c = Y.shape[1]
    lab = np.argmax(Y, 1)
    X[X > 0] = 1

    c = Y.shape[1]

    device = torch.device(f"cuda:{args.device}") if torch.cuda.is_available() else torch.device("cpu")

    X = torch.FloatTensor(X.todense()).to(device)

    edge_index = process.convert_adj_to_edge_index(A).to(device)

    ft_size = X.shape[1]
    nb_nodes = X.shape[0]
  
    neighbors, weights = get_neighbors_and_weights(X, edge_index, device, epsilon=args.epsilon)

    #W = convert_to_weight_matrix(weights, neighbors, nb_nodes, device)
    model = CombinedModel(ft_size, args.hidden, args.hidden, dropout=args.dropout).to(device)
    #model = CombinedModel(ft_size, args.hidden).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    cnt_wait = 0
    best_loss = 1e9
    best_epoch = 0
    num_epochs = args.nb_epochs

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        h = model(X, edge_index)  
        #loss = weighted_info_nce_loss(h, edge_index, weights, temperature=0.5)
        loss = contrastive_loss(h, neighbors, weights, device , args.tau)
        print('epoch:',epoch+1 , 'Loss:', loss)
        if loss < best_loss:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), args.best_model_path)
        else:
            cnt_wait += 1

        if cnt_wait == 10:
            print('Early stopping!')
            break

        loss.backward(retain_graph=True)
        optimizer.step()


        #     
        # if (epoch + 1) % 20 == 0:
        #     model.load_state_dict(torch.load(args.best_model_path))
        #     model.eval()
        #     embeds = model(X, edge_index)  

        #     embeds = embeds.detach().cpu()

        #     Accuaracy_test_allK = []
        #     numRandom = 20

        #     for train_num in [20]:
        #         AccuaracyAll = []
        #         for random_state in range(numRandom):
        #             print(
        #                 "\n=============================%d-th random split with training num %d============================="
        #                 % (random_state + 1, train_num))

        #             if train_num == 20:
        #                 if args.dataset in ['cora', 'citeseer', 'pubmed']:
        #                     # train_num per class: 20, val_num: 500, test: 1000
        #                     val_num = 500
        #                     idx_train, idx_val, idx_test = random_planetoid_splits(c, torch.tensor(lab), train_num,
        #                                                                             random_state)
        #                 else:
        #                     # Coauthor CS, Amazon Computers, Amazon Photo
        #                     # train_num per class: 20, val_num per class: 30, test: rest
        #                     val_num = 30
        #                     idx_train, idx_val, idx_test = get_train_data(Y, train_num, val_num, random_state)
        #             else:
        #                 val_num = 0  # do not use a validation set when the training labels are extremely limited
        #                 idx_train, idx_val, idx_test = get_train_data(Y, train_num, val_num, random_state)

        #             train_embs = embeds[idx_train, :]
        #             val_embs = embeds[idx_val, :]
        #             test_embs = embeds[idx_test, :]

        #             if train_num == 20:
        #                 # find the best parameter C using validation set
        #                 best_val_score = 0.0
        #                 for param in [1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100]:
        #                     LR = LogisticRegression(solver='liblinear', multi_class='ovr', random_state=0, C=param)
        #                     LR.fit(train_embs, lab[idx_train])
        #                     val_score = LR.score(val_embs, lab[idx_val])
        #                     if val_score > best_val_score:
        #                         best_val_score = val_score
        #                         best_parameters = {'C': param}

        #                 LR_best = LogisticRegression(solver='liblinear', multi_class='ovr', random_state=0, **best_parameters)
        #                 LR_best.fit(train_embs, lab[idx_train])
        #                 y_pred_test = LR_best.predict(test_embs)  # pred label
        #                 print("Best accuracy on validation set:{:.4f}".format(best_val_score))
        #                 print("Best parameters:{}".format(best_parameters))

        #             else:  # not use a validation set when the training labels are extremely limited
        #                 LR = LogisticRegression(solver='liblinear', multi_class='ovr', random_state=0)
        #                 LR.fit(train_embs, lab[idx_train])
        #                 y_pred_test = LR.predict(test_embs)  # pred label

        #             test_acc = accuracy_score(lab[idx_test], y_pred_test)
        #             print("test accuracy:{:.4f}".format(test_acc))
        #             AccuaracyAll.append(test_acc)

        #         average_acc = np.mean(AccuaracyAll) * 100
        #         std_acc = np.std(AccuaracyAll) * 100
        #         print('avg accuracy over %d random splits: %.1f +/- %.1f, for train_num: %d, val_num:%d\n' % (
        #             numRandom, average_acc, std_acc, train_num, val_num))
        #         Accuaracy_test_allK.append(average_acc)
    model.load_state_dict(torch.load(args.best_model_path))

    model.eval()
    embeds = model(X, edge_index)  

    #embeds = F.normalize(embeds, p=2, dim=1)
    embeds = embeds.detach().cpu()

    Accuaracy_test_allK = []
    numRandom = 20

    for train_num in [20]:

        AccuaracyAll = []
        for random_state in range(numRandom):
            print(
                "\n=============================%d-th random split with training num %d============================="
                % (random_state + 1, train_num))

            if train_num == 20:
                if args.dataset in ['cora', 'citeseer', 'pubmed']:
                    # train_num per class: 20, val_num: 500, test: 1000
                    val_num = 500
                    idx_train, idx_val, idx_test = random_planetoid_splits(c, torch.tensor(lab), train_num,
                                                                            random_state)
                else:
                    # Coauthor CS, Amazon Computers, Amazon Photo
                    # train_num per class: 20, val_num per class: 30, test: rest
                    val_num = 30
                    idx_train, idx_val, idx_test = get_train_data(Y, train_num, val_num, random_state)

            else:
                val_num = 0  # do not use a validation set when the training labels are extremely limited
                idx_train, idx_val, idx_test = get_train_data(Y, train_num, val_num, random_state)

            train_embs = embeds[idx_train, :]
            val_embs = embeds[idx_val, :]
            test_embs = embeds[idx_test, :]

            if train_num == 20:
                # find the best parameter C using validation set
                best_val_score = 0.0
                for param in [1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100]:
                    LR = LogisticRegression(solver='liblinear', multi_class='ovr', random_state=0, C=param)
                    LR.fit(train_embs, lab[idx_train])
                    val_score = LR.score(val_embs, lab[idx_val])
                    if val_score > best_val_score:
                        best_val_score = val_score
                        best_parameters = {'C': param}

                LR_best = LogisticRegression(solver='liblinear', multi_class='ovr', random_state=0, **best_parameters)

                LR_best.fit(train_embs, lab[idx_train])
                y_pred_test = LR_best.predict(test_embs)  # pred label
                print("Best accuracy on validation set:{:.4f}".format(best_val_score))
                print("Best parameters:{}".format(best_parameters))

            else:  # not use a validation set when the training labels are extremely limited
                LR = LogisticRegression(solver='liblinear', multi_class='ovr', random_state=0)
                LR.fit(train_embs, lab[idx_train])
                y_pred_test = LR.predict(test_embs)  # pred label

            test_acc = accuracy_score(lab[idx_test], y_pred_test)
            print("test accuaray:{:.4f}".format(test_acc))
            AccuaracyAll.append(test_acc)

        average_acc = np.mean(AccuaracyAll) * 100
        std_acc = np.std(AccuaracyAll) * 100
        print('avg accuracy over %d random splits: %.1f +/- %.1f, for train_num: %d, val_num:%d\n' % (
            numRandom, average_acc, std_acc, train_num, val_num))
        Accuaracy_test_allK.append(average_acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1,help="random seed")
    parser.add_argument("--tau", type=float, default=0.7, help="tau")
    parser.add_argument("--dropout", type=float, default=0.5, help="dropout")
    parser.add_argument("--epsilon", type=float, default=0.01, help="epsilon")
    parser.add_argument("--nb_epochs", type=int, default=40, help="Number of epochs")
    parser.add_argument("--device", type=int, default=0, help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--dataset", type=str, default="cora", help="which dataset for training")
    parser.add_argument("--hidden", type=int, default=500, help="layer hidden")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=5e-5,help="weight decay")
    parser.add_argument("--sparse", type=bool, default=False, help="Sparse or not")
    parser.add_argument("--patience", type=int, default=20, help="Patience for early stopping")
    parser.add_argument("--best_model_path", type=str, default='best_model.pkl', help="Path to save the best model")
    train()

