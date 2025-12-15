import argparse
from datetime import time

import numpy as np
import torch
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
from synthetic_generation import generate_data
from utils import normalize, edge_index_to_adjacency_matrix, mutual_information_loss, load_data, \
    sparse_mx_to_torch_sparse_tensor, load_real_data
from models import GraphDVAE, MLP, output_MLP

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--nocuda', type=int, default=0, help='Disables CUDA training.')
parser.add_argument('--dataset', type=str, default='FlickrGraphDVAE')
parser.add_argument('--extrastr', type=str, default='')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=1e-2, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=100, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=1e-4, help='trade-off of representation balancing.')
parser.add_argument('--clip', type=float, default=100., help='gradient clipping')
parser.add_argument('--nout', type=int, default=2)
parser.add_argument('--nin', type=int, default=2)
parser.add_argument('--tr', type=float, default=0.6)
parser.add_argument('--path', type=str, default='./datasets/')
parser.add_argument('--normy', type=int, default=1)

args = parser.parse_args()
Tensor = torch.FloatTensor
LongTensor = torch.LongTensor


def predict(model, X, edge_index):
    z_iv = model.encoder_z_iv(X, edge_index)
    z_c = model.encoder_z_c(X, edge_index)
    z_r = model.encoder_z_r(X, edge_index)

    concatnated = torch.cat((z_iv, z_c), dim=1)
    pred_T = F.log_softmax(model.treatment_pred(concatnated), dim=1)

    return z_iv, z_c, z_r, pred_T


if __name__ == '__main__':
    # Generate synthetic data
    # dim_u, dim_x, num_samples, scaling_factor, prob, lamd, beta
    pehe_ts_ls = list()
    mae_ate_ts_ls = list()

    for exp_id in range(10):
        X, A, T, Y, Y1, Y0 = load_data(args.path, name=args.dataset, original_X=False, exp_id=str(exp_id),
                                       extra_str=args.extrastr)
        # X, A, T, Y = load_real_data()

        n = X.shape[0]
        n_train = int(n * 0.6)
        n_test = int(n * 0.2)
        # n_valid = n_test

        idx = np.random.permutation(n)
        idx_train, idx_test, idx_val = idx[:n_train], idx[n_train:n_train + n_test], idx[n_train + n_test:]

        X = normalize(X)  # row-normalize
        # A = utils.normalize(A+sp.eye(n))

        X = X.todense()
        X = Tensor(X)

        Y1 = Tensor(np.squeeze(Y1))
        Y0 = Tensor(np.squeeze(Y0))
        Y = Tensor(np.squeeze(Y))
        T = LongTensor(np.squeeze(T))

        A, true_adj = sparse_mx_to_torch_sparse_tensor(A)
        true_adj = torch.from_numpy(true_adj)
        print(true_adj.sum())
        # print(X.shape, Y1.shape, A.shape)

        idx_train = LongTensor(idx_train)
        idx_val = LongTensor(idx_val)
        idx_test = LongTensor(idx_test)
        input_dim, hidden_dim, output_dim = X.size(1), 32, 1
        # Example dimensions
        num_classes = torch.unique(T).shape[0]

        model = GraphDVAE(input_dim, hidden_dim, num_classes)

        optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

        criterion_G = nn.BCEWithLogitsLoss()

        patience = 15

        min_val_loss = float('inf')
        best_model_path = 'best_model.pth'

        count_patience = 0
        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()
            criterion = nn.NLLLoss()

            num_nodes = T.size(0)
            z_iv, z_iv_rec, z_c, z_c_rec, z_r, z_r_rec, mi_total_loss, pred_T = model(X, A)

            # true_adj = edge_index_to_adjacency_matrix(A, num_nodes)

            # reconstruction loss
            rec_loss_z1 = criterion_G(z_iv_rec, true_adj)
            rec_loss_z2 = criterion_G(z_c_rec, true_adj)
            rec_loss_z3 = criterion_G(z_r_rec, true_adj)

            #
            T_train = T[idx_train].squeeze().long()
            pred_T_train = pred_T[idx_train]

            treatment_pred_train_loss = criterion(pred_T_train, T_train)

            # training_loss = rec_loss_z1 + rec_loss_z2 + rec_loss_z3 + mi_total_loss + treatment_pred_train_loss

            rec_loss = rec_loss_z1 + rec_loss_z2 + rec_loss_z3

            training_loss = rec_loss_z1 + rec_loss_z2 + rec_loss_z3 + treatment_pred_train_loss

            # training_loss.backward(retain_graph=True)
            training_loss.backward()

            optimizer.step()

            count_patience = count_patience + 1

            if epoch % 1 == 0:
                # validation
                model.eval()
                T_val = T[idx_val].squeeze().long()
                pred_T_val = pred_T[idx_val]

                treatment_pred_loss_val = criterion(pred_T_val, T_val)

                if treatment_pred_loss_val < min_val_loss:
                    min_val_loss = treatment_pred_loss_val
                    count_patience = 0
                    # Save this model
                    torch.save(model.state_dict(), best_model_path)

                print('Epoch: {:04d}'.format(epoch + 1),
                      'loss_train: {:.4f}'.format(training_loss.item()),
                      'treatment_pred: {:.4f}'.format(treatment_pred_train_loss.item()),
                      'loss_val: {:.4f}'.format(treatment_pred_loss_val.item()),
                      'reconstruction loss: {:4f}'.format(rec_loss.item()))
                # 'pehe_val: {:.4f}'.format(pehe_val.item()),
                # 'mae_ate_val: {:.4f}'.format(mae_ate_val.item()),

            if count_patience > patience:
                print('Early Stopping')
                break

        print("Optimization Finished For the First Stage")

        best_model = GraphDVAE(input_dim, hidden_dim, num_classes)
        best_model.load_state_dict(torch.load(best_model_path))
        z_iv, z_c, z_r, t_pred = predict(best_model, X, A)
        prob_T = torch.exp(t_pred)

        t_pred = torch.argmax(prob_T, dim=1).unsqueeze(-1).float()

        # Freeze the encoder's weights
        for param in model.parameters():
            param.requires_grad = False

        # Stage 2: Training the Classifier with Learned Embeddings
        # idx_train: training set index
        # idx_val: validation set index
        # idx_test: testing set index

        # Use z_c, z_r, t_pred to obtain Y
        input_dim = t_pred.size(1)

        pred_module = output_MLP(input_dim, hidden_dim, output_dim)
        optimizer_2nd = Adam(pred_module.parameters(), lr=0.001, weight_decay=1e-5)
        MSEloss = nn.MSELoss()

        min_val_loss = float('inf')
        for epoch in range(args.epochs):
            optimizer_2nd.zero_grad()

            t_pred_train = t_pred[idx_train]
            z_c_train = z_c[idx_train]
            z_r_train = z_r[idx_train]

            out = pred_module(t_pred_train, z_c_train, z_r_train)
            '''
            if args.normy:
                # recover the normalized outcomes
                ym, ys = torch.mean(YF[idx_train]), torch.std(YF[idx_train])
                YFtr, YFva = (YF[idx_train] - ym) / ys, (YF[idx_val] - ym) / ys
            '''

            Y_train = Y[idx_train].float()
            loss_classifier = MSEloss(out, Y_train)

            loss_classifier.backward(retain_graph=True)
            optimizer_2nd.step()

            with torch.no_grad():
                t_pred_val = t_pred[idx_val]
                z_c_val = z_c[idx_val]
                z_r_val = z_r[idx_val]
                out_val = pred_module(t_pred_val, z_c_val, z_r_val)
                Y_val = Y[idx_val].float()

                val_loss = MSEloss(out_val, Y_val)

                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    count_patience = 0
                    # Save this model
                    torch.save(pred_module.state_dict(), best_model_path)

                print('Epoch: {:04d}'.format(epoch + 1),
                      'loss_train: {:.4f}'.format(loss_classifier.item()),
                      'loss_val: {:.4f}'.format(val_loss.item()))

        best_model = output_MLP(input_dim, hidden_dim, output_dim)
        best_model.load_state_dict(torch.load(best_model_path))

        t_pred_f = t_pred[idx_test]
        t_pred_cf = 1 - t_pred_f
        z_c_test = z_c[idx_test]
        z_r_test = z_r[idx_test]

        Y_1_test = Y1[idx_test].float()
        Y_0_test = Y0[idx_test].float()

        factual_outcome = best_model(t_pred_f, z_c_test, z_r_test)
        counter_factual_outcome = best_model(t_pred_cf, z_c_test, z_r_test)

        YF = torch.where(t_pred_f > 0, Y1, Y0)
        YCF = torch.where(t_pred_f > 0, Y0, Y1)

        y1_pred, y0_pred = torch.where(t_pred_f > 0, factual_outcome, counter_factual_outcome), torch.where(
            t_pred_f > 0, counter_factual_outcome, factual_outcome)

        pehe_ts = torch.sqrt(MSEloss((y1_pred - y0_pred), (Y1 - Y0)[idx_test]))
        mae_ate_ts = torch.abs(torch.mean((y1_pred - y0_pred)) - torch.mean((Y1 - Y0)[idx_test]))

        mae_ate_ts_ls.append(mae_ate_ts.item())
        pehe_ts_ls.append(pehe_ts.item())

        print("Test set results:",
              "pehe_ts= {:.4f}".format(pehe_ts.item()),
              "mae_ate_ts= {:.4f}".format(mae_ate_ts.item()))

    print(np.mean(pehe_ts_ls), np.std(pehe_ts_ls))
    print(np.mean(mae_ate_ts_ls), np.std(mae_ate_ts_ls))
