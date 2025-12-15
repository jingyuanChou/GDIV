import torch.nn as nn
import torch.nn.functional as F
import torch
import torch_geometric.nn as pyg_nn
from utils import mutual_information_loss
import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class output_MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(output_MLP, self).__init__()
        self.treatment_trans = nn.Linear(input_size, hidden_size)
        self.zc = nn.Linear(hidden_size, hidden_size)  # First layer
        self.zr = nn.Linear(hidden_size, hidden_size)  # Output layer
        self.out1 = nn.Linear(hidden_size * 3, hidden_size)  # Output layer
        self.out2 = nn.Linear(hidden_size, output_size)  # Output layer

    def forward(self, treatment, z_c, z_r):
        t_rep = self.treatment_trans(treatment)
        z_c_rep = self.zc(z_c)
        z_r_rep = self.zr(z_r)
        concatenated = torch.cat((t_rep, z_r_rep, z_c_rep), dim=1)
        x = F.relu(self.out1(concatenated))  # Activation function after first layer
        x = self.out2(x)  # Output layer
        return x


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # First layer
        self.fc2 = nn.Linear(hidden_size, output_size)  # Output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Activation function after first layer
        x = self.fc2(x)  # Output layer
        return x


class MINE(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MINE, self).__init__()
        self.fc1 = nn.Linear(input_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.activation = nn.ReLU()

    def forward(self, z1, z2):
        combined = torch.cat([z1, z2], dim=1)
        x = self.activation(self.fc1(combined))
        return self.fc2(x)


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.act = act

    def forward(self, z):
        adj = self.act(torch.mm(z, z.t()))
        return adj


class GCNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, activation=nn.ReLU):
        super(GCNEncoder, self).__init__()
        self.convs = nn.ModuleList()
        self.activations = nn.ModuleList()

        # Input layer
        self.convs.append(GraphConvolution(input_dim, hidden_dim))
        self.activations.append(activation())

        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GraphConvolution(hidden_dim, hidden_dim))
            self.activations.append(activation())

    def forward(self, x, edge_index):
        # x is the node features, edge_index is the graph structure (COO format)
        for conv, activation in zip(self.convs, self.activations):
            x = activation(conv(x, edge_index))
        return x


class GraphDVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_class):
        super(GraphDVAE, self).__init__()

        self.treatment_pred = MLP(hidden_dim * 2, hidden_dim, num_class)

        self.encoder_z_iv = GCNEncoder(input_dim, hidden_dim, num_layers=2, activation=nn.ReLU)
        self.encoder_z_c = GCNEncoder(input_dim, hidden_dim, num_layers=3, activation=nn.Tanh)
        self.encoder_z_r = GCNEncoder(input_dim, hidden_dim, num_layers=4, activation=nn.LeakyReLU)

        self.mine_net = MINE(hidden_dim, hidden_dim)

        self.decoder = InnerProductDecoder()

    def forward(self, X, edge_index):
        # x is the node features, edge_index is the graph structure (COO format)

        z_iv = self.encoder_z_iv(X, edge_index)
        z_c = self.encoder_z_c(X, edge_index)
        z_r = self.encoder_z_r(X, edge_index)

        mi_total_loss = mutual_information_loss(z_iv, z_c, self.mine_net).mean() + mutual_information_loss(z_iv, z_r,
                                                                                                           self.mine_net).mean() + mutual_information_loss(
            z_c, z_r, self.mine_net).mean()

        # reconstruct the original graph structure
        z_iv_rec = self.decoder(z_iv)
        z_c_rec = self.decoder(z_c)
        z_r_rec = self.decoder(z_r)

        concatnated = torch.cat((z_iv, z_c), dim=1)
        pred_T = F.log_softmax(self.treatment_pred(concatnated), dim=1)

        # use z_iv and z_c to predict T

        return z_iv, z_iv_rec, z_c, z_c_rec, z_r, z_r_rec, mi_total_loss, pred_T



class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'