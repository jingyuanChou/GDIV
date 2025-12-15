from torch.nn.parameter import Parameter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    return Parameter(torch.FloatTensor(input_dim, output_dim).uniform_(-init_range, init_range))

class GraphConvolution(nn.Module):
    # Assuming adjacency matrix is given in dense format
    def __init__(self, input_dim, output_dim, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()
        self.weight = glorot_init(input_dim, output_dim)
        self.dropout = dropout
        self.act = act

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.mm(x, self.weight)
        x = torch.mm(adj, x)
        return self.act(x)

class GraphConvolutionSparse(nn.Module):
    # Assuming adjacency matrix is given in sparse format
    def __init__(self, input_dim, output_dim, dropout=0., act=F.relu):
        super(GraphConvolutionSparse, self).__init__()
        self.weight = glorot_init(input_dim, output_dim)
        self.dropout = dropout
        self.act = act

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.spmm(adj, x)
        x = torch.spmm(adj, torch.mm(x, self.weight))
        return self.act(x)

class Graphite(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0., act=F.relu):
        super(Graphite, self).__init__()
        self.weight = glorot_init(input_dim, output_dim)
        self.dropout = dropout
        self.act = act

    def forward(self, x, recon_1, recon_2):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.mm(x, self.weight)
        x = torch.mm(recon_1, torch.mm(recon_1.T, x)) + torch.mm(recon_2, torch.mm(recon_2.T, x))
        return self.act(x)

class GraphiteSparse(nn.Module):
    # For sparse inputs
    def __init__(self, input_dim, output_dim, dropout=0., act=F.relu):
        super(GraphiteSparse, self).__init__()
        self.weight = glorot_init(input_dim, output_dim)
        self.dropout = dropout
        self.act = act

    def forward(self, x, recon_1, recon_2):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.spmm(x, self.weight)
        x = torch.mm(recon_1, torch.mm(recon_1.T, x)) + torch.mm(recon_2, torch.mm(recon_2.T, x))
        return self.act(x)

class InnerProductDecoder(nn.Module):
    def __init__(self, dropout=0., act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = torch.mm(z, z.T)
        return self.act(adj)

class ScaledInnerProductDecoder(nn.Module):
    # ScaledInnerProductDecoder is similar to InnerProductDecoder but with a scaling factor
    def __init__(self, input_dim, dropout=0., act=torch.sigmoid):
        super(ScaledInnerProductDecoder, self).__init__()
        self.scale = glorot_init(1, 1)  # Scaling factor
        self.dropout = dropout
        self.input_dim = input_dim
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        scaled_eye = torch.eye(self.input_dim) * self.scale.squeeze()
        adj = torch.mm(z, scaled_eye)
        adj = torch.mm(adj, z.T)
        return self.act(adj)


class FLAGS:
    hidden1 = 32
    hidden2 = 16
    hidden3 = 16
    vae = True
    autoregressive_scalar = 0.5

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def build(self):
        raise NotImplementedError

    def fit(self):
        pass

    def predict(self):
        pass

class GCNModelVAE(Model):
    def __init__(self, num_features, num_nodes, features_nonzero):
        super(GCNModelVAE, self).__init__()
        self.gc1 = GCNConv(num_features, FLAGS.hidden1)
        self.gc2 = GCNConv(FLAGS.hidden1, FLAGS.hidden2)
        self.gc3 = GCNConv(FLAGS.hidden1, FLAGS.hidden2)
        self.dc = InnerProductDecoder(FLAGS.hidden2)

    def encode(self, x, edge_index):
        hidden1 = self.gc1(x, edge_index)
        return self.gc2(hidden1, edge_index), self.gc3(hidden1, edge_index)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar / 2)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, edge_index):
        mu, logvar = self.encode(x, edge_index)
        z = self.reparameterize(mu, logvar)
        return self.dc(z), mu, logvar

class GCNModelFeedback(GCNModelVAE):
    def __init__(self, num_features, num_nodes, features_nonzero):
        super(GCNModelFeedback, self).__init__(num_features, num_nodes, features_nonzero)
        self.l0 = GraphiteSparse(num_features, FLAGS.hidden3)
        self.l1 = Graphite(FLAGS.hidden2, FLAGS.hidden3)
        self.l2 = Graphite(FLAGS.hidden3, FLAGS.hidden2)
        self.l3 = InnerProductDecoder(FLAGS.hidden2)

    def decoder(self, z, recon_1, recon_2):
        update = self.l1(z, recon_1, recon_2) + self.l0(self.inputs, recon_1, recon_2)
        update = self.l2(update, recon_1, recon_2)
        update = (1 - FLAGS.autoregressive_scalar) * z + FLAGS.autoregressive_scalar * update
        return self.l3(update)

    def sample(self, num_samples):
        z = torch.randn(num_samples, FLAGS.hidden2)
        recon = torch.sigmoid(self.decoder(z))
        return recon