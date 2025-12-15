import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import scipy.io as sio
import scipy.sparse as sp
from scipy.sparse import csc_matrix


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx, cuda=False):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)

    sparse_tensor = torch.sparse.FloatTensor(indices, values, shape)
    if cuda:
        sparse_tensor = sparse_tensor.cuda()
    return sparse_tensor


def mutual_information_loss(z1, z2, mine_net):
    # Estimate MI between z1 and z2
    joint = mine_net(z1, z2)  # Joint distribution
    marginal = mine_net(z1, torch.roll(z2, shifts=1, dims=0))  # Marginal distribution
    mi_estimate = joint - torch.log(torch.exp(marginal) + 1e-6)
    return -mi_estimate


def reconstruction(label, pred):
    return F.binary_cross_entropy(label, pred)


def edge_index_to_adjacency_matrix(edge_index, num_nodes):
    """Convert an edge_index to an adjacency matrix."""
    adjacency_matrix = torch.zeros((num_nodes, num_nodes))
    # Add edges to the adjacency matrix
    for i in range(edge_index.shape[1]):
        node1, node2 = edge_index[:, i]
        adjacency_matrix[node1, node2] = 1
        adjacency_matrix[node2, node1] = 1  # Assuming undirected graph
    return adjacency_matrix

def load_data(path, name='BlogCatalog', exp_id='0', original_X=False, extra_str=""):
    data = sio.loadmat(path + name + extra_str + '/' + 'FlickrGraphDVAE' + exp_id + '.mat')
    A = data['Network']  # csr matrix

    # try:
    # 	A = np.array(A.todense())
    # except:
    # 	pass

    if not original_X:
        X = data['X_100']
    else:
        X = data['Attributes']
    Y = data['Label'] # 'Y_actual' for blogcatalog and flickr, 'Label' for synthetic
    Y1 = data['Y1']
    Y0 = data['Y0']
    T = data['T']

    return X, A, T, Y, Y1, Y0

def load_real_data():
    temp = pd.read_csv('2020-04-15-df-cases.csv')
    T = temp.iloc[:,-1].values

    Y = temp['Cases']/temp['Population']
    Y = Y.values
    temp = temp.iloc[:,:-2]
    X = temp.values

    A = pd.read_csv('real_data/counties_adj_VA.csv')
    nodes = pd.unique(A[['fipscounty', 'fipsneighbor']].values.ravel('K'))
    adj_matrix = pd.DataFrame(0, index=nodes, columns=nodes)
    for _, row in A.iterrows():
        adj_matrix.at[row['fipscounty'], row['fipsneighbor']] = 1
        # If the graph is undirected, also set the opposite direction
        adj_matrix.at[row['fipsneighbor'], row['fipscounty']] = 1
    A = csc_matrix(adj_matrix.values)
    X = X[:,1:]
    return X, A, T, Y

def sparse_mx_to_torch_sparse_tensor(sparse_mx, cuda=False):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    real_adj = sparse_mx.toarray()
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)

    sparse_tensor = torch.sparse.FloatTensor(indices, values, shape)
    if cuda:
        sparse_tensor = sparse_tensor.cuda()
    return sparse_tensor, real_adj