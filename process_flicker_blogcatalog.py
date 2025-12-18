import scipy.io as sio
import numpy as np
from matplotlib import rc, rcParams
import torch
from torch import nn

font = {'weight': 'bold',
        'size': 14}

rc('font', **font)
path = 'datasets/'
name = 'BlogCatalog1'  # BlogCatalog1
data = sio.loadmat(path + name + '/{}.mat'.format(name))
data.keys()
X = data['X_100']
nodes = X.shape[0]
A = data['Network']
A_dense = np.array(A.todense())
lamd = 0.5
beta = 0.5
# random sample an instance and use its topic distribution as the centroid
# repeat 10 times
dense_adjacency_matrix = A.toarray()
adjacency_dict = {i: np.where(row)[0].tolist() for i, row in enumerate(dense_adjacency_matrix)}
mean = 0  # Mean (mu)
std_dev = 0.5

for exp_id in range(10):
    dim_u = X.shape[1]
    dim_x = X.shape[1]
    # Scaling factor
    scaling_factor = 20
    # Creating an identity matrix I
    I = np.identity(dim_u)
    # Creating the covariance matrix mu * I
    covariance_matrix = scaling_factor * I
    # Mean vector (zero in this case)
    mean_u = np.zeros(dim_u)

    # Generate samples from the multivariate normal distribution
    unobserved_conf = np.random.multivariate_normal(mean_u, covariance_matrix, size=nodes)

    # Simulate treatment for all nodes
    treatment_det_f = nn.Sigmoid()
    treatment_ls = list()
    Y_ls = list()
    Y_1 = list()
    Y_0 = list()
    for i in range(nodes):
        X_i = torch.from_numpy(X[i].toarray()).float()
        U_i = torch.from_numpy(unobserved_conf[i]).float().unsqueeze(-1)
        neighbors = adjacency_dict[i]
        epsilon_t = np.random.normal(0, 0.01)
        num_neighbors = len(neighbors)
        Theta_t_x = torch.from_numpy(np.random.normal(mean, std_dev, size=(dim_x, 1))).float()
        Theta_t_u = torch.from_numpy(np.random.normal(mean, std_dev, size=(dim_u, 1))).float()
        Theta_t_u_trans = torch.transpose(Theta_t_u, 1, 0)

        neighbor_sum_ls = list()
        for neighbor in neighbors:
            X_neighbor = X[neighbor].toarray().transpose()
            X_neighbor = torch.from_numpy(X_neighbor).float()
            Theta_t_x_trans = torch.transpose(Theta_t_x, 1, 0)
            temp = torch.matmul(Theta_t_x_trans, X_neighbor)
            neighbor_sum_ls.append(temp)
        neighbor_sum = torch.stack(neighbor_sum_ls)
        neighbor_sum = torch.sum(neighbor_sum, dim=0)

        T_i = (1 - lamd) * torch.matmul(torch.transpose(Theta_t_x, 1, 0), torch.transpose(X_i, 1, 0)) + (
            lamd) * neighbor_sum / num_neighbors + \
              torch.matmul(Theta_t_u_trans, U_i) + epsilon_t

        T_i = treatment_det_f(T_i)

        # Outcome Generation
        Theta_y_i = torch.from_numpy(np.random.normal(mean, std_dev, size=(dim_x, 1))).float()
        Theta_y_i_trans = torch.transpose(Theta_y_i, 1, 0)

        Theta_0_i = torch.from_numpy(np.random.normal(mean, std_dev, size=(dim_x, 1))).float()
        Theta_0_i_trans = torch.transpose(Theta_0_i, 1, 0)

        Theta_u = torch.from_numpy(np.random.normal(mean, std_dev, size=(dim_u, 1))).float()
        Theta_u_trans = torch.transpose(Theta_u, 1, 0)

        epsilon_Y = np.random.normal(0, 0.1)

        if T_i > 0.5:
            Y_i = 1 * torch.matmul(Theta_y_i_trans, torch.transpose(X_i, 1, 0)) + torch.matmul(Theta_0_i_trans,
                                                                                               torch.transpose(X_i, 1,
                                                                                                               0)) + beta * \
                  torch.matmul(Theta_u_trans, U_i) + epsilon_Y
            treatment_ls.append(1)
        else:
            Y_i = 0 * torch.matmul(Theta_y_i_trans, torch.transpose(X_i, 1, 0)) + torch.matmul(Theta_0_i_trans,
                                                                                               torch.transpose(X_i, 1,
                                                                                                               0)) + beta * \
                  torch.matmul(Theta_u_trans, U_i) + epsilon_Y
            treatment_ls.append(0)

        Y_i_0 = 0 * torch.matmul(Theta_y_i_trans, torch.transpose(X_i, 1, 0)) + torch.matmul(Theta_0_i_trans,
                                                                                             torch.transpose(X_i, 1,
                                                                                                             0)) + beta * \
                torch.matmul(Theta_u_trans, U_i) + epsilon_Y
        Y_i_1 = 1 * torch.matmul(Theta_y_i_trans, torch.transpose(X_i, 1, 0)) + torch.matmul(Theta_0_i_trans,
                                                                                             torch.transpose(X_i, 1,
                                                                                                             0)) + beta * \
                torch.matmul(Theta_u_trans, U_i) + epsilon_Y

        Y_0.append(Y_i_0.item())
        Y_1.append(Y_i_1.item())

        Y_ls.append(Y_i.item())
    Y_ls = torch.from_numpy(np.array(Y_ls))

    # save the data
    sio.savemat('./datasets/' + name + 'GraphDVAE' + '/' + name + str(exp_id) + '.mat', {
        'X_100': data['X_100'], 'T': treatment_ls, 'Y1': Y_1, 'Y0': Y_0, 'Y_actual': Y_ls,
        'Attributes': data['Attributes'], 'Label': data['Label'],
        'Network': data['Network']})
