import numpy as np
import torch
import torch.nn as nn
import networkx as nx
import scipy.io as sio
from scipy.sparse import csc_matrix


def generate_data(dim_u, dim_x, num_samples, scaling_factor, prob, lamd, beta):

    for exp_id in range(10):
        # Size of the identity matrix
        dim_u = dim_u
        # Scaling factor
        scaling_factor = scaling_factor
        # Creating an identity matrix I
        I = np.identity(dim_u)
        # Creating the covariance matrix mu * I
        covariance_matrix = scaling_factor * I
        # Mean vector (zero in this case)
        mean_u = np.zeros(dim_u)

        # Number of samples to generate
        num_samples = num_samples  # for example, generating 100 samples

        # Generate samples from the multivariate normal distribution
        unobserved_conf = np.random.multivariate_normal(mean_u, covariance_matrix, size=num_samples)
        unobserved_conf = torch.from_numpy(unobserved_conf).float()
        feature_transform = torch.nn.Linear(dim_u, dim_x)
        Transformed_X = feature_transform(unobserved_conf)

        mean = 0  # Mean (mu)
        std_dev = 1  # Standard deviation (sigma)
        size_x = dim_x  # Length of the noise vector
        # Generate the noise vector
        noise_vector = np.random.normal(mean, std_dev, size=(num_samples, size_x))
        noise_vector = torch.from_numpy(noise_vector).float()

        Transformed_X = Transformed_X + noise_vector

        # Generate graphs using Erdös-Rényi model

        G = nx.erdos_renyi_graph(num_samples, p=prob)
        adjacency_dict = {node: list(neighbors) for node, neighbors in G.adjacency()}

        num_nodes = len(G.nodes())
        graph = csc_matrix((num_nodes, num_nodes), dtype=np.int32)

        # Populate the CSC matrix based on the graph edges
        for edge in G.edges():
            row, col = edge
            graph[row, col] = 1
            graph[col, row] = 1

        # Simulate treatment for all nodes
        treatment_det_f = nn.Sigmoid()
        treatment_ls = list()
        Y_ls = list()
        Y_1 = list()
        Y_0 = list()
        for i in range(num_samples):
            X_i = Transformed_X[i]
            U_i = unobserved_conf[i]
            neighbors = adjacency_dict[i]
            epsilon_t = np.random.normal(0, 0.01)
            num_neighbors = len(neighbors)
            Theta_t_x = torch.from_numpy(np.random.normal(mean, std_dev, size=(dim_x, 1))).float()
            Theta_t_u = torch.from_numpy(np.random.normal(mean, std_dev, size=(dim_u, 1))).float()
            Theta_t_u_trans = torch.transpose(Theta_t_u, 1, 0)

            neighbor_sum_ls = list()
            for neighbor in neighbors:
                X_neighbor = Transformed_X[neighbor].unsqueeze(-1)
                Theta_t_x_trans = torch.transpose(Theta_t_x, 1, 0)
                temp = torch.matmul(Theta_t_x_trans, X_neighbor)
                neighbor_sum_ls.append(temp)
            neighbor_sum = torch.stack(neighbor_sum_ls)
            neighbor_sum = torch.sum(neighbor_sum, dim = 0)

            T_i = (1 - lamd) * torch.matmul(torch.transpose(Theta_t_x, 1, 0), X_i) + (lamd) * neighbor_sum / num_neighbors + \
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
                Y_i = 1 * torch.matmul(Theta_y_i_trans, X_i) + torch.matmul(Theta_0_i_trans, X_i) + beta * \
                      torch.matmul(Theta_u_trans, U_i) + epsilon_Y
                treatment_ls.append(1)
            else:
                Y_i = 0 * torch.matmul(Theta_y_i_trans, X_i) + torch.matmul(Theta_0_i_trans, X_i) + beta * \
                      torch.matmul(Theta_u_trans, U_i) + epsilon_Y
                treatment_ls.append(0)

            Y_i_0 = 0 * torch.matmul(Theta_y_i_trans, X_i) + torch.matmul(Theta_0_i_trans, X_i) + beta * \
                    torch.matmul(Theta_u_trans, U_i) + epsilon_Y
            Y_i_1 = 1 * torch.matmul(Theta_y_i_trans, X_i) + torch.matmul(Theta_0_i_trans, X_i) + beta * \
                    torch.matmul(Theta_u_trans, U_i) + epsilon_Y


            Y_0.append(Y_i_0.item())
            Y_1.append(Y_i_1.item())

            Y_ls.append(Y_i.item())
        Y_ls = np.array(Y_ls)

        Transformed_X = Transformed_X.detach().numpy()

        sio.savemat('./datasets/' + 'synthetic_beta_0' + '/' + 'synthetic_beta_0_GraphDVAE' + str(exp_id) + '.mat', {
            'X_100': Transformed_X, 'T': np.array(treatment_ls), 'Y1': np.array(Y_1), 'Y0': np.array(Y_0), 'Label': Y_ls,
            'Network': graph})
    return treatment_ls, Transformed_X, Y_ls, G, Y_1, Y_0


if __name__ == '__main__':
    dim_u = 100
    dim_x = 100
    nodes = 1000
    scaling_factor = 20
    prob = 0.2
    lamd = 0.5
    BETA = 0

    T, X, Y, G, Y_1, Y_0 = generate_data(dim_u, dim_x, nodes, scaling_factor, prob, lamd, BETA)
