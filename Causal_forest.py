import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from econml.dml import CausalForestDML
from sklearn.metrics import mean_squared_error, mean_absolute_error
import scipy.io as sio
import scipy.sparse as sp

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, default='./datasets/')
parser.add_argument('--name', type=str, default='FlickrGraphDVAE')
parser.add_argument('--extrastr', type=str, default='')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
args = parser.parse_args()

pehe_ls = list()
mae_ate_ls = list()

for exp_id in range(10):
    Graph_data = sio.loadmat(args.dataset_path + args.name + '/' + args.name + str(exp_id) + '.mat')
    # Load your data into a DataFrame
    # Separate covariates (X), treatments (T), and factual/counterfactual outcomes (Y_factual, Y_counterfactual)
    X = Graph_data['X_100']
    T = Graph_data['T']
    Y_1 = Graph_data['Y1']
    Y_0 = Graph_data['Y0']
    Y_factual = Graph_data['Y_actual']

    Y_counterfactual = list()

    for idx in range(Y_factual.shape[1]):
        if Y_factual[0,idx] == Y_1[0,idx]:
            Y_counterfactual.append(Y_0[0,idx])
        else:
            Y_counterfactual.append(Y_1[0,idx])

    Y_counterfactual = np.array(Y_counterfactual)

    # Split the data into training and testing sets
    X_train, X_test, T_train, T_test, Y_factual_train, Y_factual_test, Y_counterfactual_train, Y_counterfactual_test = train_test_split(
        X.toarray(), T[0], Y_factual[0], Y_counterfactual, test_size=0.2, random_state=42)

    # Initialize the Causal Forest model
    causal_forest = CausalForestDML(n_estimators=100, random_state=42)

    # Fit the model on your training data
    causal_forest.fit(Y = Y_factual_train, T = T_train, X = X_train)

    # Estimate the Individual Treatment Effects (ITE) for the test data
    ite = causal_forest.effect(X_test)

    # Estimate the Average Treatment Effect (ATE)
    ate = np.mean(ite)
    print(f'ATE: {ate}')

    # Assuming you have the true ITE (ground truth) stored in `true_ite` for the test data
    true_ite = Y_factual_test - Y_counterfactual_test

    # Calculate PEHE (Rooted Precision in Estimation of Heterogeneous Effect)
    pehe = np.sqrt(mean_squared_error(true_ite, ite))
    print(f'PEHE: {pehe}')

    # Calculate Mean Absolute Error (ATE)
    ate_error = mean_absolute_error(true_ite, ite)
    print(f'Mean Absolute Error (ATE): {ate_error}')

    pehe_ls.append(pehe)
    mae_ate_ls.append(ate_error)

print(np.nanmean(pehe_ls), np.nanstd(pehe_ls))
print(np.nanmean(mae_ate_ls), np.nanstd(mae_ate_ls))
