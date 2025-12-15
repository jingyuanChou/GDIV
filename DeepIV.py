import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import scipy.io as sio
import scipy.sparse as sp
from econml.iv.nnet import DeepIV
from keras.layers import Input, Dense, Concatenate
from keras.models import Model


# Define custom treatment model (m)
def treatment_model(z, x):
    # Inputs for the treatment model
    input_z = Input(shape=(z.shape[1],))  # Input for treatment
    input_x = Input(shape=(x.shape[1],))  # Input for covariates

    # Combine inputs
    concatenated = Concatenate()([input_z, input_x])

    # Neural network layers for the treatment model
    hidden_layer1 = Dense(64, activation='relu')(concatenated)
    hidden_layer2 = Dense(32, activation='relu')(hidden_layer1)

    # Output layer for the treatment model
    treatment_output = Dense(1)(hidden_layer2)  # Assuming one-dimensional treatment

    # Create the treatment model
    model = Model(inputs=[input_z, input_x], outputs=treatment_output)

    return model


# Define custom outcome model (h)
def outcome_model(t, x):
    # Inputs for the outcome model
    input_t = Input(shape=(t.shape[1],))  # Input for treatment
    input_x = Input(shape=(x.shape[1],))  # Input for covariates

    # Combine inputs
    concatenated = Concatenate()([input_t, input_x])

    # Neural network layers for the outcome model
    hidden_layer1 = Dense(64, activation='relu')(concatenated)
    hidden_layer2 = Dense(32, activation='relu')(hidden_layer1)

    # Output layer for the outcome model
    outcome_output = Dense(1)(hidden_layer2)  # Assuming one-dimensional outcome

    # Create the outcome model
    model = Model(inputs=[input_t, input_x], outputs=outcome_output)

    return model


# Define command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, default='./datasets/')
parser.add_argument('--name', type=str, default='FlickrGraphDVAE')
parser.add_argument('--extrastr', type=str, default='')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
args = parser.parse_args()

if __name__ == '__main__':
    pehe_ls = list()
    mae_ate_ls = list()

    for exp_id in range(10):
        # Load your data into a DataFrame
        Graph_data = sio.loadmat(args.dataset_path + args.name + '/' + args.name + str(exp_id) + '.mat')
        # Separate covariates (X), treatments (T), and factual/counterfactual outcomes (Y_factual, Y_counterfactual)
        X = Graph_data['X_100']
        T = Graph_data['T']
        Y_1 = Graph_data['Y1']
        Y_0 = Graph_data['Y0']
        Y_factual = Graph_data['Y_actual']

        Y_counterfactual = list()

        for idx in range(Y_factual.shape[1]):
            if Y_factual[0, idx] == Y_1[0, idx]:
                Y_counterfactual.append(Y_0[0, idx])
            else:
                Y_counterfactual.append(Y_1[0, idx])

        Y_counterfactual = np.array(Y_counterfactual)
        # Split the data into training and testing sets
        X_train, X_test, T_train, T_test, Y_factual_train, Y_factual_test, Y_counterfactual_train, Y_counterfactual_test = train_test_split(
            X.toarray(), T[0], Y_factual[0], Y_counterfactual, test_size=0.2, random_state=42)

        # Define custom treatment and outcome models
        treatment_model = treatment_model(T_train[:,np.newaxis], X_train)
        outcome_model = outcome_model(T_train[:,np.newaxis], X_train)

        # Initialize the DeepIV model with custom treatment and outcome models
        model = DeepIV(n_components=10, m=treatment_model, h=outcome_model, n_samples=1)

        # Fit the model on your training data
        model.fit(Y_factual_train, T_train, X_train)

        # Predict the causal effects on the test data
        predicted_effect = model.effect(X_test)

        # Assuming you have the true treatment effect (ground truth) stored in `true_effect` for the test data
        true_effect = Y_factual_test - Y_counterfactual_test

        # Calculate PEHE (Rooted Precision in Estimation of Heterogeneous Effect)
        pehe = np.sqrt(mean_squared_error(true_effect, predicted_effect))
        print(f'PEHE: {pehe}')

        # Calculate Mean Absolute Error (ATE)
        ate = mean_absolute_error(true_effect, predicted_effect)
        print(f'Mean Absolute Error (ATE): {ate}')

        pehe_ls.append(pehe)
        mae_ate_ls.append(ate)
    print(np.nanmean(pehe_ls), np.nanstd(pehe_ls))
    print(np.nanmean(mae_ate_ls), np.nanstd(mae_ate_ls))
