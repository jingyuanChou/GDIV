# GDIV: Graph-Disentangled Instrumental Variables

This repository implements **GDIV**, a two-stage framework for causal effect estimation on **graph-structured data with entangled treatments**.

---

## Overview

### Stage 1: Graph Representation Learning
Given a graph with node features and treatments, the model learns three node-level representations:
- **Z_iv**: instrument-related embedding
- **Z_c**: confounder-related embedding
- **Z_r**: risk / residual embedding

Training objectives:
- Reconstruct the adjacency matrix from each representation (inner-product decoder)
- Predict treatment assignment using `concat(Z_iv, Z_c)`

### Stage 2: Outcome Prediction
- Use the trained Stage-1 model to compute `Z_c`, `Z_r`, and predicted treatment `T_hat`
- Detach these representations from the computation graph
- Train an outcome head to predict outcomes from `(T_hat, Z_c, Z_r)`
- Evaluate causal effects using PEHE and ATE error

---

## Repository Structure
main.py # main training + evaluation pipeline

models.py # GDIV model and outcome MLP

datasets.py # dataset loading utilities

process_flicker_blogcatalog.py # example dataset generation

MI_estimation.py # mutual information utility 

Causal_forest.py # causal forest baseline

DeepIV.py # DeepIV baseline

plot_results.py # plotting helper

---

## Environment

Recommended:
- Python 3.8+
- PyTorch

Minimal installation:
```bash
pip install torch numpy scipy scikit-learn matplotlib
```
Causal_forest.py requires econml

DeepIV.py requires econml and keras

## Data Format
./datasets/<DATASET_NAME>/<DATASET_NAME><exp_id>.mat

Each .mat file should contain:

Node features X

Adjacency matrix Network

Treatment T

Observed outcome Y

Potential outcomes Y1, Y0

## How to Run

python main.py --dataset FlickrGraphDVAE --path ./datasets/ --epochs 200

Common arguments:

--dataset dataset name

--path dataset root directory

--epochs number of training epochs

--hidden hidden dimension size

--lr learning rate

--clip gradient clipping threshold

## Output

The script prints:

Stage 1: adjacency reconstruction loss and treatment prediction loss

Stage 2: outcome prediction MSE

Final evaluation: mean and standard deviation of PEHE and MAE(ATE)
