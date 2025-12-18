import argparse
import numpy as np
import torch
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F

from utils import normalize, load_data, sparse_mx_to_torch_sparse_tensor
from models import GDIV, Output_MLP


# -------------------------
# Helpers
# -------------------------
def normalize_adj_torch_sparse(A: torch.Tensor, add_self_loops: bool = True, eps: float = 1e-12) -> torch.Tensor:
    """
    Compute GCN normalized adjacency: D^{-1/2} (A + I) D^{-1/2}
    A: torch.sparse.FloatTensor [N, N] (COO)
    returns: torch.sparse.FloatTensor [N, N]
    """
    if not A.is_sparse:
        raise ValueError("A must be a torch sparse tensor")

    A = A.coalesce()
    n = A.size(0)
    device = A.device

    if add_self_loops:
        eye = torch.arange(n, device=device)
        I = torch.sparse_coo_tensor(
            torch.stack([eye, eye], dim=0),
            torch.ones(n, device=device),
            size=(n, n),
        ).coalesce()
        A = (A + I).coalesce()

    deg = torch.sparse.sum(A, dim=1).to_dense()  # [N]
    deg_inv_sqrt = torch.pow(deg.clamp_min(eps), -0.5)

    idx = A.indices()
    vals = A.values()
    vals = vals * deg_inv_sqrt[idx[0]] * deg_inv_sqrt[idx[1]]

    A_norm = torch.sparse_coo_tensor(idx, vals, size=A.size(), device=device).coalesce()
    return A_norm


def predict(model, X, A):
    z_iv = model.encoder_z_iv(X, A)
    z_c = model.encoder_z_c(X, A)
    z_r = model.encoder_z_r(X, A)
    concatnated = torch.cat((z_iv, z_c), dim=1)
    pred_T = F.log_softmax(model.treatment_pred(concatnated), dim=1)
    return z_iv, z_c, z_r, pred_T


# -------------------------
# Args
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--nocuda', type=int, default=0, help='Disables CUDA training.')
parser.add_argument('--dataset', type=str, default='FlickrGraphDVAE')
parser.add_argument('--extrastr', type=str, default='')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay.')
parser.add_argument('--hidden', type=int, default=32, help='Hidden dim.')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate.')
parser.add_argument('--clip', type=float, default=5.0, help='Gradient clipping max norm.')
parser.add_argument('--path', type=str, default='./datasets/')
parser.add_argument('--normy', type=int, default=1)
parser.add_argument('--debug', type=int, default=0, help='Print debug stats (0/1).')

args = parser.parse_args()

# device + seeds
use_cuda = (args.nocuda == 0) and torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed_all(args.seed)


# -------------------------
# Main
# -------------------------
if __name__ == '__main__':
    pehe_ts_ls = []
    mae_ate_ts_ls = []

    for exp_id in range(10):
        # ----- load -----
        X, A_scipy, T, Y, Y1, Y0 = load_data(
            args.path, name=args.dataset, original_X=False, exp_id=str(exp_id), extra_str=args.extrastr
        )

        n = X.shape[0]
        n_train = int(n * 0.6)
        n_test = int(n * 0.2)

        idx = np.random.permutation(n)
        idx_train = idx[:n_train]
        idx_test = idx[n_train:n_train + n_test]
        idx_val = idx[n_train + n_test:]

        idx_train = torch.LongTensor(idx_train).to(device)
        idx_val = torch.LongTensor(idx_val).to(device)
        idx_test = torch.LongTensor(idx_test).to(device)

        # ----- preprocess X -----
        X = normalize(X)  # row-normalize features
        X = torch.FloatTensor(np.asarray(X.todense())).to(device)

        # ----- tensors -----
        Y1 = torch.FloatTensor(np.squeeze(Y1)).to(device)
        Y0 = torch.FloatTensor(np.squeeze(Y0)).to(device)
        Y = torch.FloatTensor(np.squeeze(Y)).to(device)
        T = torch.LongTensor(np.squeeze(T)).to(device)

        # ----- adjacency -----
        A_torch, true_adj = sparse_mx_to_torch_sparse_tensor(A_scipy)
        A_torch = A_torch.to(device)
        A_torch = normalize_adj_torch_sparse(A_torch, add_self_loops=True)

        true_adj = torch.from_numpy(true_adj).float().to(device)

        print("Edges (sum true_adj):", true_adj.sum().item())

        input_dim = X.size(1)
        hidden_dim = args.hidden
        output_dim = 1
        num_classes = torch.unique(T).shape[0]

        # =========================
        # Stage 1
        # =========================
        model = GDIV(input_dim, hidden_dim, num_classes).to(device)
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        criterion_G = nn.BCEWithLogitsLoss()
        criterion_T = nn.NLLLoss()

        patience1 = 15
        min_val_loss1 = float('inf')
        best_model_path1 = f'best_model_stage1_exp{exp_id}.pth'
        count_patience1 = 0

        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()

            z_iv, z_iv_rec, z_c, z_c_rec, z_r, z_r_rec, mi_total_loss, pred_T = model(X, A_torch)

            rec_loss_z1 = criterion_G(z_iv_rec, true_adj)
            rec_loss_z2 = criterion_G(z_c_rec, true_adj)
            rec_loss_z3 = criterion_G(z_r_rec, true_adj)

            T_train = T[idx_train].squeeze().long()
            pred_T_train = pred_T[idx_train]
            treat_loss = criterion_T(pred_T_train, T_train)

            loss_train = rec_loss_z1 + rec_loss_z2 + rec_loss_z3 + treat_loss

            loss_train.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip)
            optimizer.step()

            # validation (treatment only, like your original)
            model.eval()
            with torch.no_grad():
                T_val = T[idx_val].squeeze().long()
                pred_T_val = pred_T[idx_val]
                loss_val = criterion_T(pred_T_val, T_val)

            count_patience1 += 1
            if loss_val.item() < min_val_loss1:
                min_val_loss1 = loss_val.item()
                count_patience1 = 0
                torch.save(model.state_dict(), best_model_path1)

            if args.debug and (epoch % 5 == 0):
                print(
                    f"[dbg] e={epoch:03d} "
                    f"z_max={max(z_iv.abs().max().item(), z_c.abs().max().item(), z_r.abs().max().item()):.2e} "
                    f"rec_max={max(z_iv_rec.abs().max().item(), z_c_rec.abs().max().item(), z_r_rec.abs().max().item()):.2e} "
                    f"predT_min={pred_T.min().item():.2e}"
                )

            print(
                f"Epoch: {epoch+1:04d} "
                f"loss_train: {loss_train.item():.4f} "
                f"treatment_pred: {treat_loss.item():.4f} "
                f"loss_val: {loss_val.item():.4f}"
            )

            if count_patience1 > patience1:
                print("Early Stopping (stage 1)")
                break

        print("Optimization Finished For the First Stage")

        # load best stage 1 model
        best_stage1 = GDIV(input_dim, hidden_dim, num_classes).to(device)
        best_stage1.load_state_dict(torch.load(best_model_path1, map_location=device))
        best_stage1.eval()

        # =========================
        # Stage 2  (FIXED)
        # =========================
        # Compute embeddings and predicted treatment WITHOUT building a graph
        with torch.no_grad():
            z_iv, z_c, z_r, t_logp = predict(best_stage1, X, A_torch)
            prob_T = torch.exp(t_logp)
            t_pred = torch.argmax(prob_T, dim=1).unsqueeze(-1).float()

        # Detach to be explicit (safe)
        z_c = z_c.detach()
        z_r = z_r.detach()
        t_pred = t_pred.detach()

        # Outcome module
        t_dim = t_pred.size(1)  # usually 1
        pred_module = Output_MLP(t_dim, hidden_dim, output_dim).to(device)
        optimizer_2nd = Adam(pred_module.parameters(), lr=1e-3, weight_decay=1e-5)
        MSEloss = nn.MSELoss()

        patience2 = 15
        min_val_loss2 = float('inf')
        best_model_path2 = f'best_model_stage2_exp{exp_id}.pth'
        count_patience2 = 0

        for epoch in range(args.epochs):
            pred_module.train()
            optimizer_2nd.zero_grad()

            out = pred_module(t_pred[idx_train], z_c[idx_train], z_r[idx_train])  # [n_train, 1]
            Y_train = Y[idx_train].float().unsqueeze(-1)                          # [n_train, 1]
            loss_classifier = MSEloss(out, Y_train)

            loss_classifier.backward()  # <-- NO retain_graph
            torch.nn.utils.clip_grad_norm_(pred_module.parameters(), max_norm=args.clip)
            optimizer_2nd.step()

            pred_module.eval()
            with torch.no_grad():
                out_val = pred_module(t_pred[idx_val], z_c[idx_val], z_r[idx_val])
                Y_val = Y[idx_val].float().unsqueeze(-1)
                val_loss = MSEloss(out_val, Y_val)

            count_patience2 += 1
            if val_loss.item() < min_val_loss2:
                min_val_loss2 = val_loss.item()
                count_patience2 = 0
                torch.save(pred_module.state_dict(), best_model_path2)

            print(
                f"[Stage2] Epoch: {epoch+1:04d} "
                f"loss_train: {loss_classifier.item():.4f} "
                f"loss_val: {val_loss.item():.4f}"
            )

            if count_patience2 > patience2:
                print("Early Stopping (stage 2)")
                break

        # load best stage 2 model
        best_stage2 = Output_MLP(t_dim, hidden_dim, output_dim).to(device)
        best_stage2.load_state_dict(torch.load(best_model_path2, map_location=device))
        best_stage2.eval()

        # =========================
        # Evaluation
        # =========================
        t_pred_f = t_pred[idx_test]
        t_pred_cf = 1 - t_pred_f
        z_c_test = z_c[idx_test]
        z_r_test = z_r[idx_test]

        factual_outcome = best_stage2(t_pred_f, z_c_test, z_r_test)         # [n_test,1]
        counterfactual_outcome = best_stage2(t_pred_cf, z_c_test, z_r_test) # [n_test,1]

        y1_pred = torch.where(t_pred_f > 0, factual_outcome, counterfactual_outcome)
        y0_pred = torch.where(t_pred_f > 0, counterfactual_outcome, factual_outcome)

        # ensure shapes [n_test, 1]
        delta_pred = (y1_pred - y0_pred)
        delta_true = (Y1 - Y0)[idx_test].unsqueeze(-1)

        pehe_ts = torch.sqrt(MSEloss(delta_pred, delta_true))
        mae_ate_ts = torch.abs(torch.mean(delta_pred) - torch.mean(delta_true))

        mae_ate_ts_ls.append(mae_ate_ts.item())
        pehe_ts_ls.append(pehe_ts.item())

        print(
            "Test set results:",
            f"pehe_ts= {pehe_ts.item():.4f}",
            f"mae_ate_ts= {mae_ate_ts.item():.4f}",
        )

    print("PEHE mean/std:", np.mean(pehe_ts_ls), np.std(pehe_ts_ls))
    print("MAE(ATE) mean/std:", np.mean(mae_ate_ts_ls), np.std(mae_ate_ts_ls))
