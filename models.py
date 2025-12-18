import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Utils / Layers
# -------------------------
def glorot_init(input_dim: int, output_dim: int) -> nn.Parameter:
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    w = torch.empty(input_dim, output_dim).uniform_(-init_range, init_range)
    return nn.Parameter(w)


class GraphConvolutionSparse(nn.Module):
    """
    A simple GCN layer that works with a torch.sparse adjacency matrix A (NxN).
    x: [N, Fin]
    A: torch.sparse.FloatTensor [N, N]
    output: [N, Fout]
    """
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.0, act=F.relu):
        super().__init__()
        self.weight = glorot_init(input_dim, output_dim)
        self.dropout = dropout
        self.act = act

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = F.dropout(x, self.dropout, training=self.training)          # [N, Fin]
        x = torch.mm(x, self.weight)                                    # [N, Fout]
        x = torch.sparse.mm(adj, x)                                     # [N, Fout]
        return self.act(x)


class InnerProductDecoderLogits(nn.Module):
    """
    Returns logits (NOT sigmoid) so you can use BCEWithLogitsLoss.
    z: [N, d]
    out: [N, N] logits
    """
    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = dropout

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = F.dropout(z, self.dropout, training=self.training)
        return torch.mm(z, z.t())  # logits


class Graphite(nn.Module):
    """
    GraphITE-style refinement layer that uses two dense support matrices recon_1, recon_2 (NxN).
    x: [N, d]
    recon_1/recon_2: [N, N] dense
    """
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.0, act=F.relu):
        super().__init__()
        self.weight = glorot_init(input_dim, output_dim)
        self.dropout = dropout
        self.act = act

    def forward(self, x: torch.Tensor, recon_1: torch.Tensor, recon_2: torch.Tensor) -> torch.Tensor:
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.mm(x, self.weight)  # [N, out_dim]
        # x <- R1 R1^T x + R2 R2^T x
        x = torch.mm(recon_1, torch.mm(recon_1.t(), x)) + torch.mm(recon_2, torch.mm(recon_2.t(), x))
        return self.act(x)


# -------------------------
# GDIV
# -------------------------
class GDIV(nn.Module):
    """
    Drop-in GDIV that matches your main.py expectations.

    Forward returns:
      z_iv, z_iv_rec, z_c, z_c_rec, z_r, z_r_rec, mi_total_loss, pred_T

    - A is assumed to be a torch.sparse adjacency matrix [N, N]
    - recon outputs are logits [N, N] for BCEWithLogitsLoss
    - pred_T is log-prob [N, num_classes] for NLLLoss
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        use_graphite_refine: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout = dropout
        self.use_graphite_refine = use_graphite_refine

        # Shared first layer (helps keep parameter count reasonable)
        self.enc_shared = GraphConvolutionSparse(input_dim, hidden_dim, dropout=dropout, act=F.relu)

        # Three factor-specific heads
        self.enc_iv = GraphConvolutionSparse(hidden_dim, hidden_dim, dropout=dropout, act=F.relu)
        self.enc_c  = GraphConvolutionSparse(hidden_dim, hidden_dim, dropout=dropout, act=F.relu)
        self.enc_r  = GraphConvolutionSparse(hidden_dim, hidden_dim, dropout=dropout, act=F.relu)

        # Decoders for adjacency reconstruction (logits)
        self.dec_iv = InnerProductDecoderLogits(dropout=dropout)
        self.dec_c  = InnerProductDecoderLogits(dropout=dropout)
        self.dec_r  = InnerProductDecoderLogits(dropout=dropout)

        # Optional GraphITE refinement modules (operate on dense NxN supports)
        # They map hidden_dim -> hidden_dim
        self.graphite_iv = Graphite(hidden_dim, hidden_dim, dropout=dropout, act=F.relu)
        self.graphite_c  = Graphite(hidden_dim, hidden_dim, dropout=dropout, act=F.relu)
        self.graphite_r  = Graphite(hidden_dim, hidden_dim, dropout=dropout, act=F.relu)

        # Treatment predictor uses concat([z_iv, z_c])
        self.treatment_pred = nn.Linear(2 * hidden_dim, num_classes)

    # These 3 methods are exactly what your predict() calls.
    def encoder_z_iv(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        h = self.enc_shared(X, A)
        return self.enc_iv(h, A)

    def encoder_z_c(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        h = self.enc_shared(X, A)
        return self.enc_c(h, A)

    def encoder_z_r(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        h = self.enc_shared(X, A)
        return self.enc_r(h, A)

    def forward(self, X: torch.Tensor, A: torch.Tensor):
        # --- 1) Initial encodings
        z_iv0 = self.encoder_z_iv(X, A)  # [N, d]
        z_c0  = self.encoder_z_c(X, A)
        z_r0  = self.encoder_z_r(X, A)

        # --- 2) Initial reconstructions (logits)
        z_iv_rec0 = self.dec_iv(z_iv0)   # [N, N] logits
        z_c_rec0  = self.dec_c(z_c0)
        z_r_rec0  = self.dec_r(z_r0)

        # --- 3) Optional GraphITE refinement
        if self.use_graphite_refine:
            # GraphITE needs dense supports.
            # If your graph is large, this can be expensive!
            A_dense = A.to_dense()

            # Use probabilities as supports (more stable than raw logits)
            A_iv_sup = torch.sigmoid(z_iv_rec0)
            A_c_sup  = torch.sigmoid(z_c_rec0)
            A_r_sup  = torch.sigmoid(z_r_rec0)

            # Refine each factor using (observed adjacency, its own reconstructed adjacency)
            z_iv = self.graphite_iv(z_iv0, A_dense, A_iv_sup)
            z_c  = self.graphite_c(z_c0,  A_dense, A_c_sup)
            z_r  = self.graphite_r(z_r0,  A_dense, A_r_sup)

            # Re-decode from refined embeddings (so your rec loss matches refined z)
            z_iv_rec = self.dec_iv(z_iv)
            z_c_rec  = self.dec_c(z_c)
            z_r_rec  = self.dec_r(z_r)
        else:
            z_iv, z_c, z_r = z_iv0, z_c0, z_r0
            z_iv_rec, z_c_rec, z_r_rec = z_iv_rec0, z_c_rec0, z_r_rec0

        # --- 4) Treatment prediction (log-prob for NLLLoss)
        concat = torch.cat([z_iv, z_c], dim=1)              # [N, 2d]
        pred_T = F.log_softmax(self.treatment_pred(concat), dim=1)

        # --- 5) MI / balancing loss placeholder (keep API compatible)
        mi_total_loss = torch.tensor(0.0, device=X.device)

        return z_iv, z_iv_rec, z_c, z_c_rec, z_r, z_r_rec, mi_total_loss, pred_T


# -------------------------
# Stage-2 outcome head
# -------------------------
class Output_MLP(nn.Module):
    """
    Your main.py calls: Output_MLP(input_dim, hidden_dim, output_dim)
    where input_dim == t_pred.size(1) == 1 typically.

    We assume z_c and z_r are both of dimension hidden_dim (same as encoder hidden_dim),
    which matches your main.py usage.
    """
    def __init__(self, t_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = dropout
        in_dim = t_dim + 2 * hidden_dim

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, t_pred: torch.Tensor, z_c: torch.Tensor, z_r: torch.Tensor) -> torch.Tensor:
        x = torch.cat([t_pred, z_c, z_r], dim=1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x
