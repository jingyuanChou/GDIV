import torch
import torch.nn as nn
import torch.nn.functional as F


class OptimizerVAE:
    def __init__(self, model, num_nodes, pos_weight, norm, learning_rate=0.01):
        self.model = model
        self.num_nodes = num_nodes
        self.pos_weight = pos_weight
        self.norm = norm
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def compute_loss(self, preds, labels):
        # Assuming that preds and labels are torch tensors
        pos_weight = torch.full_like(labels, self.pos_weight)
        loss = F.binary_cross_entropy_with_logits(preds, labels, weight=pos_weight, reduction='mean')
        return self.norm * loss

    def step(self, x, edge_index, labels):
        self.model.train()
        self.optimizer.zero_grad()

        preds, mu, logvar = self.model(x, edge_index)
        labels = labels.view_as(preds)  # Ensure labels are the same shape as preds

        # Standard BCE Loss
        recon_loss = self.compute_loss(preds, labels)

        # KL Divergence
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / self.num_nodes

        # Total loss
        loss = recon_loss + kl_div

        # Backpropagation
        loss.backward()
        self.optimizer.step()

        # Calculate accuracy as well (if relevant)
        preds_bin = (preds > 0.5).float()  # Using 0.5 as threshold
        correct_prediction = (preds_bin == labels).float()
        accuracy = correct_prediction.mean()

        return loss.item(), accuracy.item()
