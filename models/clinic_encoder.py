import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(
            self,
            n_feats, 
            eps=1e-05,
            out_dim=128
    ):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(n_feats, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim)
        )

    def forward(self, x):
        out_emb = self.mlp(x)
        return out_emb

