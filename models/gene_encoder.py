import torch
import torch.nn as nn
import torch.nn.functional as F

GENETICS_EMBEDDING_SIZE=1024

class Gen_MLP(nn.Module):
    def __init__(
            self,
            n_feats,  # this can change depending on what kind of feature dimension reduction was done
            n_hidden1_u=1024,
            n_hidden2_u=1024,
            eps=1e-05,
            out_dim=128
    ):
        super().__init__()
        # 1st hidden layer
        self.hidden_1 = nn.Linear(n_feats, n_hidden1_u)
        self.bn1 = nn.BatchNorm1d(num_features=n_hidden1_u, eps=eps)
        # self.ln1 = nn.LayerNorm(normalized_shape=n_hidden1_u, eps=eps)

        # 2nd hidden layer
        self.hidden_2 = None
        if n_hidden2_u is not None:
            self.hidden_2 = nn.Linear(n_hidden1_u, n_hidden2_u)
            self.bn2 = nn.BatchNorm1d(num_features=n_hidden2_u, eps=eps)
            # self.ln2 = nn.LayerNorm(normalized_shape=n_hidden2_u, eps=eps)
        
        gen_embed_size = GENETICS_EMBEDDING_SIZE

        # projection MLP for genetics model
        self.genetics_l1 = nn.Linear(gen_embed_size, GENETICS_EMBEDDING_SIZE)
        self.genetics_l2 = nn.Linear(GENETICS_EMBEDDING_SIZE, out_dim)

    def forward(self, x):
        z1 = self.hidden_1(x)
        a1 = torch.relu(z1)
        a1 = self.bn1(a1)
        outputs = a1

        if self.hidden_2 is not None:
            z2 = self.hidden_2(a1)
            a2 = torch.relu(z2)
            a2 = self.bn2(a2)
            outputs = a2

        x = self.genetics_l1(outputs)
        x = F.relu(x)
        out_emb = self.genetics_l2(x)

        return out_emb