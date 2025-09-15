import torch
import torch.nn as nn
import torch.nn.functional as F


class UserMetaFC(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        # BatchNorm1d requires a batch size > 1 when training which isn't
        # guaranteed in the pairwise ranking loop. LayerNorm works on a single
        # example, so switch to it to avoid runtime errors.
        self.bn1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 128)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class ScoreMLP(nn.Module):
    def __init__(self, num_users: int, num_items: int):
        super().__init__()
        # Reserve zero index for missing users/items
        self.user_bias = nn.Embedding(num_users, 1, padding_idx=0)
        self.item_bias = nn.Embedding(num_items, 1, padding_idx=0)

        self.fc1 = nn.Linear(512, 512)
        self.bn1 = nn.LayerNorm(512)
        self.fc2 = nn.Linear(512, 512)
        self.bn2 = nn.LayerNorm(512)
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.LayerNorm(256)
        self.fc4 = nn.Linear(256, 1)

    def forward(
        self,
        x,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
        user_idx: torch.LongTensor | None = None,
        item_idx: torch.LongTensor | None = None,
    ):
        dp = (user_emb * item_emb).sum(dim=1, keepdim=True)

        h = self.fc1(x)
        h = self.bn1(h)
        h = F.relu(h)
        h = F.dropout(h, p=0.3, training=self.training)

        res = h
        h = self.fc2(h)
        h = self.bn2(h)
        h = F.relu(h)
        h = F.dropout(h, p=0.3, training=self.training)
        h = h + res

        h = self.fc3(h)
        h = self.bn3(h)
        h = F.relu(h)
        h = F.dropout(h, p=0.3, training=self.training)
        out = self.fc4(h)
        out = out + dp

        if user_idx is not None:
            out = out + self.user_bias(user_idx)
        if item_idx is not None:
            out = out + self.item_bias(item_idx)
        return out

