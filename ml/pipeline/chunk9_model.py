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
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 512)
        self.bn1 = nn.LayerNorm(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.LayerNorm(256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.fc3(x)
        return x


class ScoreBilinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.bilinear = nn.Bilinear(256, 256, 1)

    def forward(self, x):
        user = x[:, :256]
        item = x[:, 256:]
        return self.bilinear(user, item)

