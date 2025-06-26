import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import load_numpy, save_numpy, save_torch, save_json


class ItemMetaFC(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(522, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)

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


def run_item_meta_embedding() -> None:
    structured = load_numpy('ml/data/structured_raw.npy')
    app_ids = load_numpy('ml/data/structured_app_ids.npy')
    s_min = load_numpy('ml/data/structured_min.npy')
    s_max = load_numpy('ml/data/structured_max.npy')
    T_pca_norm = load_numpy('ml/data/T_pca_norm.npy')
    E_pca = load_numpy('ml/data/E_pca_all.npy')

    denom = s_max - s_min
    denom[denom == 0] = 1e-6
    structured_norm = (structured - s_min) / denom
    item_meta_raw = np.concatenate([structured_norm, T_pca_norm, E_pca], axis=1)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ItemMetaFC().to(device)
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(item_meta_raw).float().to(device)
        embs = model(x)
    embs_np = embs.cpu().numpy().astype('float32')
    os.makedirs('ml/data', exist_ok=True)
    save_numpy(embs_np, 'ml/data/item_meta_embs.npy')
    save_torch(model.state_dict(), 'ml/data/item_meta_fc.pth')
    mapping = {int(app_id): int(idx) for idx, app_id in enumerate(app_ids)}
    save_json(mapping, 'ml/data/app_id_to_meta_index.json')
    print('✅ Chunk 5 item meta embeddings saved')

