import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import load_numpy, save_numpy, save_torch, save_json
import json


class ItemMetaFC(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
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
    # Load stationary structured features (already scaled) and row index mapping
    X_structured = load_numpy('ml/data/X_structured.npy')
    with open('ml/data/appid_to_rowidx.json', 'r', encoding='utf-8') as f:
        appid_to_rowidx = {int(k): int(v) for k, v in json.load(f).items()}

    # Load tag/text embeddings and their app_id arrays
    T_pca_norm = load_numpy('ml/data/T_pca_norm.npy')
    E_pca = load_numpy('ml/data/E_pca_all.npy')
    tag_app_ids = load_numpy('ml/data/tag_pca_app_ids.npy')

    tag_idx_map = {int(aid): idx for idx, aid in enumerate(tag_app_ids)}
    # Build reverse mapping rowidx->appid to iterate in structured order
    num_rows = len(appid_to_rowidx)
    row_to_appid = [None] * num_rows
    for aid, ridx in appid_to_rowidx.items():
        if 0 <= ridx < num_rows:
            row_to_appid[ridx] = int(aid)

    structured_f = []
    T_f = []
    E_f = []
    app_ids_f = []
    for ridx, aid in enumerate(row_to_appid):
        if aid is None:
            continue
        t_idx = tag_idx_map.get(int(aid))
        if t_idx is None:
            continue
        structured_f.append(X_structured[ridx])
        T_f.append(T_pca_norm[t_idx])
        E_f.append(E_pca[t_idx])
        app_ids_f.append(aid)

    structured = np.array(structured_f)
    T_pca_norm = np.array(T_f)
    E_pca = np.array(E_f)
    app_ids = np.array(app_ids_f)
    # X_structured is already scaled; concatenate directly with tag/text PCs
    item_meta_raw = np.concatenate([structured, T_pca_norm, E_pca], axis=1)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ItemMetaFC(item_meta_raw.shape[1]).to(device)
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

