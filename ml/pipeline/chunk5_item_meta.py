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
    E_qwen3 = load_numpy('ml/data/E_qwen3.npy')
    print(f"E_qwen3 loaded with shape {E_qwen3.shape} and dtype {E_qwen3.dtype}")
    e_norms = np.linalg.norm(E_qwen3, axis=1)
    zero_norms = np.sum(e_norms == 0)
    if zero_norms:
        print(f"E_qwen3 zero-norm rows: {zero_norms}")
    e_norms[e_norms == 0] = 1
    E_qwen3 = E_qwen3 / e_norms[:, None]
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
        E_f.append(E_qwen3[t_idx])
        app_ids_f.append(aid)

    structured = np.array(structured_f)
    T_pca_norm = np.array(T_f)
    E_qwen3 = np.array(E_f)
    app_ids = np.array(app_ids_f)
    print(f"X_structured dims: {structured.shape}")
    print(f"T_pca_norm dims: {T_pca_norm.shape}")
    print(f"E_qwen3 dims: {E_qwen3.shape}")
    # X_structured is already scaled; concatenate directly with tag PCs and text embeddings
    item_meta_raw = np.concatenate([structured, T_pca_norm, E_qwen3], axis=1)
    print(f"item_meta_raw shape: {item_meta_raw.shape}")
    norms = np.linalg.norm(item_meta_raw, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    item_meta_raw = item_meta_raw / norms
    print(
        f"Sample post-normalization norm: {np.linalg.norm(item_meta_raw[0]):.4f}"
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ItemMetaFC(item_meta_raw.shape[1]).to(device)
    model.eval()
    batch_size = 20000
    total_batches = (len(item_meta_raw) + batch_size - 1) // batch_size
    emb_chunks = []
    with torch.no_grad():
        for i in range(total_batches):
            start = i * batch_size
            end = min(start + batch_size, len(item_meta_raw))
            batch = torch.from_numpy(item_meta_raw[start:end]).float().to(device)
            emb_chunks.append(model(batch).cpu())
            print(f"Processed batch {i+1}/{total_batches}")
    embs = torch.cat(emb_chunks, dim=0)
    print(f"ItemMetaFC output shape: {embs.shape}")
    embs_np = embs.numpy().astype('float32')
    os.makedirs('ml/data', exist_ok=True)
    save_numpy(embs_np, 'ml/data/item_meta_embs.npy')
    save_torch(model.state_dict(), 'ml/data/item_meta_fc.pth')
    mapping = {int(app_id): int(idx) for idx, app_id in enumerate(app_ids)}
    save_json(mapping, 'ml/data/app_id_to_meta_index.json')
    schema = {
        'structured': int(structured.shape[1]),
        'tag': int(T_pca_norm.shape[1]),
        'text': int(E_qwen3.shape[1]),
        'final': int(embs_np.shape[1]),
    }
    print(f"Schema: {schema}")
    save_json(schema, 'ml/data/item_meta_schema.json')
    print(
        f"✅ Processed {len(app_ids)} items. Outputs saved to ml/data/"
    )

