import os
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.decomposition import PCA

from .utils import save_numpy, save_torch


def embed_texts(games_df: pd.DataFrame, device: str = 'cpu') -> np.ndarray:
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased').to(device)
    model.eval()

    embeddings = []
    with torch.no_grad():
        for _, row in games_df.iterrows():
            text_short = row['short_description'] or ''
            text_long = row['long_description'] or ''
            tokens_short = tokenizer(text_short, return_tensors='pt', truncation=True, max_length=128).to(device)
            out_short = model(**tokens_short)
            cls_short = out_short.last_hidden_state[:,0,:]

            tokens_long = tokenizer(text_long, return_tensors='pt', truncation=True, max_length=512).to(device)
            out_long = model(**tokens_long)
            mean_long = out_long.last_hidden_state.mean(dim=1)

            e_raw = torch.cat([cls_short, mean_long], dim=1)
            embeddings.append(e_raw.cpu().numpy().squeeze())
    return np.stack(embeddings, axis=0)


def run_text_pca(games_df: pd.DataFrame) -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    E_raw = embed_texts(games_df, device)
    pca = PCA(n_components=256)
    E_pca = pca.fit_transform(E_raw)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    if cumsum[255] < 0.90:
        raise RuntimeError(f'Text-PCA covers only {cumsum[255]*100:.1f}% variance')

    os.makedirs('data', exist_ok=True)
    save_numpy(E_pca.astype('float32'), 'data/E_pca_all.npy')
    save_torch(pca, 'data/pca_text.pkl')
    print('✅ Chunk 4 text embeddings saved')

