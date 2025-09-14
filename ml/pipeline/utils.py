import json
from datetime import datetime
from typing import Iterable, Sequence

import numpy as np
import torch


def save_json(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f)


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_numpy(arr, path):
    np.save(path, arr)


def load_numpy(path):
    return np.load(path)


def save_torch(obj, path):
    torch.save(obj, path)


def load_torch(path, device='cpu'):
    return torch.load(path, map_location=device)


def build_id_to_index_map(ids, start: int = 0):
    """Create a mapping from identifier to sequential index.

    Parameters
    ----------
    ids : Iterable
        Sequence of identifiers to map.
    start : int, optional
        Starting index for the mapping. Use ``start=1`` when reserving
        index ``0`` for padding, by default ``0``.
    """
    return {id_: idx + start for idx, id_ in enumerate(ids)}


def get_current_utc_date() -> datetime:
    return datetime.utcnow()


def compute_alpha(half_life_days: float = 30.0) -> float:
    return np.log(2.0) / half_life_days


def save_checkpoint(state, path: str) -> None:
    torch.save(state, path)


