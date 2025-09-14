import torch
import torch.nn as nn


class EmbeddingTables:
    def __init__(self, num_users: int, num_items: int):
        """Initialize embedding tables with padding index 0.

        The 0 index is reserved as a padding/unknown token, so real users and
        items are shifted by +1. This lets downstream code map missing IDs to
        0 and keeps the padding vector fixed to zeros during training.
        """

        # +1 to account for the padding index. ``padding_idx`` ensures the
        # first row stays zeros and is ignored by gradient updates.
        self.user_emb = nn.Embedding(num_users + 1, 128, padding_idx=0)
        self.item_emb = nn.Embedding(num_items + 1, 128, padding_idx=0)
        self.mean_user_emb = torch.zeros(128)
        self.mean_item_emb = torch.zeros(128)

    def save(self, path_prefix: str):
        torch.save(self.user_emb.state_dict(), f'{path_prefix}_user.pth')
        torch.save(self.item_emb.state_dict(), f'{path_prefix}_item.pth')
        torch.save(self.mean_user_emb, f'{path_prefix}_mean_user.pt')
        torch.save(self.mean_item_emb, f'{path_prefix}_mean_item.pt')

    def load(self, path_prefix: str, device='cpu'):
        user_state = torch.load(f'{path_prefix}_user.pth', map_location=device)
        item_state = torch.load(f'{path_prefix}_item.pth', map_location=device)

        num_users = user_state['weight'].shape[0]
        num_items = item_state['weight'].shape[0]

        self.user_emb = nn.Embedding(num_users, 128, padding_idx=0).to(device)
        self.item_emb = nn.Embedding(num_items, 128, padding_idx=0).to(device)
        self.user_emb.load_state_dict(user_state)
        self.item_emb.load_state_dict(item_state)
        self.mean_user_emb = torch.load(f'{path_prefix}_mean_user.pt', map_location=device)
        self.mean_item_emb = torch.load(f'{path_prefix}_mean_item.pt', map_location=device)

    def compute_means(self):
        # Skip the padding row when computing means so it does not skew the
        # averages used as fallbacks for unknown IDs.
        self.mean_user_emb = self.user_emb.weight.data[1:].mean(dim=0)
        self.mean_item_emb = self.item_emb.weight.data[1:].mean(dim=0)

