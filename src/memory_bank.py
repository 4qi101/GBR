"""Momentum-based binary memory bank utilities for cross-modal hashing."""

from __future__ import annotations

import torch
import torch.nn.functional as F


class MomentumBinaryMemoryBank:
    """Maintains a momentum-updated memory bank for hash targets."""

    def __init__(
        self,
        num_samples: int,
        code_dim: int,
        momentum: float = 0.99,
        device: torch.device | str | None = None,
        init_std: float = 0.01,
    ) -> None:
        self.num_samples = num_samples
        self.code_dim = code_dim
        self.momentum = momentum
        self.device = torch.device(device or "cpu")
        self.bank = torch.empty(num_samples, code_dim, device=self.device)
        self.bank.normal_(mean=0.0, std=init_std)

    def __len__(self) -> int:
        return self.num_samples

    def to(self, device: torch.device | str) -> "MomentumBinaryMemoryBank":
        self.device = torch.device(device)
        self.bank = self.bank.to(self.device)
        return self

    @torch.no_grad()
    def update(self, indices: torch.Tensor, features: torch.Tensor) -> None:
        if indices.numel() == 0:
            return
        idx = indices.to(self.device, dtype=torch.long)
        feats = F.normalize(features.detach(), dim=1, eps=1e-12)
        if feats.device != self.device:
            feats = feats.to(self.device)
        updated = self.momentum * self.bank[idx] + (1.0 - self.momentum) * feats
        updated = F.normalize(updated, dim=1, eps=1e-12)
        self.bank[idx] = updated

    def _get_continuous(self, indices: torch.Tensor) -> torch.Tensor:
        idx = indices.to(self.device, dtype=torch.long)
        return self.bank[idx]

    def _get_binary(self, indices: torch.Tensor) -> torch.Tensor:
        vec = torch.sign(self._get_continuous(indices))
        vec[vec == 0] = 1.0
        return F.normalize(vec, dim=1, eps=1e-12)

    def _available_indices(self, exclude: torch.Tensor | None = None) -> torch.Tensor:
        if exclude is None or exclude.numel() == 0:
            return torch.arange(self.num_samples, device=self.device)
        mask = torch.ones(self.num_samples, dtype=torch.bool, device=self.device)
        mask[exclude.to(self.device, dtype=torch.long)] = False
        if mask.any():
            return mask.nonzero(as_tuple=False).squeeze(1)
        return torch.arange(self.num_samples, device=self.device)

    def sample_negative_indices(
        self,
        num_negatives: int,
        exclude: torch.Tensor | None = None,
    ) -> torch.Tensor:
        candidates = self._available_indices(exclude)
        if num_negatives >= candidates.numel():
            repeats = (num_negatives + candidates.numel() - 1) // candidates.numel()
            idx = candidates.repeat(repeats)[:num_negatives]
        else:
            perm = torch.randperm(candidates.numel(), device=self.device)
            idx = candidates[perm[:num_negatives]]
        return idx

    def contrastive_loss(
        self,
        queries: torch.Tensor,
        indices: torch.Tensor,
        num_negatives: int = 1024,
        temperature: float = 0.07,
    ) -> torch.Tensor:
        if self.num_samples == 0 or queries.numel() == 0:
            return torch.tensor(0.0, device=queries.device)

        queries = F.normalize(queries, dim=1, eps=1e-12)
        pos_keys = self._get_binary(indices).to(queries.device)

        if num_negatives <= 0:
            pos_logits = torch.sum(queries * pos_keys, dim=1) / temperature
            return -F.logsigmoid(pos_logits).mean()

        neg_idx = self.sample_negative_indices(num_negatives, exclude=indices)
        neg_keys = self._get_binary(neg_idx).to(queries.device)

        pos_logits = torch.sum(queries * pos_keys, dim=1, keepdim=True) / temperature
        neg_logits = queries @ neg_keys.t() / temperature

        logits = torch.cat([pos_logits, neg_logits], dim=1)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=queries.device)
        return F.cross_entropy(logits, labels)
