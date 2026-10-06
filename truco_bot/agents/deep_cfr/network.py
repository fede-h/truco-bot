"""ResMLP PolicyNetwork with action masking for Deep CFR."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ResBlock(nn.Module):
    """Residual block with LayerNorm and GELU."""

    def __init__(self, dim: int = 256) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ponytail: standard residual connection
        return x + self.fc2(self.act(self.norm(self.fc1(x))))


class PolicyNetwork(nn.Module):
    """Deep CFR Policy Network with 156-dim input and 8-dim masked policy head."""

    def __init__(self, in_dim: int = 156, num_actions: int = 8, hidden_dim: int = 256) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.num_actions = num_actions
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            ResBlock(hidden_dim),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass supporting unbatched and batched inputs with optional action mask."""
        unbatched = x.dim() == 1
        if unbatched:
            x = x.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)

        logits = self.net(x)

        if mask is None:
            probs = F.softmax(logits, dim=-1)
        else:
            # ponytail: convert float masks to boolean if needed
            bool_mask = (mask > 0) if mask.dtype != torch.bool else mask
            all_false = ~bool_mask.any(dim=-1, keepdim=True)

            masked_logits = logits.masked_fill(~bool_mask, -1e9)
            probs = F.softmax(masked_logits, dim=-1)

            # ponytail: enforce strict 0.0 on illegal actions while preserving gradients
            probs = torch.where(bool_mask, probs, torch.zeros_like(probs))
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)

            # ponytail: fallback to uniform distribution if entire mask is empty
            if all_false.any():
                uniform = torch.full_like(probs, 1.0 / self.num_actions)
                probs = torch.where(all_false, uniform, probs)

        if unbatched:
            probs = probs.squeeze(0)
        return probs
