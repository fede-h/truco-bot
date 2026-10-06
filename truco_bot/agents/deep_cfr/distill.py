"""Distillation pipeline: extracts state-strategy pairs from CFR table and trains PolicyNetwork."""

from __future__ import annotations

import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from truco_engine import BitboardState, SharedPolicyTable, get_policy_distribution

from truco_bot.agents.deep_cfr.encoder import STATE_FEATURE_DIM, encode_state
from truco_bot.agents.deep_cfr.network import PolicyNetwork


def extract_samples(
    table: SharedPolicyTable,
    num_samples: int = 50_000,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract (features, targets, masks) training tensors from CFR table via random rollouts."""
    rng = random.Random(seed)
    features_list: list[torch.Tensor] = []
    targets_list: list[list[float]] = []
    masks_list: list[list[bool]] = []

    while len(features_list) < num_samples:
        cards = rng.sample(range(40), 6)
        mano = rng.choice([0, 1])
        bstate = BitboardState([cards[:3], cards[3:]], mano)

        # Rollout the hand until termination
        while not bstate.is_done and len(features_list) < num_samples:
            dist = get_policy_distribution(bstate, table)
            if not dist:
                break

            feat = encode_state(bstate)
            target = [0.0] * 8
            mask = [False] * 8
            for i, (_, prob) in enumerate(dist[:8]):
                target[i] = prob
                mask[i] = True

            features_list.append(feat)
            targets_list.append(target)
            masks_list.append(mask)

            # ponytail: sample action from CFR distribution to step trajectory
            probs = [p for _, p in dist]
            actions = [a for a, _ in dist]
            chosen_action = rng.choices(actions, weights=probs, k=1)[0]
            bstate = bstate.step(chosen_action)

    features = torch.stack(features_list, dim=0)
    targets = torch.tensor(targets_list, dtype=torch.float32)
    masks = torch.tensor(masks_list, dtype=torch.bool)
    return features, targets, masks


def train_policy_network(
    policy_net: PolicyNetwork,
    features: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cpu",
) -> PolicyNetwork:
    """Train PolicyNetwork using supervised Cross-Entropy against CFR target distributions."""
    policy_net = policy_net.to(device)
    policy_net.train()

    optimizer = torch.optim.AdamW(policy_net.parameters(), lr=lr, weight_decay=weight_decay)
    dataset = TensorDataset(features, targets, masks)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for _ in range(epochs):
        for batch_x, batch_y, batch_m in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_m = batch_m.to(device)

            probs = policy_net(batch_x, mask=batch_m)
            # ponytail: masked cross-entropy loss against soft targets
            loss = -torch.sum(batch_y * torch.log(probs + 1e-8)) / batch_x.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    policy_net.eval()
    return policy_net


def distill_table(
    checkpoint_path: str | Path,
    output_path: str | Path,
    num_samples: int = 50_000,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 42,
    capacity: int = 67_108_864,
    device: str = "cpu",
) -> PolicyNetwork:
    """Full distillation pipeline: loads binary CFR table, extracts samples, and trains network."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    table = SharedPolicyTable(capacity)
    table.load_from_file(str(path))

    features, targets, masks = extract_samples(table, num_samples=num_samples, seed=seed)
    policy_net = PolicyNetwork(in_dim=STATE_FEATURE_DIM, num_actions=8)
    train_policy_network(
        policy_net,
        features,
        targets,
        masks,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy_net.state_dict(), str(out))
    return policy_net
