"""Tests for Deep CFR PolicyNetwork."""

import pytest
import torch
from torch import nn
from truco_bot.agents.deep_cfr.network import PolicyNetwork


def test_policy_network_initialization():
    """Verify PolicyNetwork(in_dim=156, num_actions=8) initialization and structure."""
    net = PolicyNetwork(in_dim=156, num_actions=8)

    assert isinstance(net, nn.Module)
    assert hasattr(net, "in_dim")
    assert net.in_dim == 156
    assert hasattr(net, "num_actions")
    assert net.num_actions == 8

    # Verify model has trainable parameters
    params = list(net.parameters())
    assert len(params) > 0
    total_params = sum(p.numel() for p in params)
    assert total_params > 1000  # Non-trivial architecture
    assert all(p.requires_grad for p in params)


def test_policy_network_forward_shapes():
    """Verify forward pass supports unbatched and batched inputs."""
    net = PolicyNetwork(in_dim=156, num_actions=8)

    # Unbatched (156,)
    x_single = torch.randn(156)
    out_single = net(x_single)
    assert out_single.shape == (8,)
    assert torch.isfinite(out_single).all()

    # Batched (32, 156)
    x_batch = torch.randn(32, 156)
    out_batch = net(x_batch)
    assert out_batch.shape == (32, 8)
    assert torch.isfinite(out_batch).all()


def test_policy_network_action_masking_invariants():
    """Verify masked actions have exactly 0.0 probability and legal actions sum to 1.0."""
    net = PolicyNetwork(in_dim=156, num_actions=8)
    x = torch.randn(156)

    # 3 legal actions (indices 0, 1, 4), 5 illegal actions
    mask = torch.tensor([True, True, False, False, True, False, False, False], dtype=torch.bool)
    probs = net(x, mask=mask)

    # Invariant 1: Illegal actions must have EXACTLY 0.0 probability
    assert (probs[~mask] == 0.0).all(), "Illegal actions must have exactly 0.0 probability"

    # Invariant 2: Legal actions sum to 1.0
    assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-6), "Policy must sum to 1.0"

    # Invariant 3: Legal actions have positive probabilities
    assert (probs[mask] > 0.0).all(), "Legal actions must have strictly positive probability"


def test_policy_network_batched_action_masking():
    """Verify batched forward pass with per-sample action masks."""
    net = PolicyNetwork(in_dim=156, num_actions=8)
    x_batch = torch.randn(4, 156)

    masks = torch.tensor(
        [
            [True, False, False, False, False, False, False, False],
            [True, True, True, False, False, False, False, False],
            [False, True, False, True, False, False, False, False],
            [True, True, True, True, True, True, True, True],
        ],
        dtype=torch.bool,
    )

    probs = net(x_batch, mask=masks)

    assert probs.shape == (4, 8)
    assert (probs[~masks] == 0.0).all()
    assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-6)


def test_policy_network_action_masking_edge_cases():
    """Verify masking edge cases: single legal action, all legal actions, float mask."""
    net = PolicyNetwork(in_dim=156, num_actions=8)
    x = torch.randn(156)

    # Edge case 1: Single legal action (forced move)
    mask_single = torch.tensor(
        [False, False, True, False, False, False, False, False], dtype=torch.bool
    )
    probs_single = net(x, mask=mask_single)
    assert probs_single[2].item() == pytest.approx(1.0, abs=1e-6)
    assert (probs_single[:2] == 0.0).all()
    assert (probs_single[3:] == 0.0).all()

    # Edge case 2: All legal actions
    mask_all = torch.ones(8, dtype=torch.bool)
    probs_all = net(x, mask=mask_all)
    assert torch.isclose(probs_all.sum(), torch.tensor(1.0), atol=1e-6)
    assert (probs_all > 0.0).all()

    # Edge case 3: Float mask support
    mask_float = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    probs_float = net(x, mask=mask_float)
    assert probs_float[1].item() == 0.0
    assert (probs_float[3:] == 0.0).all()
    assert torch.isclose(probs_float.sum(), torch.tensor(1.0), atol=1e-6)


def test_policy_network_backpropagation_gradients():
    """Verify cross-entropy loss between predicted distribution and target yields non-zero gradients."""
    net = PolicyNetwork(in_dim=156, num_actions=8)

    x = torch.randn(16, 156)
    mask = torch.tensor(
        [[True, True, True, False, False, False, False, False]] * 16, dtype=torch.bool
    )
    target_policy = torch.tensor(
        [[0.5, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0]] * 16, dtype=torch.float32
    )

    probs = net(x, mask=mask)
    # Cross-entropy loss on valid action distribution
    loss = -torch.sum(target_policy * torch.log(probs + 1e-8)) / 16.0

    loss.backward()

    # Verify all trainable parameters receive valid non-zero gradients
    for name, param in net.named_parameters():
        assert param.grad is not None, f"Gradient for {name} is None"
        assert torch.isfinite(param.grad).all(), f"Gradient for {name} contains NaN/Inf"
        assert (param.grad.abs() > 0.0).any(), f"Gradient for {name} has all zeros"


def test_policy_network_gradient_step_updates_weights():
    """Verify optimizer step actually updates parameters across all layers."""
    net = PolicyNetwork(in_dim=156, num_actions=8)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

    initial_params = [p.clone().detach() for p in net.parameters()]

    x = torch.randn(8, 156)
    mask = torch.ones(8, 8, dtype=torch.bool)
    target = torch.full((8, 8), 0.125)

    probs = net(x, mask=mask)
    loss = -torch.sum(target * torch.log(probs + 1e-8)) / 8.0

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Verify at least one weight in each parameter tensor has been updated
    for init_p, curr_p in zip(initial_params, net.parameters(), strict=True):
        assert not torch.equal(init_p, curr_p), "Weight tensor was not updated by optimizer step"


def test_policy_network_reproducibility():
    """Verify seed reproducibility for initialization and forward evaluation."""
    torch.manual_seed(1234)
    net_a = PolicyNetwork(in_dim=156, num_actions=8)

    torch.manual_seed(1234)
    net_b = PolicyNetwork(in_dim=156, num_actions=8)

    # Weights match
    for p_a, p_b in zip(net_a.parameters(), net_b.parameters(), strict=True):
        assert torch.equal(p_a, p_b)

    x = torch.randn(4, 156)
    assert torch.equal(net_a(x), net_b(x))
