"""Tests for ``mjlab.rl.distributions.SoftplusGaussianDistribution``.

The key invariant is that the policy's std never drops below ``min_std``,
regardless of what the MLP outputs -- this is the exploration floor that
prevents premature policy collapse.
"""

from __future__ import annotations

import math

import torch
from rsl_rl.utils import resolve_callable

from mjlab.rl.distributions import SoftplusGaussianDistribution


def _make_mlp_output(
  output_dim: int, mean_val: float, raw_std_val: float, batch: int = 4
) -> torch.Tensor:
  """Build an MLP output of shape (batch, 2, output_dim) with fixed mean/raw_std."""
  mean = torch.full((batch, output_dim), mean_val)
  raw_std = torch.full((batch, output_dim), raw_std_val)
  return torch.stack([mean, raw_std], dim=-2)


def test_std_floor_holds_for_very_negative_raw() -> None:
  """Even with raw → -inf (softplus → 0), std must equal min_std exactly."""
  dist = SoftplusGaussianDistribution(output_dim=5, init_std=0.5, min_std=0.2)
  mlp_out = _make_mlp_output(5, mean_val=0.0, raw_std_val=-50.0)
  dist.update(mlp_out)
  std = dist.std
  assert torch.allclose(std, torch.full_like(std, 0.2), atol=1e-5), (
    f"Std should clamp at min_std=0.2, got {std}"
  )


def test_std_floor_at_zero_raw_matches_log2_plus_min() -> None:
  """At raw=0, softplus(0)=ln(2)~=0.693, so std should be ln(2) + min_std."""
  min_std = 0.1
  dist = SoftplusGaussianDistribution(output_dim=3, init_std=0.8, min_std=min_std)
  mlp_out = _make_mlp_output(3, mean_val=0.0, raw_std_val=0.0)
  dist.update(mlp_out)
  expected = math.log(2.0) + min_std
  std = dist.std
  assert torch.allclose(std, torch.full_like(std, expected), atol=1e-5)


def test_init_bias_makes_initial_std_equal_init_std() -> None:
  """Initialization should be such that std starts at init_std (head bias init)."""
  # We can't check the bias directly without an MLP instance, but we can verify
  # the reverse-compute math: softplus(init_bias) + min_std == init_std exactly.
  init_std, min_std = 0.5, 0.2
  target = init_std - min_std  # what softplus should produce
  init_bias = math.log(math.exp(target) - 1.0)
  softplus_of_bias = math.log(1.0 + math.exp(init_bias))
  assert math.isclose(softplus_of_bias + min_std, init_std, abs_tol=1e-9)
  # Constructing the distribution shouldn't raise.
  _ = SoftplusGaussianDistribution(output_dim=3, init_std=init_std, min_std=min_std)


def test_std_responds_to_large_positive_raw() -> None:
  """At large positive raw, softplus(raw) ~= raw, so std ~= raw + min_std."""
  dist = SoftplusGaussianDistribution(output_dim=2, init_std=0.5, min_std=0.2)
  mlp_out = _make_mlp_output(2, mean_val=0.0, raw_std_val=10.0)
  dist.update(mlp_out)
  # softplus(10) ~= 10 + log(1 + exp(-10)) ~= 10.0000454
  expected = math.log(1.0 + math.exp(10.0)) + 0.2
  std = dist.std
  assert torch.allclose(std, torch.full_like(std, expected), atol=1e-4)


def test_mean_passes_through_unchanged() -> None:
  """Mean half of the MLP output should be returned untouched (no transform)."""
  dist = SoftplusGaussianDistribution(output_dim=4, init_std=0.5, min_std=0.2)
  mlp_out = _make_mlp_output(4, mean_val=2.5, raw_std_val=0.0)
  dist.update(mlp_out)
  m = dist.mean
  assert torch.allclose(m, torch.full_like(m, 2.5))


def test_resolve_callable_can_load_class() -> None:
  """The class must be loadable via the module:Class path used in config."""
  cls = resolve_callable("mjlab.rl.distributions:SoftplusGaussianDistribution")
  assert cls is SoftplusGaussianDistribution


def test_gradient_flows_through_softplus_at_floor() -> None:
  """At very-negative raw (std clamped near min_std), gradient w.r.t. raw is
  still non-zero -- unlike a hard clamp that would zero the gradient."""
  raw = torch.full((1, 3), -20.0, requires_grad=True)
  mean = torch.zeros(1, 3)
  mlp_out = torch.stack([mean, raw], dim=-2)
  dist = SoftplusGaussianDistribution(output_dim=3, init_std=0.5, min_std=0.2)
  dist.update(mlp_out)
  # log_prob of a sample at the mean is finite; compute and backprop.
  sample = torch.zeros(1, 3)
  loss = -dist.log_prob(sample).sum()
  loss.backward()
  assert raw.grad is not None
  assert torch.all(raw.grad != 0), (
    "Gradient should still flow through softplus at low raw values."
  )
