"""Custom action distributions for use with rsl_rl.

mjlab plugs custom distributions in by passing the qualified class name
through the ``distribution_cfg`` of ``RslRlModelCfg``::

  distribution_cfg={
      "class_name": "mjlab.rl.distributions:SoftplusGaussianDistribution",
      "init_std": 0.5,
      "min_std": 0.2,
  }

rsl_rl's ``resolve_callable`` accepts the ``module.path:ClassName`` form, so
no fork is required to use classes defined here.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from rsl_rl.modules.distribution import (
  Distribution,
  HeteroscedasticGaussianDistribution,
)
from torch.distributions import Normal

__all__ = ["SoftplusGaussianDistribution"]


class SoftplusGaussianDistribution(HeteroscedasticGaussianDistribution):
  """State-dependent Gaussian whose std is ``softplus(raw) + min_std``.

  Equivalent to the Brax-style heteroscedastic Gaussian used in
  mujoco_playground / wuji-mjlab's reorient task. The MLP outputs a tensor
  of shape ``[..., 2, output_dim]`` -- the first slice is the mean, the
  second is the raw std parameter. We apply::

      std = softplus(raw) + min_std

  The ``+ min_std`` is a **hard floor** on the policy's exploration noise:
  the std never drops below ``min_std`` regardless of what the network
  outputs. This prevents premature exploration collapse (which is the
  classic failure mode where PPO converges to a near-deterministic policy
  before discovering high-reward behaviors) without needing to keep
  ``entropy_coef`` artificially high throughout training.

  Compared to rsl_rl's ``HeteroscedasticGaussianDistribution`` with a
  ``std_range`` clamp, the softplus + add construction has two advantages:
    1. The gradient w.r.t. ``raw`` is non-zero at the floor (softplus is
       smooth everywhere), so PPO still gets a learning signal at the
       boundary. A hard clamp kills the gradient there.
    2. The std approaches but never reaches ``min_std`` -- monotonically
       differentiable, no discontinuities for autograd to trip on.
  """

  def __init__(
    self, output_dim: int, init_std: float = 0.5, min_std: float = 0.01
  ) -> None:
    # Reverse-compute the bias init so that softplus(bias) + min_std == init_std
    # at the start of training. softplus(b) = log(1 + exp(b)), so we need
    # log(1 + exp(b)) = init_std - min_std, i.e. b = log(exp(target) - 1).
    target_softplus = max(init_std - min_std, 1e-6)
    init_bias = math.log(math.exp(target_softplus) - 1.0)
    # Skip the std_range / log_std handling of the parent -- our std is
    # always computed via softplus, so the parent's range fields are unused.
    # We do reuse the parent's MLP-head init mechanism by passing the
    # computed bias as init_std with std_type="scalar".
    super().__init__(output_dim, init_std=init_bias, std_type="scalar")
    self._min_std = float(min_std)

  def update(self, mlp_output: torch.Tensor) -> None:
    mean, raw = torch.unbind(mlp_output, dim=-2)
    std = F.softplus(raw) + self._min_std
    self._distribution = Normal(mean, std)


# Re-export the base class so ``isinstance`` checks work for either alias.
_ = Distribution  # silence "unused import" while keeping the symbol importable
