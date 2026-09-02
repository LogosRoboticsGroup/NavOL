"""Lightweight canonical training parameters and behavior helpers.

This module deliberately avoids Isaac Lab imports so configuration and behavior can be
validated on machines that do not have Isaac Sim installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


CriticCollisionReduction = Literal["mean", "sum"]


@dataclass(frozen=True)
class CanonicalTrainingConfig:
    """Parameters used by the canonical NavOL distillation recipe."""

    rollout_steps: int = 128
    iterations: int = 1000
    learning_epochs: int = 10
    mini_batches: int = 16
    environments_per_process: int = 32
    distributed_processes: int = 8
    learning_rate: float = 1e-5
    lambda_critic: float = 0.1
    camera_height_range: tuple[float, float] = (0.25, 1.25)
    camera_pitch_range: tuple[float, float] = (-30.0, 0.0)
    policy_probability: float = 0.8
    use_mpc: bool = True
    navmesh_radius: float = 0.25
    local_search_radius: float = 0.1
    min_keypoints: int = 5
    critic_collision_reduction: CriticCollisionReduction = "mean"
    critic_safe_distance: float = 0.5
    critic_progress_alpha: float = 0.1

    @property
    def batch_size(self) -> int:
        """Global mini-batch size for the canonical distributed launch."""

        rollout_size = self.rollout_steps * self.environments_per_process * self.distributed_processes
        if rollout_size % self.mini_batches:
            raise ValueError("canonical rollout size must be divisible by mini_batches")
        return rollout_size // self.mini_batches


CANONICAL_TRAINING = CanonicalTrainingConfig()


def select_policy_action(
    action_type: str,
    policy_probability: float,
    random_value: float,
) -> bool:
    """Return whether an environment should execute the policy action.

    ``random_value`` is supplied by the caller so selection remains reproducible and
    can use the simulator's existing random-number source.
    """

    if not 0.0 <= policy_probability <= 1.0:
        raise ValueError("policy_probability must be in [0, 1]")
    if not 0.0 <= random_value <= 1.0:
        raise ValueError("random_value must be in [0, 1]")
    if action_type == "rand":
        return random_value < policy_probability
    if action_type == "policy":
        return True
    if action_type == "command":
        return False
    raise ValueError(f"unknown action_type: {action_type!r}")


def _reduce_last(values: Any, reduction: CriticCollisionReduction) -> Any:
    """Reduce the final dimension for either Torch tensors or NumPy arrays."""

    operation = getattr(values, reduction)
    try:
        return operation(dim=-1)
    except TypeError:
        return operation(axis=-1)


def compute_critic_targets(
    distances: Any,
    *,
    safe_distance: float = 0.5,
    progress_alpha: float = 0.1,
    collision_reduction: CriticCollisionReduction = "mean",
) -> Any:
    """Compute NavOL critic targets for a batch of path-clearance sequences.

    ``mean`` preserves the released checkpoint implementation. ``sum`` implements
    the unreduced collision-count term printed in the paper and may require retraining.
    """

    if collision_reduction not in ("mean", "sum"):
        raise ValueError("collision_reduction must be 'mean' or 'sum'")
    if distances.shape[-1] < 2:
        raise ValueError("distances must contain at least two path samples")

    collision_mask = distances < safe_distance
    if hasattr(collision_mask, "to"):
        collision_values = collision_mask.to(dtype=distances.dtype)
    else:
        collision_values = collision_mask.astype(distances.dtype, copy=False)

    collision_term = -_reduce_last(collision_values, collision_reduction)
    progress = distances[..., 1:] - distances[..., :-1]
    progress_term = progress_alpha * _reduce_last(progress, "sum")
    return collision_term + progress_term
