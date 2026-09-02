from __future__ import annotations

from collections.abc import Sequence

from isaaclab.managers.recorder_manager import RecorderTerm


class PostStepRewardRecorder(RecorderTerm):
    """Recorder term that records the reward of the environment at the end of each step."""

    def record_post_step(self):
        return self._env.reward_manager._episode_sums
