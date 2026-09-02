# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg, RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass

from . import records

@configclass
class PostStepRewardRecorderCfg(RecorderTermCfg):
    """Configuration for the step reward recorder term."""

    class_type: type[RecorderTerm] = records.PostStepRewardRecorder

@configclass
class RewardRecorderManagerCfg(RecorderManagerBaseCfg):
    """Recorder configurations for recording actions and states."""
    record_post_step_states = PostStepRewardRecorderCfg()