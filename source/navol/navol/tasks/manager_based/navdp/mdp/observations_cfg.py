from __future__ import annotations

import torch
from collections.abc import Callable
from dataclasses import MISSING
from typing import TYPE_CHECKING, Any
from isaaclab.utils import configclass

from isaaclab.managers import ObservationTermCfg
from .observations import RGBD_feature



@configclass
class RGBDFeatureCfg(ObservationTermCfg):
    """Configuration for the RGBD image feature observation."""

    class_type: type = RGBD_feature

    image_size: int = 224
    token_dim: int = 384
    memory_size: int = 8
