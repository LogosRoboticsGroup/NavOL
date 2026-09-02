from __future__ import annotations

import torch
from collections.abc import Callable
from dataclasses import MISSING
from typing import TYPE_CHECKING, Any
from isaaclab.utils import configclass

from isaaclab.managers import ObservationTermCfg
from .observations import gs_image_feature



@configclass
class GSImageFeatureCfg(ObservationTermCfg):
    """Configuration for the GS image feature observation."""

    class_type: type = gs_image_feature

    data_dir: str = MISSING

    camera_pos: list = MISSING,
    camera_rot: list = MISSING,
    asset_offset_pos: list = MISSING,
    asset_offset_rot: list = MISSING,