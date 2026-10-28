from dataclasses import dataclass, field
from typing import List, Tuple, Dict

@dataclass
class Keypoint2d:
    x: float
    y: float

@dataclass
class FramePose:
    frame_idx: int
    timestamp: float
    barbell: Keypoint2d
    hip: Keypoint2d
    knee: Keypoint2d
    ankle: Keypoint2d
    shoulder: Keypoint2d
    plates: List[Keypoint2d] = field(default_factory=list)

@dataclass
class KinematicState:
    timestamp: float
    pos_x: float
    pos_y: float
    vel_x: float
    vel_y: float
    acc_x: float
    acc_y: float
    v_mag: float
    a_mag: float

@dataclass
class DynamicState:
    timestamp: float
    energy: float
    power: float
    force: float
    impulse: float