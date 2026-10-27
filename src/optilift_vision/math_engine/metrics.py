import numpy as np
from typing import List, Dict, Any
from .types import KinematicState

def find_sticky_point(states: List[KinematicState]) -> Dict[str, Any]:
    """
    Identify the sticky point:
    Occurs on the ascendent phase(vy > 0) where the vertical acceleration (ay)
    reachs its maximal local minimun after the first peak of accelaration
    """

    vy = np.array([s.vel_y for s in states])
    ay = np.array([s.acc_y for s in states])

    #Filter just for the concentric phase
    concentric_mask = vy > 0.5
    if not np.any(concentric_mask):
        return {"detected": False, "reason": "No concentric phase found"}

    #Finding the local minimun of vertical acceleration
    concentric_indices = np.where(concentric_mask)[0]
    ay_concentric = ay[concentric_indices]

    #Minimun index of acceleration
    min_acc_idx_rel = np.argmin(ay_concentric)
    target_idx = concentric_indices[min_acc_idx_rel]

    return{
        "detected": True,
        "frame_idx": target_idx,
        "timestamp": states[target_idx].timestamp,
        "vel_y": states[target_idx].vel_y,
        "acc_y": states[target_idx].acc_y,
        "position": (states[target_idx].pos_x, states[target_idx].pos_y)
    }
