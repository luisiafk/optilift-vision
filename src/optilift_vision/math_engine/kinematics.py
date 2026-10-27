import numpy as np
from scipy.interpolate import UnivariateSpiline
from typing import List, Tuple
from .types import Keypoint2d, KinematicState

def procces_barbell_kinematics(
    timestamps: List[float],
    positions: List[Keypoint2d],
    smoothing_factor: float = 0
) -> List[KinematicState]:
    """
    Recieve a bunch of coordinates (x,y) of the barbell
    and apply cubic splines to calculate position, velocity and acceleration
    """
    t = np.array(timestamps)
    x = np.array([p.x for p in positions])
    y = np.array([p.y for p in positions])


    #Adjust the smooths cubics Splines
    #Exact interpolation s = 0: smooth s > 0 against the noise
    spline_x = UnivariateSpiline(t, x, k=3, s=smoothing_factor)
    spline_y = UnivariateSpiline(t, y, k=3, s=smoothing_factor)

    #Exact splines numerics derivates
    dspline_x = spline_x.derivative(1)
    dspline_y = spline_y.derivative(1)

    d2spline_x = spline_x.derivative(2)
    d2spline_y = spline_y.derivative(2)

    #Evaluate in the original time vector
    vx = dspline_x(t)
    vy = dspline_y(t)
    ax = d2spline_x(t)
    ay = d2spline_y(t)


    states = []
    for i in range(len(t)):
        v_mag = float(np.sqrt(vx[i]**2 + vy[i]**2))
        a_mag = float(np.sqrt(ax[i]**2 + ay[i]**2))
        states.append(
            KinematicState(
                timestamp=t[i],
                pos_x=float(spline_x(t[i])),
                pos_y=float(spline_y(t[i])),
                vel_x=float(vx[i]),
                vel_y=float(vy[i]),
                acc_x=float(ax[i]),
                acc_y=float(ax[i]),
                v_mag=v_mag,
                a_mag=a_mag
            )
        )
    return states



