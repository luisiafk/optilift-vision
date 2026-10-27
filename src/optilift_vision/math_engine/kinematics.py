import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline
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

    #SCALE
    meters_per_pixel = 0.002  # Example scale factor, adjust as needed
    x *= meters_per_pixel
    y *= meters_per_pixel

    window_len = min(15, len(t) if len(t) % 2 == 1 else len(t) - 1)
    if window_len < 5:
        window_len = 5

    x_smooth = savgol_filter(x, window_length=window_len, polyorder=3)
    y_smooth = savgol_filter(y, window_length=window_len, polyorder=3)

    #Adjust the smooths cubics Splines
    #Exact interpolation s = 0: smooth s > 0 against the noise
    spline_x = UnivariateSpline(t, x_smooth, k=3, s=1000.0)
    spline_y = UnivariateSpline(t, y_smooth, k=3, s=1000.0)

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



