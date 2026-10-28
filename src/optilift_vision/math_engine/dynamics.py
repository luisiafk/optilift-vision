import numpy as np
from typing import List
from .types import KinematicState, DynamicState

DEFAULT_BARBELL_MASS_KG = 0.025  # kg: barra + discos aprox. 25kg


def instantaneous_energy(state: KinematicState, mass_kg: float) -> float:
    """
    Energia cinetica instantanea del barbell:
        E(t) = 0.5 * m * |v(t)|^2
    """
    return float(0.5 * mass_kg * state.v_mag ** 2)


def instantaneous_power(
    state: KinematicState,
    mass_kg: float,
) -> float:
    """
    Potencia instantanea:
        P(t) = F . v = m * (a . v)
    Donde F es la fuerza resultante (Ley de Newton: F = m * a) y se proyecta
    sobre la direccion del movimiento (producto punto F . v).
    """
    return float(mass_kg * (state.acc_x * state.vel_x + state.acc_y * state.vel_y))


def impulse_force(
    prev_state: KinematicState,
    curr_state: KinematicState,
    mass_kg: float,
) -> float:
    """
    Fuerza promedio (y su impulso) durante el salto entre frames:
        F = m * |dv| / dt
        Impulso J = F * dt = m * |dv|
    """
    dt = curr_state.timestamp - prev_state.timestamp
    if dt <= 0:
        dt = 1.0  # fallback ante timestamps duplicados

    dv_mag = float(np.sqrt(
        (curr_state.vel_x - prev_state.vel_x) ** 2
        + (curr_state.vel_y - prev_state.vel_y) ** 2
    ))
    force = mass_kg * dv_mag / dt
    impulse = mass_kg * dv_mag
    return float(force)


def procces_dynamics(
    states: List[KinematicState],
    mass_kg: float = DEFAULT_BARBELL_MASS_KG,
) -> List[DynamicState]:
    """
    Recibe los estados kinematicos (pos, vel, acc) y calcula las metricas
    dinamicas derivadas: energia, potencia e impulso/accion por frame.
    """
    dynamics: List[DynamicState] = []

    for i, state in enumerate(states):
        energy = instantaneous_energy(state, mass_kg)

        if i == 0 or i >= len(states) - 1:
            power = 0.0
            force = 0.0
            impulse = 0.0
        else:
            power = instantaneous_power(state, mass_kg)
            force = impulse_force(states[i - 1], state, mass_kg)
            impulse = force * (state.timestamp - states[i - 1].timestamp)

        dynamics.append(
            DynamicState(
                timestamp=state.timestamp,
                energy=energy,
                power=power,
                force=force,
                impulse=impulse,
            )
        )

    return dynamics
