"""Constant-velocity Kalman model for short gaps on the office floor."""

from dataclasses import dataclass
from math import hypot, sqrt


@dataclass(frozen=True, slots=True)
class FloorState:
    time_s: float
    position: tuple[float, float]
    velocity: tuple[float, float]
    # Symmetric covariance of (position, velocity), identical for x and y.
    position_var: float
    position_velocity_cov: float
    velocity_var: float

    def predict(self, time_s, acceleration_var):
        dt = time_s - self.time_s
        # Integrated white acceleration noise for the actual time between frames.
        return FloorState(
            time_s,
            tuple(p + v * dt for p, v in zip(self.position, self.velocity)),
            self.velocity,
            self.position_var
            + 2 * dt * self.position_velocity_cov
            + dt**2 * self.velocity_var
            + acceleration_var * dt**3 / 3,
            self.position_velocity_cov
            + dt * self.velocity_var
            + acceleration_var * dt**2 / 2,
            self.velocity_var + acceleration_var * dt,
        )

    def update(self, point, measurement_var):
        innovation_var = self.position_var + measurement_var
        position_gain = self.position_var / innovation_var
        velocity_gain = self.position_velocity_cov / innovation_var
        residual = tuple(
            observed - predicted for observed, predicted in zip(point, self.position)
        )
        return FloorState(
            self.time_s,
            tuple(p + position_gain * r for p, r in zip(self.position, residual)),
            tuple(v + velocity_gain * r for v, r in zip(self.velocity, residual)),
            position_gain * measurement_var,
            velocity_gain * measurement_var,
            self.velocity_var - velocity_gain * self.position_velocity_cov,
        )

    @property
    def velocity_reliability(self):
        """Continuous confidence from velocity relative to its standard error."""
        speed = hypot(*self.velocity)
        return speed / (speed + sqrt(self.velocity_var))


def fit_floor_state(points, measurement_var, acceleration_var, initial_velocity_var):
    """Fit distinct recent times with bounded support, keeping the last position."""
    if not points:
        return None
    recent = [points[-1]]
    for sample in reversed(points[:-1]):
        if points[-1][0] - sample[0] > 1.6 or recent[-1][0] - sample[0] > 0.8:
            break
        if recent[-1][0] - sample[0] >= 0.15 - 1e-9:
            recent.append(sample)
    if len(recent) < 3 or recent[0][0] - recent[-1][0] < 0.4 - 1e-9:
        return None

    time_s, x, y = recent.pop()
    state = FloorState(
        time_s, (x, y), (0.0, 0.0), measurement_var, 0.0, initial_velocity_var
    )
    for time_s, x, y in reversed(recent):
        state = state.predict(time_s, acceleration_var).update((x, y), measurement_var)
    return state
