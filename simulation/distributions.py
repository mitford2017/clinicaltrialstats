"""
Piecewise exponential distribution for survival analysis.
Implements similar functionality to simtrial's rpwexp.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class PiecewiseExponential:
    """
    Piecewise exponential distribution for survival times.

    The distribution has different hazard rates in different time intervals.
    This is commonly used to model survival data where the hazard rate
    changes over time (e.g., delayed treatment effect).

    Attributes:
        durations: Duration of each interval (last interval extends to infinity)
        hazard_rates: Hazard rate in each interval
    """
    durations: list[float]
    hazard_rates: list[float]

    def __post_init__(self):
        if len(self.durations) != len(self.hazard_rates):
            raise ValueError("durations and hazard_rates must have the same length")
        if any(d < 0 for d in self.durations):
            raise ValueError("durations must be non-negative")
        if any(h < 0 for h in self.hazard_rates):
            raise ValueError("hazard_rates must be non-negative")

    @classmethod
    def from_median(cls, median_survival: float) -> "PiecewiseExponential":
        """Create exponential distribution with given median survival."""
        hazard_rate = np.log(2) / median_survival
        return cls(durations=[np.inf], hazard_rates=[hazard_rate])

    @classmethod
    def with_delayed_effect(
        cls,
        control_median: float,
        treatment_hr: float,
        delay_duration: float,
    ) -> tuple["PiecewiseExponential", "PiecewiseExponential"]:
        """
        Create control and treatment distributions with delayed treatment effect.

        Args:
            control_median: Median survival for control arm
            treatment_hr: Hazard ratio for treatment effect (after delay)
            delay_duration: Duration of delay before treatment effect begins

        Returns:
            Tuple of (control_distribution, treatment_distribution)
        """
        control_hazard = np.log(2) / control_median

        control = cls(
            durations=[np.inf],
            hazard_rates=[control_hazard]
        )

        # Treatment has same hazard as control during delay, then reduced hazard
        treatment = cls(
            durations=[delay_duration, np.inf],
            hazard_rates=[control_hazard, control_hazard * treatment_hr]
        )

        return control, treatment

    def sample(self, n: int, rng: np.random.Generator | None = None) -> NDArray[np.float64]:
        """
        Generate random samples from the piecewise exponential distribution.

        Uses inverse transform sampling for each piece.

        Args:
            n: Number of samples to generate
            rng: Random number generator (default: numpy default)

        Returns:
            Array of survival times
        """
        if rng is None:
            rng = np.random.default_rng()

        u = rng.uniform(size=n)
        times = np.zeros(n)

        cumulative_time = 0.0
        cumulative_survival = 1.0

        for i, (duration, hazard) in enumerate(zip(self.durations, self.hazard_rates)):
            if i == len(self.durations) - 1:
                # Last interval extends to infinity
                remaining_duration = np.inf
            else:
                remaining_duration = duration

            if hazard == 0:
                # No events in this interval
                cumulative_time += remaining_duration
                continue

            # Survival at end of this interval
            if remaining_duration < np.inf:
                interval_survival = np.exp(-hazard * remaining_duration)
                next_survival = cumulative_survival * interval_survival
            else:
                next_survival = 0.0

            # Find samples that fall in this interval
            in_interval = (u > next_survival) & (u <= cumulative_survival)

            if np.any(in_interval):
                # Inverse transform for this interval
                # S(t) = S(t_start) * exp(-hazard * (t - t_start))
                # u = S(t_start) * exp(-hazard * (t - t_start))
                # t = t_start - log(u / S(t_start)) / hazard
                times[in_interval] = cumulative_time - np.log(u[in_interval] / cumulative_survival) / hazard

            cumulative_time += remaining_duration
            cumulative_survival = next_survival

        return times

    def survival_function(self, t: float | NDArray[np.float64]) -> float | NDArray[np.float64]:
        """
        Calculate survival probability at time t.

        Args:
            t: Time point(s) to evaluate

        Returns:
            Survival probability S(t)
        """
        t = np.asarray(t)
        scalar_input = t.ndim == 0
        t = np.atleast_1d(t)

        survival = np.ones_like(t, dtype=float)

        cumulative_time = 0.0
        for duration, hazard in zip(self.durations, self.hazard_rates):
            # Time spent in this interval
            time_in_interval = np.clip(t - cumulative_time, 0, duration)

            # Multiply by interval survival
            survival *= np.exp(-hazard * time_in_interval)

            cumulative_time += duration
            if cumulative_time >= np.max(t):
                break

        if scalar_input:
            return float(survival[0])
        return survival

    def hazard_function(self, t: float | NDArray[np.float64]) -> float | NDArray[np.float64]:
        """
        Calculate hazard rate at time t.

        Args:
            t: Time point(s) to evaluate

        Returns:
            Hazard rate h(t)
        """
        t = np.asarray(t)
        scalar_input = t.ndim == 0
        t = np.atleast_1d(t)

        hazard = np.zeros_like(t, dtype=float)

        cumulative_time = 0.0
        for duration, rate in zip(self.durations, self.hazard_rates):
            in_interval = (t >= cumulative_time) & (t < cumulative_time + duration)
            hazard[in_interval] = rate
            cumulative_time += duration

        if scalar_input:
            return float(hazard[0])
        return hazard

    def median(self) -> float:
        """Calculate the median survival time."""
        # Binary search for time where S(t) = 0.5
        low, high = 0.0, 1000.0
        while self.survival_function(high) > 0.5:
            high *= 2
            if high > 1e6:
                return np.inf

        for _ in range(100):
            mid = (low + high) / 2
            if self.survival_function(mid) > 0.5:
                low = mid
            else:
                high = mid

        return (low + high) / 2

    def mean(self, max_time: float = 1000.0) -> float:
        """
        Calculate mean survival time (restricted to max_time).

        Uses numerical integration of the survival function.
        """
        from scipy import integrate
        result, _ = integrate.quad(self.survival_function, 0, max_time)
        return result
