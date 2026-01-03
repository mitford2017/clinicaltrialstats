"""
Survival trial simulation module.
Implements similar functionality to simtrial's sim_pw_surv and related functions.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray

try:
    from .distributions import PiecewiseExponential
except ImportError:
    from distributions import PiecewiseExponential


@dataclass
class EnrollmentPattern:
    """
    Piecewise constant enrollment pattern.

    Attributes:
        durations: Duration of each enrollment period (months)
        rates: Enrollment rate in each period (patients/month)
    """
    durations: list[float]
    rates: list[float]

    def __post_init__(self):
        if len(self.durations) != len(self.rates):
            raise ValueError("durations and rates must have the same length")

    @classmethod
    def uniform(cls, duration: float, total_n: int) -> "EnrollmentPattern":
        """Create uniform enrollment over a given duration."""
        rate = total_n / duration
        return cls(durations=[duration], rates=[rate])

    @classmethod
    def ramp_up(
        cls,
        ramp_duration: float,
        steady_duration: float,
        total_n: int,
        ramp_fraction: float = 0.3
    ) -> "EnrollmentPattern":
        """
        Create enrollment with initial ramp-up period.

        Args:
            ramp_duration: Duration of ramp-up period
            steady_duration: Duration of steady-state enrollment
            total_n: Total number of patients to enroll
            ramp_fraction: Fraction of steady rate during ramp-up
        """
        # Solve for steady rate: ramp_rate * ramp_dur + steady_rate * steady_dur = total_n
        # ramp_rate = ramp_fraction * steady_rate
        total_duration = ramp_duration + steady_duration
        steady_rate = total_n / (ramp_fraction * ramp_duration + steady_duration)
        ramp_rate = ramp_fraction * steady_rate

        return cls(
            durations=[ramp_duration, steady_duration],
            rates=[ramp_rate, steady_rate]
        )

    def total_patients(self) -> float:
        """Calculate total expected enrollment."""
        return sum(d * r for d, r in zip(self.durations, self.rates))


@dataclass
class TrialArm:
    """
    Configuration for a trial arm.

    Attributes:
        name: Arm identifier
        ratio: Randomization ratio (e.g., 1 for 1:1)
        survival_dist: Survival time distribution
        dropout_dist: Dropout time distribution (optional)
    """
    name: str
    ratio: float
    survival_dist: PiecewiseExponential
    dropout_dist: Optional[PiecewiseExponential] = None


@dataclass
class TrialConfig:
    """
    Complete trial configuration.

    Attributes:
        enrollment: Enrollment pattern
        arms: List of trial arms
        total_n: Total sample size
        seed: Random seed for reproducibility
    """
    enrollment: EnrollmentPattern
    arms: list[TrialArm]
    total_n: int
    seed: Optional[int] = None


@dataclass
class SimulatedTrial:
    """
    Results from a single trial simulation.

    Attributes:
        data: DataFrame with columns [patient_id, arm, enroll_time, event_time,
              dropout_time, survival_time, event, calendar_time]
        config: The configuration used for simulation
    """
    data: pd.DataFrame
    config: TrialConfig

    def events_at_time(self, calendar_time: float) -> int:
        """Count events observed by a given calendar time."""
        observed = self.data[self.data["enroll_time"] <= calendar_time].copy()
        follow_up = calendar_time - observed["enroll_time"]
        events = (observed["event"] == 1) & (observed["survival_time"] <= follow_up)
        return int(events.sum())

    def cut_at_time(self, calendar_time: float) -> pd.DataFrame:
        """Cut data at a specified calendar time for analysis."""
        return cut_data_by_date(self.data, calendar_time)

    def cut_at_events(self, target_events: int) -> tuple[pd.DataFrame, float]:
        """Cut data when target number of events is reached."""
        return cut_data_by_event(self.data, target_events)


def simulate_enrollment(
    enrollment: EnrollmentPattern,
    n: int,
    rng: np.random.Generator | None = None
) -> NDArray[np.float64]:
    """
    Simulate enrollment times using piecewise Poisson process.

    Args:
        enrollment: Enrollment pattern configuration
        n: Number of patients to enroll
        rng: Random number generator

    Returns:
        Array of enrollment times
    """
    if rng is None:
        rng = np.random.default_rng()

    times = []
    current_time = 0.0

    for duration, rate in zip(enrollment.durations, enrollment.rates):
        if rate <= 0:
            current_time += duration
            continue

        # Expected number in this period
        expected = rate * duration

        # Generate inter-arrival times (exponential with rate = rate)
        while True:
            inter_arrival = rng.exponential(1 / rate)
            current_time += inter_arrival

            # Check if we've exceeded this period
            period_end = sum(enrollment.durations[:enrollment.durations.index(duration) + 1])
            if current_time > period_end:
                current_time = period_end
                break

            times.append(current_time)

            if len(times) >= n:
                break

        if len(times) >= n:
            break

    # If we didn't get enough patients, continue in last period
    if len(times) < n and enrollment.rates[-1] > 0:
        while len(times) < n:
            inter_arrival = rng.exponential(1 / enrollment.rates[-1])
            current_time += inter_arrival
            times.append(current_time)

    return np.array(times[:n])


def simulate_piecewise_exponential(
    dist: PiecewiseExponential,
    n: int,
    rng: np.random.Generator | None = None
) -> NDArray[np.float64]:
    """
    Simulate survival times from a piecewise exponential distribution.

    Args:
        dist: Piecewise exponential distribution
        n: Number of samples
        rng: Random number generator

    Returns:
        Array of survival times
    """
    return dist.sample(n, rng)


def simulate_trial(config: TrialConfig) -> SimulatedTrial:
    """
    Simulate a complete clinical trial.

    Args:
        config: Trial configuration

    Returns:
        SimulatedTrial with patient-level data
    """
    rng = np.random.default_rng(config.seed)

    # Calculate allocation probabilities
    total_ratio = sum(arm.ratio for arm in config.arms)
    probabilities = [arm.ratio / total_ratio for arm in config.arms]

    # Allocate patients to arms
    arm_assignments = rng.choice(
        len(config.arms),
        size=config.total_n,
        p=probabilities
    )

    # Simulate enrollment times
    enroll_times = simulate_enrollment(config.enrollment, config.total_n, rng)

    # Simulate survival and dropout for each patient
    records = []
    for i, (arm_idx, enroll_time) in enumerate(zip(arm_assignments, enroll_times)):
        arm = config.arms[arm_idx]

        # Simulate event time
        event_time = arm.survival_dist.sample(1, rng)[0]

        # Simulate dropout time
        if arm.dropout_dist is not None:
            dropout_time = arm.dropout_dist.sample(1, rng)[0]
        else:
            dropout_time = np.inf

        # Observed survival time and event indicator
        survival_time = min(event_time, dropout_time)
        event = 1 if event_time <= dropout_time else 0

        records.append({
            "patient_id": i + 1,
            "arm": arm.name,
            "enroll_time": enroll_time,
            "event_time": event_time,
            "dropout_time": dropout_time,
            "survival_time": survival_time,
            "event": event,
            "calendar_time": enroll_time + survival_time,
        })

    data = pd.DataFrame(records)
    return SimulatedTrial(data=data, config=config)


def cut_data_by_date(data: pd.DataFrame, cut_date: float) -> pd.DataFrame:
    """
    Cut trial data at a specified calendar date for analysis.

    Args:
        data: Trial data with columns [patient_id, arm, enroll_time, survival_time, event]
        cut_date: Calendar time at which to cut data

    Returns:
        DataFrame with time-at-risk (tte) and event indicator updated for cut date
    """
    result = data.copy()

    # Only include patients enrolled by cut date
    result = result[result["enroll_time"] <= cut_date].copy()

    # Calculate follow-up time at cut date
    follow_up = cut_date - result["enroll_time"]

    # Update time-to-event and censoring
    result["tte"] = np.minimum(result["survival_time"], follow_up)
    result["event_observed"] = (
        (result["event"] == 1) &
        (result["survival_time"] <= follow_up)
    ).astype(int)

    return result


def cut_data_by_event(
    data: pd.DataFrame,
    target_events: int
) -> tuple[pd.DataFrame, float]:
    """
    Cut trial data when a target number of events is reached.

    Args:
        data: Trial data with columns [patient_id, arm, enroll_time, survival_time, event]
        target_events: Number of events to trigger cut

    Returns:
        Tuple of (cut data, cut date)
    """
    # Get calendar times of all events
    events = data[data["event"] == 1].copy()
    event_times = (events["enroll_time"] + events["survival_time"]).sort_values()

    if len(event_times) < target_events:
        # Not enough events - use last event time
        cut_date = float(event_times.iloc[-1]) if len(event_times) > 0 else float(data["calendar_time"].max())
    else:
        cut_date = float(event_times.iloc[target_events - 1])

    return cut_data_by_date(data, cut_date), cut_date


def simulate_trials(
    config: TrialConfig,
    n_simulations: int,
    seed: Optional[int] = None
) -> list[SimulatedTrial]:
    """
    Run multiple trial simulations.

    Args:
        config: Trial configuration
        n_simulations: Number of simulations to run
        seed: Base random seed

    Returns:
        List of SimulatedTrial objects
    """
    trials = []
    for i in range(n_simulations):
        sim_config = TrialConfig(
            enrollment=config.enrollment,
            arms=config.arms,
            total_n=config.total_n,
            seed=(seed + i) if seed is not None else None
        )
        trials.append(simulate_trial(sim_config))

    return trials
