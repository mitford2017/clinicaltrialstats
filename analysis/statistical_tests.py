"""
Statistical tests for survival analysis.
Implements weighted logrank and MaxCombo tests.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class LogrankResult:
    """Results from a logrank test."""
    z_statistic: float
    p_value: float
    observed_events: dict[str, int]
    expected_events: dict[str, float]
    hazard_ratio: float
    hr_lower_ci: float
    hr_upper_ci: float


@dataclass
class MaxComboResult:
    """Results from a MaxCombo test."""
    max_z_statistic: float
    p_value: float
    component_results: list[LogrankResult]
    winning_test: str


def counting_process(data: pd.DataFrame) -> pd.DataFrame:
    """
    Convert survival data to counting process format.

    Args:
        data: DataFrame with columns [arm, tte, event_observed]

    Returns:
        DataFrame in counting process format with at-risk and event counts
    """
    # Get unique event times
    event_times = sorted(data[data["event_observed"] == 1]["tte"].unique())

    records = []
    for t in event_times:
        for arm in data["arm"].unique():
            arm_data = data[data["arm"] == arm]

            # At risk: patients with tte >= t
            at_risk = (arm_data["tte"] >= t).sum()

            # Events at this time
            events = ((arm_data["tte"] == t) & (arm_data["event_observed"] == 1)).sum()

            records.append({
                "time": t,
                "arm": arm,
                "at_risk": at_risk,
                "events": events,
            })

    return pd.DataFrame(records)


def weighted_logrank(
    data: pd.DataFrame,
    weight_function: Optional[Callable[[float, float], float]] = None,
    rho: float = 0.0,
    gamma: float = 0.0,
) -> LogrankResult:
    """
    Weighted logrank test (Fleming-Harrington family).

    The weight function is S(t)^rho * (1-S(t))^gamma where S(t) is the
    pooled Kaplan-Meier estimate.

    Args:
        data: DataFrame with columns [arm, tte, event_observed]
        weight_function: Custom weight function (time, survival) -> weight
        rho: Parameter for Fleming-Harrington weight (default 0 = standard logrank)
        gamma: Parameter for Fleming-Harrington weight (default 0)

    Returns:
        LogrankResult with test statistics and hazard ratio estimate
    """
    # Get counting process data
    cp = counting_process(data)

    # Identify arms (assume first is control, second is treatment)
    arms = data["arm"].unique()
    if len(arms) != 2:
        raise ValueError("Weighted logrank requires exactly 2 arms")

    control_arm, treatment_arm = arms[0], arms[1]

    # Pivot to get at_risk and events by arm at each time
    times = sorted(cp["time"].unique())

    # Calculate pooled Kaplan-Meier for weights
    survival = 1.0
    observed_minus_expected = 0.0
    variance = 0.0

    observed = {control_arm: 0, treatment_arm: 0}
    expected = {control_arm: 0.0, treatment_arm: 0.0}

    for t in times:
        time_data = cp[cp["time"] == t]

        # Get at-risk and events for each arm
        n_control = time_data[time_data["arm"] == control_arm]["at_risk"].values
        n_treatment = time_data[time_data["arm"] == treatment_arm]["at_risk"].values
        d_control = time_data[time_data["arm"] == control_arm]["events"].values
        d_treatment = time_data[time_data["arm"] == treatment_arm]["events"].values

        n_control = n_control[0] if len(n_control) > 0 else 0
        n_treatment = n_treatment[0] if len(n_treatment) > 0 else 0
        d_control = d_control[0] if len(d_control) > 0 else 0
        d_treatment = d_treatment[0] if len(d_treatment) > 0 else 0

        n_total = n_control + n_treatment
        d_total = d_control + d_treatment

        if n_total == 0:
            continue

        # Calculate weight
        if weight_function is not None:
            weight = weight_function(t, survival)
        else:
            # Fleming-Harrington weight
            weight = (survival ** rho) * ((1 - survival) ** gamma)

        # Expected events under null
        e_treatment = d_total * n_treatment / n_total if n_total > 0 else 0

        # Accumulate observed and expected
        observed[control_arm] += d_control
        observed[treatment_arm] += d_treatment
        expected[control_arm] += d_total * n_control / n_total if n_total > 0 else 0
        expected[treatment_arm] += e_treatment

        # Weighted observed - expected
        observed_minus_expected += weight * (d_treatment - e_treatment)

        # Variance contribution
        if n_total > 1:
            var_contrib = d_total * n_treatment * n_control * (n_total - d_total) / (n_total ** 2 * (n_total - 1))
            variance += weight ** 2 * var_contrib

        # Update survival estimate
        survival *= (1 - d_total / n_total) if n_total > 0 else survival

    # Calculate Z-statistic
    if variance > 0:
        z_stat = observed_minus_expected / np.sqrt(variance)
    else:
        z_stat = 0.0

    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))  # Two-sided

    # Estimate hazard ratio using Cox-like approximation
    if expected[treatment_arm] > 0:
        hr = observed[treatment_arm] / expected[treatment_arm]
    else:
        hr = 1.0

    # Confidence interval for HR
    if observed[treatment_arm] > 0:
        se_log_hr = np.sqrt(1 / observed[treatment_arm] + 1 / observed[control_arm]) if observed[control_arm] > 0 else 1.0
        hr_lower = hr * np.exp(-1.96 * se_log_hr)
        hr_upper = hr * np.exp(1.96 * se_log_hr)
    else:
        hr_lower, hr_upper = 0.0, np.inf

    return LogrankResult(
        z_statistic=z_stat,
        p_value=p_value,
        observed_events=observed,
        expected_events=expected,
        hazard_ratio=hr,
        hr_lower_ci=hr_lower,
        hr_upper_ci=hr_upper,
    )


def fleming_harrington_weight(rho: float, gamma: float) -> Callable[[float, float], float]:
    """
    Create a Fleming-Harrington weight function.

    Args:
        rho: Power of S(t) in weight
        gamma: Power of (1-S(t)) in weight

    Returns:
        Weight function (time, survival) -> weight
    """
    def weight(t: float, survival: float) -> float:
        return (survival ** rho) * ((1 - survival) ** gamma)
    return weight


def maxcombo_test(
    data: pd.DataFrame,
    weight_specs: Optional[list[tuple[float, float]]] = None,
) -> MaxComboResult:
    """
    MaxCombo test combining multiple weighted logrank tests.

    The MaxCombo test takes the maximum Z-statistic across multiple
    Fleming-Harrington weighted logrank tests and computes a combined p-value.

    Args:
        data: DataFrame with columns [arm, tte, event_observed]
        weight_specs: List of (rho, gamma) tuples for FH weights.
                     Default: [(0,0), (0,1), (1,0), (1,1)]

    Returns:
        MaxComboResult with test statistics
    """
    if weight_specs is None:
        # Default: standard logrank + early/late sensitivity tests
        weight_specs = [
            (0, 0),   # Standard logrank
            (0, 1),   # Late effects (modestly-weighted)
            (1, 0),   # Early effects
            (1, 1),   # Crossing hazards
        ]

    # Run each weighted logrank test
    results = []
    test_names = []

    for rho, gamma in weight_specs:
        result = weighted_logrank(data, rho=rho, gamma=gamma)
        results.append(result)
        test_names.append(f"FH({rho},{gamma})")

    # Find maximum Z-statistic
    z_stats = [abs(r.z_statistic) for r in results]
    max_z = max(z_stats)
    max_idx = z_stats.index(max_z)

    # For p-value, we use a conservative Bonferroni-like correction
    # In practice, more sophisticated methods account for correlation
    min_p = min(r.p_value for r in results)
    adjusted_p = min(1.0, min_p * len(results))  # Bonferroni

    return MaxComboResult(
        max_z_statistic=max_z if results[max_idx].z_statistic >= 0 else -max_z,
        p_value=adjusted_p,
        component_results=results,
        winning_test=test_names[max_idx],
    )


def rmst_difference(
    data: pd.DataFrame,
    tau: float,
) -> dict:
    """
    Calculate restricted mean survival time (RMST) difference.

    RMST is the area under the survival curve up to time tau.

    Args:
        data: DataFrame with columns [arm, tte, event_observed]
        tau: Restriction time

    Returns:
        Dictionary with RMST values and difference
    """
    from lifelines import KaplanMeierFitter

    arms = data["arm"].unique()
    if len(arms) != 2:
        raise ValueError("RMST requires exactly 2 arms")

    control_arm, treatment_arm = arms[0], arms[1]

    results = {}
    for arm in [control_arm, treatment_arm]:
        arm_data = data[data["arm"] == arm]

        kmf = KaplanMeierFitter()
        kmf.fit(arm_data["tte"], arm_data["event_observed"])

        # Calculate RMST as area under survival curve
        timeline = np.linspace(0, tau, 1000)
        survival = kmf.survival_function_at_times(timeline).values

        # Trapezoidal integration
        rmst = np.trapz(survival, timeline)
        results[arm] = rmst

    difference = results[treatment_arm] - results[control_arm]

    return {
        "control_rmst": results[control_arm],
        "treatment_rmst": results[treatment_arm],
        "difference": difference,
        "tau": tau,
    }
