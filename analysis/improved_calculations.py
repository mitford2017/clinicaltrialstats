"""
Improved statistical calculations for clinical trial analysis.

Based on:
- Schoenfeld (1981) for event requirements
- Lachin-Foulkes (1986) for sample size with censoring
- gsDesign/simtrial R packages from Merck

Key improvements:
1. Trial-specific event calculations based on sample size and expected event rate
2. Proper critical HR formula
3. Correct probability of success calculation
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from dataclasses import dataclass
from typing import Optional, Tuple
from datetime import date, timedelta


@dataclass
class TrialDesign:
    """Statistical design parameters for a clinical trial."""
    sample_size: int
    control_median: float  # months
    design_hr: float
    power: float
    alpha: float  # one-sided
    allocation_ratio: float  # experimental:control
    enroll_duration: float  # months
    min_followup: float  # months after last patient enrolled

    @property
    def total_duration(self) -> float:
        """Total trial duration in months."""
        return self.enroll_duration + self.min_followup


def schoenfeld_events(
    hr: float,
    power: float,
    alpha: float,
    allocation_ratio: float = 1.0,
) -> int:
    """
    Calculate required events using Schoenfeld (1981) formula.

    d = ((r + 1) * (z_alpha + z_beta) / (sqrt(r) * log(HR)))^2

    Parameters
    ----------
    hr : float
        Design hazard ratio (< 1 for benefit)
    power : float
        Statistical power (e.g., 0.90)
    alpha : float
        One-sided significance level (e.g., 0.025)
    allocation_ratio : float
        Experimental:control allocation ratio

    Returns
    -------
    int : Required number of events
    """
    r = allocation_ratio
    z_alpha = norm.ppf(1 - alpha)
    z_beta = norm.ppf(power)

    log_hr = np.log(hr)
    if log_hr == 0:
        return 9999  # No effect, infinite events needed

    d = ((r + 1) * (z_alpha + z_beta) / (np.sqrt(r) * log_hr)) ** 2
    return int(np.ceil(d))


def expected_event_probability(
    control_median: float,
    hr: float,
    avg_followup: float,
    allocation_ratio: float = 1.0,
) -> float:
    """
    Calculate expected proportion of patients with events.

    Uses exponential model with weighted average hazard rate.

    Parameters
    ----------
    control_median : float
        Control arm median survival (months)
    hr : float
        Hazard ratio
    avg_followup : float
        Average follow-up time (months)
    allocation_ratio : float
        Experimental:control ratio

    Returns
    -------
    float : Expected event probability (0 to 1)
    """
    r = allocation_ratio

    # Hazard rates
    ctrl_rate = np.log(2) / control_median
    exp_rate = ctrl_rate * hr

    # Weighted average rate
    avg_rate = (ctrl_rate + exp_rate * r) / (1 + r)

    # Event probability (1 - survival function)
    p_event = 1 - np.exp(-avg_rate * avg_followup)

    return p_event


def estimate_achievable_events(
    sample_size: int,
    control_median: float,
    hr: float,
    enroll_duration: float,
    min_followup: float,
    allocation_ratio: float = 1.0,
) -> Tuple[int, float]:
    """
    Estimate maximum achievable events given trial parameters.

    Accounts for staggered enrollment and minimum follow-up.

    Returns
    -------
    Tuple of (expected_events, event_probability)
    """
    # Average follow-up = min_followup + (enroll_duration / 2)
    # (patients enrolled uniformly, so average enrollment time = enroll_duration/2)
    avg_followup = min_followup + enroll_duration / 2

    p_event = expected_event_probability(
        control_median, hr, avg_followup, allocation_ratio
    )

    expected_events = int(sample_size * p_event)

    return expected_events, p_event


def calculate_trial_events(
    sample_size: int,
    control_median: float,
    design_hr: float,
    power: float = 0.90,
    alpha: float = 0.025,
    enroll_duration: float = 24.0,
    min_followup: float = 12.0,
    allocation_ratio: float = 1.0,
    target_event_fraction: float = 0.66,  # For interim
) -> dict:
    """
    Calculate trial-specific event targets.

    Takes the MINIMUM of:
    1. Events required by Schoenfeld formula
    2. Events achievable given sample size and duration

    This prevents unrealistic event targets for smaller trials.
    """
    # Schoenfeld requirement
    schoenfeld_req = schoenfeld_events(design_hr, power, alpha, allocation_ratio)

    # Achievable events
    achievable, event_prob = estimate_achievable_events(
        sample_size, control_median, design_hr,
        enroll_duration, min_followup, allocation_ratio
    )

    # Use minimum (can't get more events than achievable)
    # But also ensure we have at least reasonable power
    final_events = min(schoenfeld_req, int(achievable * 0.95))  # 95% of achievable

    # If achievable < schoenfeld, trial is underpowered
    is_underpowered = achievable < schoenfeld_req

    # Interim at target fraction
    interim_events = int(final_events * target_event_fraction)

    return {
        'schoenfeld_required': schoenfeld_req,
        'achievable_events': achievable,
        'event_probability': event_prob,
        'final_events': final_events,
        'interim_events': interim_events,
        'is_underpowered': is_underpowered,
        'power_shortfall': (schoenfeld_req - achievable) if is_underpowered else 0,
    }


def critical_hr(
    events: int,
    alpha: float,
    allocation_ratio: float = 1.0,
) -> float:
    """
    Calculate critical hazard ratio threshold for significance.

    HR_crit = exp(-z_alpha * (r + 1) / sqrt(r * d))

    This is the HR at which the trial would exactly reject H0.
    Observed HR must be <= critical HR for significance.
    """
    r = allocation_ratio
    z = norm.ppf(1 - alpha)

    hr_crit = np.exp(-z * (r + 1) / np.sqrt(r * events))
    return hr_crit


def probability_of_success(
    true_hr: float,
    events: int,
    alpha: float,
    allocation_ratio: float = 1.0,
) -> float:
    """
    Calculate probability of trial success given true HR.

    Uses the conditional power formula:
    P(Success) = P(Z > z_crit | true_HR)

    Where Z ~ N(E[Z], 1) under true HR, and:
    E[Z] = -log(HR) * sqrt(d * r) / (r + 1)

    Parameters
    ----------
    true_hr : float
        Assumed true hazard ratio
    events : int
        Number of events at analysis
    alpha : float
        One-sided significance level
    allocation_ratio : float
        Experimental:control ratio

    Returns
    -------
    float : Probability of success (0 to 1)
    """
    if events <= 0:
        return 0.0

    r = allocation_ratio
    z_crit = norm.ppf(1 - alpha)

    # Expected Z under true HR
    # Note: for HR < 1 (benefit), log(HR) < 0, so E[Z] > 0
    expected_z = -np.log(true_hr) * np.sqrt(events * r) / (r + 1)

    # P(Z > z_crit) = P((Z - E[Z]) > z_crit - E[Z])
    # = 1 - Phi(z_crit - E[Z])
    prob = 1 - norm.cdf(z_crit - expected_z)

    return prob


def time_to_events(
    target_events: int,
    sample_size: int,
    enroll_duration: float,
    control_median: float,
    hr: float,
    allocation_ratio: float = 1.0,
) -> float:
    """
    Solve for calendar time to reach target events.

    Uses root-finding to solve:
    expected_events(t) = target_events
    """
    r = allocation_ratio
    ctrl_rate = np.log(2) / control_median
    exp_rate = ctrl_rate * hr
    avg_rate = (ctrl_rate + exp_rate * r) / (1 + r)

    def expected_events_at_time(t):
        # Patients enrolled by time t
        if t <= 0:
            return 0

        enrolled = min(t * (sample_size / enroll_duration), sample_size)

        # Average follow-up for enrolled patients
        if t <= enroll_duration:
            # Still enrolling - average follow-up is t/2
            avg_fu = t / 2
        else:
            # Enrollment complete
            # Min follow-up = t - enroll_duration
            # Max follow-up = t
            # Avg = (min + max) / 2 = t - enroll_duration/2
            avg_fu = t - enroll_duration / 2

        if avg_fu <= 0:
            return 0

        p_event = 1 - np.exp(-avg_rate * avg_fu)
        return enrolled * p_event

    def objective(t):
        return expected_events_at_time(t) - target_events

    # Find time when we hit target
    try:
        # Search between 1 month and 200 months
        t_solution = brentq(objective, 1, 200)
        return t_solution
    except ValueError:
        # Can't reach target events
        return float('inf')


@dataclass
class TrialAnalysis:
    """Complete analysis results for a clinical trial."""
    # Trial info
    sample_size: int
    control_median: float
    design_hr: float

    # Event calculations
    schoenfeld_events: int
    achievable_events: int
    final_events: int
    interim_events: int
    event_probability: float
    is_underpowered: bool

    # Critical thresholds
    interim_critical_hr: float
    final_critical_hr: float

    # Probability of success under different scenarios
    pos_base_case: float  # At design HR
    pos_bull_case: float  # 15% better than design
    pos_bear_case: float  # 20% worse than design

    # Timeline
    interim_time_months: float
    final_time_months: float


def analyze_trial(
    sample_size: int,
    control_median: float,
    design_hr: float,
    power: float = 0.90,
    alpha: float = 0.025,
    enroll_duration: Optional[float] = None,
    min_followup: float = 12.0,
    allocation_ratio: float = 1.0,
    interim_fraction: float = 0.66,
    interim_alpha: float = 0.01,
    final_alpha: float = 0.024,
) -> TrialAnalysis:
    """
    Comprehensive trial analysis with proper statistical calculations.

    Parameters
    ----------
    sample_size : int
        Total enrollment
    control_median : float
        Control arm median survival (months)
    design_hr : float
        Design hazard ratio assumption
    power : float
        Target power (default 0.90)
    alpha : float
        Overall one-sided alpha (default 0.025)
    enroll_duration : float, optional
        Enrollment duration in months (estimated if not provided)
    min_followup : float
        Minimum follow-up after last enrollment (default 12)
    allocation_ratio : float
        Experimental:control ratio (default 1.0)
    interim_fraction : float
        Information fraction at interim (default 0.66)
    interim_alpha : float
        Alpha spent at interim (default 0.01)
    final_alpha : float
        Alpha spent at final (default 0.024)
    """
    # Estimate enrollment duration if not provided
    if enroll_duration is None:
        # Assume 30-50 patients/month for Phase 3 oncology
        enroll_rate = 40
        enroll_duration = sample_size / enroll_rate

    # Calculate trial-specific events
    event_calc = calculate_trial_events(
        sample_size=sample_size,
        control_median=control_median,
        design_hr=design_hr,
        power=power,
        alpha=alpha,
        enroll_duration=enroll_duration,
        min_followup=min_followup,
        allocation_ratio=allocation_ratio,
        target_event_fraction=interim_fraction,
    )

    final_events = event_calc['final_events']
    interim_events = event_calc['interim_events']

    # Critical HRs
    interim_hr_crit = critical_hr(interim_events, interim_alpha, allocation_ratio)
    final_hr_crit = critical_hr(final_events, final_alpha, allocation_ratio)

    # Probability of success under different scenarios
    pos_base = probability_of_success(design_hr, final_events, final_alpha, allocation_ratio)

    bull_hr = design_hr * 0.85  # 15% better
    pos_bull = probability_of_success(bull_hr, final_events, final_alpha, allocation_ratio)

    bear_hr = min(design_hr * 1.20, 0.98)  # 20% worse, cap at 0.98
    pos_bear = probability_of_success(bear_hr, final_events, final_alpha, allocation_ratio)

    # Timeline estimation
    interim_time = time_to_events(
        interim_events, sample_size, enroll_duration,
        control_median, design_hr, allocation_ratio
    )
    final_time = time_to_events(
        final_events, sample_size, enroll_duration,
        control_median, design_hr, allocation_ratio
    )

    return TrialAnalysis(
        sample_size=sample_size,
        control_median=control_median,
        design_hr=design_hr,
        schoenfeld_events=event_calc['schoenfeld_required'],
        achievable_events=event_calc['achievable_events'],
        final_events=final_events,
        interim_events=interim_events,
        event_probability=event_calc['event_probability'],
        is_underpowered=event_calc['is_underpowered'],
        interim_critical_hr=interim_hr_crit,
        final_critical_hr=final_hr_crit,
        pos_base_case=pos_base,
        pos_bull_case=pos_bull,
        pos_bear_case=pos_bear,
        interim_time_months=interim_time,
        final_time_months=final_time,
    )


# Test function
if __name__ == "__main__":
    print("=" * 60)
    print("Testing improved calculations")
    print("=" * 60)

    # Test 1: XALUTE-like (smaller trial)
    print("\n--- XALUTE-like (675 patients, mCRPC post-taxane) ---")
    xalute = analyze_trial(
        sample_size=675,
        control_median=12.0,  # mCRPC post-taxane
        design_hr=0.75,
        enroll_duration=24,
        min_followup=18,
    )
    print(f"Schoenfeld required: {xalute.schoenfeld_events}")
    print(f"Achievable events: {xalute.achievable_events}")
    print(f"Final events target: {xalute.final_events}")
    print(f"Event probability: {xalute.event_probability:.1%}")
    print(f"Is underpowered: {xalute.is_underpowered}")
    print(f"Critical HR (final): {xalute.final_critical_hr:.3f}")
    print(f"P(Success) base case: {xalute.pos_base_case:.1%}")
    print(f"P(Success) bull case: {xalute.pos_bull_case:.1%}")
    print(f"P(Success) bear case: {xalute.pos_bear_case:.1%}")
    print(f"Time to final: {xalute.final_time_months:.1f} months")

    # Test 2: IDeate-like (larger trial)
    print("\n--- IDeate-like (1440 patients, mCRPC 1L) ---")
    ideate = analyze_trial(
        sample_size=1440,
        control_median=14.0,  # mCRPC 1L taxane
        design_hr=0.75,
        enroll_duration=30,
        min_followup=18,
    )
    print(f"Schoenfeld required: {ideate.schoenfeld_events}")
    print(f"Achievable events: {ideate.achievable_events}")
    print(f"Final events target: {ideate.final_events}")
    print(f"Event probability: {ideate.event_probability:.1%}")
    print(f"Is underpowered: {ideate.is_underpowered}")
    print(f"Critical HR (final): {ideate.final_critical_hr:.3f}")
    print(f"P(Success) base case: {ideate.pos_base_case:.1%}")
    print(f"P(Success) bull case: {ideate.pos_bull_case:.1%}")
    print(f"P(Success) bear case: {ideate.pos_bear_case:.1%}")
    print(f"Time to final: {ideate.final_time_months:.1f} months")

    print("\n--- Comparison ---")
    print(f"XALUTE final events: {xalute.final_events} vs IDeate: {ideate.final_events}")
    print(f"XALUTE P(Success): {xalute.pos_base_case:.1%} vs IDeate: {ideate.pos_base_case:.1%}")
    print(f"XALUTE critical HR: {xalute.final_critical_hr:.3f} vs IDeate: {ideate.final_critical_hr:.3f}")
