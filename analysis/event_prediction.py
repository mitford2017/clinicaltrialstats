"""
Event prediction using Monte Carlo simulation.

Provides analysis functions to predict interim and final analysis dates.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict
from datetime import date, timedelta, datetime
from scipy.stats import norm

try:
    from ..simulation.core import sim_pw_surv, get_cut_date_by_event
except ImportError:
    from simulation.core import sim_pw_surv, get_cut_date_by_event


def simulate_trial(
    n_sim: int,
    sample_size: int,
    enroll_rate: pd.DataFrame,
    fail_rate: pd.DataFrame,
    dropout_rate: Optional[pd.DataFrame] = None,
    block: Optional[List[str]] = None,
    target_events: Optional[List[int]] = None,
    seed: Optional[int] = None
) -> Dict:
    """
    Run multiple trial simulations and collect results.

    Parameters
    ----------
    n_sim : int
        Number of simulations to run
    sample_size : int
        Total enrollment per simulation
    enroll_rate : pd.DataFrame
        Enrollment rate schedule (columns: duration, rate)
    fail_rate : pd.DataFrame
        Failure rates by treatment (columns: treatment, duration, rate)
    dropout_rate : pd.DataFrame, optional
        Dropout rates by treatment
    block : list, optional
        Block randomization pattern
    target_events : list of int, optional
        Event counts to track (e.g., [250, 356] for interim and final)
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    dict with:
        - 'event_times': DataFrame of calendar times to reach each target event
        - 'summary': Summary statistics for each target
    """
    if seed is not None:
        np.random.seed(seed)

    if target_events is None:
        target_events = [int(sample_size * 0.5)]

    if dropout_rate is None:
        treatments = fail_rate['treatment'].unique()
        dropout_rate = pd.DataFrame({
            'treatment': treatments,
            'duration': [1000] * len(treatments),
            'rate': [0.001] * len(treatments)
        })

    if block is None:
        block = ['control', 'control', 'experimental', 'experimental']

    results = {event: [] for event in target_events}

    for i in range(n_sim):
        data = sim_pw_surv(
            n=sample_size,
            enroll_rate=enroll_rate,
            fail_rate=fail_rate,
            dropout_rate=dropout_rate,
            block=block
        )

        for event in target_events:
            cut_time = get_cut_date_by_event(data, event)
            results[event].append(cut_time)

    event_times = pd.DataFrame(results)

    summary = {}
    for event in target_events:
        times = event_times[event].values
        summary[event] = {
            'median': np.median(times),
            'mean': np.mean(times),
            'q05': np.percentile(times, 5),
            'q25': np.percentile(times, 25),
            'q75': np.percentile(times, 75),
            'q95': np.percentile(times, 95),
        }

    return {
        'event_times': event_times,
        'summary': summary,
        'n_sim': n_sim
    }


def predict_analysis_dates(
    sample_size: int,
    enroll_duration: float,
    ctrl_median: float,
    scenario_hr: float,
    start_date: date,
    design_hr: Optional[float] = None,
    interim_events: Optional[int] = None,
    final_events: Optional[int] = None,
    interim_alpha: float = 0.01,
    final_alpha: float = 0.024,
    power: float = 0.90,
    allocation_ratio: float = 1.0,
    n_sim: int = 1000,
    seed: Optional[int] = None
) -> Dict:
    """
    Predict interim and final analysis dates using Monte Carlo simulation.

    Parameters
    ----------
    sample_size : int
        Total enrollment
    enroll_duration : float
        Enrollment duration in months
    ctrl_median : float
        Control arm median survival (months)
    scenario_hr : float
        True Hazard Ratio for simulation (what actually happens)
    start_date : date
        Trial start date
    design_hr : float, optional
        Hazard Ratio used for sample size calculation (protocol assumption).
        If None, defaults to scenario_hr.
    interim_events : int, optional
        Target events for interim analysis
    final_events : int, optional
        Target events for final analysis
    interim_alpha : float
        Alpha spend at interim (default 0.01)
    final_alpha : float
        Alpha spend at final (default 0.024)
    power : float
        Target power (default 0.90)
    allocation_ratio : float
        Allocation ratio experimental:control (default 1.0)
    n_sim : int
        Number of simulations (default 1000)
    seed : int, optional
        Random seed

    Returns
    -------
    dict with interim, final analysis predictions and event curve
    """
    # Use design_hr for event calculation, scenario_hr for simulation
    calc_hr = design_hr if design_hr is not None else scenario_hr

    total_alpha = interim_alpha + final_alpha
    r = allocation_ratio

    def calc_required_events(alpha_adj: float, pwr: float) -> float:
        z_alpha = norm.ppf(1 - alpha_adj)
        z_beta = norm.ppf(pwr)
        d = ((r + 1) * (z_alpha + z_beta) / (np.sqrt(r) * np.log(calc_hr))) ** 2
        return d

    # Two-sided test adjustment
    total_alpha_adj = total_alpha / 2
    interim_alpha_adj = interim_alpha / 2
    final_alpha_adj = final_alpha / 2

    if final_events is None:
        final_events = int(np.ceil(calc_required_events(total_alpha_adj, power)))

    if interim_events is None:
        interim_events = int(np.ceil(final_events * 0.66))

    # Build enrollment rate DataFrame
    enroll_rate = pd.DataFrame({
        'duration': [enroll_duration],
        'rate': [sample_size / enroll_duration]
    })

    # Build failure rate using scenario_hr (what actually happens)
    ctrl_rate = np.log(2) / ctrl_median
    exp_rate = ctrl_rate * scenario_hr

    fail_rate = pd.DataFrame({
        'treatment': ['control', 'experimental'],
        'duration': [1000, 1000],
        'rate': [ctrl_rate, exp_rate]
    })

    dropout_rate = pd.DataFrame({
        'treatment': ['control', 'experimental'],
        'duration': [1000, 1000],
        'rate': [0.001, 0.001]
    })

    if allocation_ratio == 1.0:
        block = ['control', 'control', 'experimental', 'experimental']
    else:
        n_exp = int(np.round(allocation_ratio))
        block = ['control'] * 2 + ['experimental'] * (2 * n_exp)

    target_events_list = [interim_events, final_events]

    sim_results = simulate_trial(
        n_sim=n_sim,
        sample_size=sample_size,
        enroll_rate=enroll_rate,
        fail_rate=fail_rate,
        dropout_rate=dropout_rate,
        block=block,
        target_events=target_events_list,
        seed=seed
    )

    # Calculate critical HR at each analysis
    def calc_critical_hr(alpha_adj: float, n_events: int) -> float:
        z = norm.ppf(1 - alpha_adj)
        hr_crit = np.exp(-z * (r + 1) / np.sqrt(r * n_events))
        return hr_crit

    interim_hr_crit = calc_critical_hr(interim_alpha_adj, interim_events)
    final_hr_crit = calc_critical_hr(final_alpha_adj, final_events)

    def months_to_date(months: float) -> date:
        days = int(months * 30.44)
        return start_date + timedelta(days=days)

    interim_summary = sim_results['summary'][interim_events]
    final_summary = sim_results['summary'][final_events]

    # Build event curve
    max_time = final_summary['q95'] * 1.1
    time_points = np.linspace(0, max_time, 100)

    avg_rate = (ctrl_rate + exp_rate * r) / (1 + r)

    enroll_by_time = []
    events_by_time = []

    for t in time_points:
        enrolled = min(t * (sample_size / enroll_duration), sample_size)
        enroll_by_time.append(enrolled)

        if t > 0 and enrolled > 0:
            avg_followup = t - min(t, enroll_duration) / 2
            if avg_followup > 0:
                exp_events = enrolled * (1 - np.exp(-avg_rate * avg_followup))
            else:
                exp_events = 0
        else:
            exp_events = 0
        events_by_time.append(exp_events)

    event_curve = pd.DataFrame({
        'time': time_points,
        'date': [months_to_date(t) for t in time_points],
        'enrollment': enroll_by_time,
        'events': events_by_time
    })

    def get_enrolled_count(t_months: float) -> int:
        return int(min(t_months * (sample_size / enroll_duration), sample_size))

    return {
        'interim': {
            'events': interim_events,
            'median_time': interim_summary['median'],
            'median_date': months_to_date(interim_summary['median']),
            'q05_date': months_to_date(interim_summary['q05']),
            'q95_date': months_to_date(interim_summary['q95']),
            'critical_hr': interim_hr_crit,
            'alpha': interim_alpha,
            'enrolled_count': get_enrolled_count(interim_summary['median']),
            'enrolled_pct': get_enrolled_count(interim_summary['median']) / sample_size
        },
        'final': {
            'events': final_events,
            'median_time': final_summary['median'],
            'median_date': months_to_date(final_summary['median']),
            'q05_date': months_to_date(final_summary['q05']),
            'q95_date': months_to_date(final_summary['q95']),
            'critical_hr': final_hr_crit,
            'alpha': final_alpha,
            'enrolled_count': get_enrolled_count(final_summary['median']),
            'enrolled_pct': get_enrolled_count(final_summary['median']) / sample_size
        },
        'event_curve': event_curve,
        'simulation_results': sim_results,
        'parameters': {
            'sample_size': sample_size,
            'ctrl_median': ctrl_median,
            'design_hr': calc_hr,
            'scenario_hr': scenario_hr,
            'start_date': start_date,
            'n_sim': n_sim,
            'interim_events': interim_events,
            'final_events': final_events
        }
    }


def calc_probability_of_success(
    scenario_hr: float,
    final_events: int,
    critical_hr: float,
) -> float:
    """
    Calculate probability of success given the true HR.

    Parameters
    ----------
    scenario_hr : float
        True hazard ratio
    final_events : int
        Number of events at final analysis
    critical_hr : float
        Critical HR threshold for significance

    Returns
    -------
    float
        Probability of observing HR <= critical_hr
    """
    # Z-score for observed HR under true HR
    # Under H_a: Z ~ N(theta, 1) where theta = log(HR) * sqrt(d/4)
    log_scenario = np.log(scenario_hr)
    log_critical = np.log(critical_hr)

    se = 2 / np.sqrt(final_events)  # Approximate SE of log(HR)

    # P(observed log(HR) < log(critical_hr))
    z = (log_critical - log_scenario) / se
    prob = norm.cdf(z)

    return prob
