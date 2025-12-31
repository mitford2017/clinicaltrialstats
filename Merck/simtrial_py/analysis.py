"""
Analysis functions for trial simulation.

Provides tools to run multiple simulations and predict analysis dates.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple
from datetime import date, timedelta
from scipy.stats import norm

try:
    from .core import sim_pw_surv, get_cut_date_by_event
except ImportError:
    from core import sim_pw_surv, get_cut_date_by_event


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
        target_events = [int(sample_size * 0.5)]  # Default: 50% events
    
    if dropout_rate is None:
        treatments = fail_rate['treatment'].unique()
        dropout_rate = pd.DataFrame({
            'treatment': treatments,
            'duration': [100] * len(treatments),
            'rate': [0.001] * len(treatments)
        })
    
    if block is None:
        block = ['control', 'control', 'experimental', 'experimental']
    
    # Store results
    results = {event: [] for event in target_events}
    
    for i in range(n_sim):
        # Simulate trial
        data = sim_pw_surv(
            n=sample_size,
            enroll_rate=enroll_rate,
            fail_rate=fail_rate,
            dropout_rate=dropout_rate,
            block=block
        )
        
        # Get time to each target event count
        for event in target_events:
            cut_time = get_cut_date_by_event(data, event)
            results[event].append(cut_time)
    
    # Create summary
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
    enroll_rate: pd.DataFrame,
    ctrl_median: float,
    true_hr: float,
    design_hr: float,
    start_date: date,
    interim_events: Optional[int] = None,
    final_events: Optional[int] = None,
    interim_alpha: float = 0.01,
    final_alpha: float = 0.04,
    power: float = 0.80,
    allocation_ratio: float = 1.0,
    n_sim: int = 1000,
    seed: Optional[int] = None
) -> Dict:
    """
    Predict interim and final analysis dates using simulation.
    
    Parameters
    ----------
    sample_size : int
        Total enrollment
    enroll_rate : pd.DataFrame
        Enrollment rates (columns: duration, rate)
    ctrl_median : float
        Control arm median survival (months)
    true_hr : float
        Actual Hazard Ratio for generating data (e.g. 0.47)
    design_hr : float
        Design Hazard Ratio for calculating target events (e.g. 0.70)
    start_date : date
        Trial start date
    interim_events : int, optional
        Target events for interim analysis
    final_events : int, optional
        Target events for final analysis
    interim_alpha : float
        Alpha spend at interim (default 0.01)
    final_alpha : float
        Alpha spend at final (default 0.04)
    power : float
        Target power (default 0.80)
    allocation_ratio : float
        Allocation ratio experimental:control (default 1.0 = 1:1)
    n_sim : int
        Number of simulations (default 1000)
    seed : int, optional
        Random seed
        
    Returns
    -------
    dict with:
        - 'interim': Dict with date, events, critical_hr, percentiles
        - 'final': Dict with date, events, critical_hr, percentiles
        - 'event_curve': DataFrame for plotting expected events over time
    """
    # Calculate required events if not provided
    total_alpha = interim_alpha + final_alpha
    r = allocation_ratio
    
    def calc_required_events(alpha_adj: float, pwr: float) -> float:
        z_alpha = norm.ppf(1 - alpha_adj)
        z_beta = norm.ppf(pwr)
        d = ((r + 1) * (z_alpha + z_beta) / (np.sqrt(r) * np.log(design_hr))) ** 2
        return d
    
    # Two-sided test adjustment
    total_alpha_adj = total_alpha / 2
    interim_alpha_adj = interim_alpha / 2
    final_alpha_adj = final_alpha / 2
    
    # Calculate events based on DESIGN HR
    if final_events is None:
        final_events = int(np.ceil(calc_required_events(total_alpha_adj, power)))
    
    if interim_events is None:
        # Interim at ~66% information fraction
        interim_events = int(np.ceil(final_events * 0.66))
    
    # Build fail_rate DataFrame using TRUE HR
    ctrl_rate = np.log(2) / ctrl_median  # Hazard rate
    exp_rate = ctrl_rate * true_hr  # Experimental hazard rate
    
    fail_rate = pd.DataFrame({
        'treatment': ['control', 'experimental'],
        'duration': [1000, 1000],  # Essentially infinite
        'rate': [ctrl_rate, exp_rate]
    })
    
    # Low dropout
    dropout_rate = pd.DataFrame({
        'treatment': ['control', 'experimental'],
        'duration': [1000, 1000],
        'rate': [0.001, 0.001]
    })
    
    # Block randomization based on allocation ratio
    if allocation_ratio == 1.0:
        block = ['control', 'control', 'experimental', 'experimental']
    else:
        n_exp = int(np.round(allocation_ratio))
        block = ['control'] * 2 + ['experimental'] * (2 * n_exp)
    
    # Run simulations
    target_events = [interim_events, final_events]
    
    sim_results = simulate_trial(
        n_sim=n_sim,
        sample_size=sample_size,
        enroll_rate=enroll_rate,
        fail_rate=fail_rate,
        dropout_rate=dropout_rate,
        block=block,
        target_events=target_events,
        seed=seed
    )
    
    # Calculate critical HR at each analysis
    def calc_critical_hr(alpha_adj: float, n_events: int) -> float:
        z = norm.ppf(1 - alpha_adj)
        hr_crit = np.exp(-z * (r + 1) / np.sqrt(r * n_events))
        return hr_crit
    
    interim_hr_crit = calc_critical_hr(interim_alpha_adj, interim_events)
    final_hr_crit = calc_critical_hr(final_alpha_adj, final_events)
    
    # Convert times to dates
    def months_to_date(months: float) -> date:
        days = int(months * 30.44)
        return start_date + timedelta(days=days)
    
    interim_summary = sim_results['summary'][interim_events]
    final_summary = sim_results['summary'][final_events]
    
    # Build event curve (median expected events over time)
    max_time = final_summary['q95'] * 1.1
    time_points = np.linspace(0, max_time, 100)
    
    # Expected events at each time point (analytical approximation)
    # Using average of control and experimental rates weighted by allocation
    avg_rate = (ctrl_rate + exp_rate * r) / (1 + r)
    
    # Account for enrollment - use piecewise model
    total_enroll_time = enroll_rate['duration'].sum()
    enroll_by_time = []
    events_by_time = []
    
    for t in time_points:
        # Enrolled by time t
        enrolled = 0
        remaining_t = t
        for _, row in enroll_rate.iterrows():
            if remaining_t <= 0:
                break
            period_enrolled = min(remaining_t, row['duration']) * row['rate']
            enrolled += period_enrolled
            remaining_t -= row['duration']
        enrolled = min(enrolled, sample_size)
        enroll_by_time.append(enrolled)
        
        # Expected events - integrate over enrollment times
        # Simplified: assume uniform enrollment, average follow-up
        if t > 0 and enrolled > 0:
            avg_followup = t - min(t, total_enroll_time) / 2
            if avg_followup > 0:
                # Expected proportion with event
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
    
    # Calculate enrollment at analysis times
    def get_enrolled_count(t_months: float) -> int:
        enrolled = 0
        remaining_t = t_months
        for _, row in enroll_rate.iterrows():
            if remaining_t <= 0:
                break
            period_enrolled = min(remaining_t, row['duration']) * row['rate']
            enrolled += period_enrolled
            remaining_t -= row['duration']
        return int(min(enrolled, sample_size))

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
            'hr': true_hr,
            'start_date': start_date,
            'n_sim': n_sim
        }
    }
