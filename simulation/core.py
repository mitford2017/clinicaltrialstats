"""
Core simulation functions - Python port of Merck's simtrial R package.

This module provides piecewise exponential simulation for:
- Enrollment times
- Failure (event) times
- Dropout times
"""

import numpy as np
import pandas as pd
from typing import Optional, List


def rpwexp(n: int, fail_rate: pd.DataFrame) -> np.ndarray:
    """
    Generate random piecewise exponential failure times.

    Uses inverse CDF method for piecewise exponential distribution.

    Parameters
    ----------
    n : int
        Number of observations to generate
    fail_rate : pd.DataFrame
        DataFrame with columns:
        - 'duration': Duration of each period
        - 'rate': Hazard rate during that period

    Returns
    -------
    np.ndarray
        Random failure times
    """
    u = np.random.uniform(0, 1, n)

    durations = fail_rate['duration'].values
    rates = fail_rate['rate'].values

    # Period boundaries
    boundaries = np.concatenate([[0], np.cumsum(durations)])

    # Cumulative hazard at each boundary
    cum_hazard = np.concatenate([[0], np.cumsum(durations * rates)])

    times = np.zeros(n)

    for i in range(n):
        # Target cumulative hazard: H(t) = -log(u)
        target_H = -np.log(u[i])

        # Find which period this falls into
        period_idx = np.searchsorted(cum_hazard[1:], target_H)

        if period_idx >= len(rates):
            period_idx = len(rates) - 1

        H_start = cum_hazard[period_idx]
        t_start = boundaries[period_idx]
        remaining_H = target_H - H_start

        if rates[period_idx] > 0:
            times[i] = t_start + remaining_H / rates[period_idx]
        else:
            times[i] = np.inf

    return times


def rpwexp_enroll(n: int, enroll_rate: pd.DataFrame) -> np.ndarray:
    """
    Generate piecewise exponential enrollment times using Poisson process.

    Parameters
    ----------
    n : int
        Number of patients to enroll
    enroll_rate : pd.DataFrame
        DataFrame with columns:
        - 'duration': Duration of each enrollment period
        - 'rate': Enrollment rate (patients per time unit)

    Returns
    -------
    np.ndarray
        Sorted enrollment times for n patients
    """
    if len(enroll_rate) == 1:
        rate = enroll_rate['rate'].values[0]
        if rate <= 0:
            raise ValueError("Enrollment rate must be > 0")
        inter_arrival = np.random.exponential(1/rate, n)
        return np.cumsum(inter_arrival)

    durations = enroll_rate['duration'].values
    rates = enroll_rate['rate'].values

    # Expected number in each period
    lambdas = durations * rates

    # Period boundaries
    boundaries = np.concatenate([[0], np.cumsum(durations)])

    # Generate Poisson counts for each period
    all_times = []
    for i, (lam, origin, finish) in enumerate(zip(lambdas, boundaries[:-1], boundaries[1:])):
        count = np.random.poisson(lam)
        if count > 0:
            period_times = np.random.uniform(origin, finish, count)
            all_times.extend(period_times)

    all_times = np.sort(all_times)

    if len(all_times) >= n:
        return all_times[:n]

    # Need more - extend with exponential inter-arrivals at last rate
    n_add = n - len(all_times)
    last_rate = rates[-1]
    if last_rate <= 0:
        raise ValueError("Last enrollment rate must be > 0")

    last_time = boundaries[-1] if len(all_times) == 0 else max(all_times[-1], boundaries[-1])
    additional = last_time + np.cumsum(np.random.exponential(1/last_rate, n_add))

    return np.concatenate([all_times, additional])[:n]


def randomize_by_fixed_block(n: int, block: List[str]) -> np.ndarray:
    """
    Randomize treatments using fixed block randomization.

    Parameters
    ----------
    n : int
        Number of patients to randomize
    block : list of str
        Treatment assignments within each block

    Returns
    -------
    np.ndarray
        Treatment assignments for n patients
    """
    block = np.array(block)
    block_size = len(block)

    n_full_blocks = n // block_size
    remainder = n % block_size

    treatments = []
    for _ in range(n_full_blocks):
        shuffled = np.random.permutation(block)
        treatments.extend(shuffled)

    if remainder > 0:
        shuffled = np.random.permutation(block)
        treatments.extend(shuffled[:remainder])

    return np.array(treatments)


def sim_pw_surv(
    n: int = 100,
    enroll_rate: Optional[pd.DataFrame] = None,
    fail_rate: Optional[pd.DataFrame] = None,
    dropout_rate: Optional[pd.DataFrame] = None,
    block: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Simulate a stratified time-to-event randomized trial.

    Generates individual patient data with enrollment times, failure times,
    dropout times, and treatment assignments.

    Parameters
    ----------
    n : int
        Number of patients
    enroll_rate : pd.DataFrame
        Enrollment rates by period (columns: duration, rate)
    fail_rate : pd.DataFrame
        Failure rates by treatment (columns: treatment, duration, rate)
    dropout_rate : pd.DataFrame
        Dropout rates by treatment (columns: treatment, duration, rate)
    block : list of str
        Block randomization pattern

    Returns
    -------
    pd.DataFrame with columns:
        - enroll_time: Enrollment time
        - treatment: Treatment assignment
        - fail_time: Time from enrollment to failure
        - dropout_time: Time from enrollment to dropout
        - cte: Calendar time of event (enroll + min(fail, dropout))
        - fail: 1 if event was failure, 0 if dropout
    """
    if enroll_rate is None:
        enroll_rate = pd.DataFrame({'duration': [1], 'rate': [n]})

    if fail_rate is None:
        fail_rate = pd.DataFrame({
            'treatment': ['control', 'experimental'],
            'duration': [1000, 1000],
            'rate': [np.log(2)/12, np.log(2)/18]
        })

    if dropout_rate is None:
        dropout_rate = pd.DataFrame({
            'treatment': ['control', 'experimental'],
            'duration': [1000, 1000],
            'rate': [0.001, 0.001]
        })

    if block is None:
        block = ['control', 'control', 'experimental', 'experimental']

    # Generate enrollment times
    enroll_times = rpwexp_enroll(n, enroll_rate)

    # Randomize treatments
    treatments = randomize_by_fixed_block(n, block)

    # Generate failure and dropout times per treatment
    fail_times = np.zeros(n)
    dropout_times = np.zeros(n)

    for trt in np.unique(treatments):
        mask = treatments == trt
        n_trt = mask.sum()

        trt_fail = fail_rate[fail_rate['treatment'] == trt][['duration', 'rate']]
        if len(trt_fail) == 0:
            trt_fail = fail_rate[['duration', 'rate']].drop_duplicates()
        fail_times[mask] = rpwexp(n_trt, trt_fail.reset_index(drop=True))

        trt_drop = dropout_rate[dropout_rate['treatment'] == trt][['duration', 'rate']]
        if len(trt_drop) == 0:
            trt_drop = dropout_rate[['duration', 'rate']].drop_duplicates()
        dropout_times[mask] = rpwexp(n_trt, trt_drop.reset_index(drop=True))

    # Calendar time of event
    tte = np.minimum(fail_times, dropout_times)
    cte = enroll_times + tte
    fail_indicator = (fail_times <= dropout_times).astype(int)

    return pd.DataFrame({
        'enroll_time': enroll_times,
        'treatment': treatments,
        'fail_time': fail_times,
        'dropout_time': dropout_times,
        'tte': tte,
        'cte': cte,
        'fail': fail_indicator
    })


def get_cut_date_by_event(data: pd.DataFrame, event: int) -> float:
    """
    Get calendar date at which target event count is reached.

    Parameters
    ----------
    data : pd.DataFrame
        Trial data from sim_pw_surv()
    event : int
        Target event count

    Returns
    -------
    float
        Calendar time when event count is reached
    """
    failures = data[data['fail'] == 1].copy()
    failures = failures.sort_values('cte')

    if len(failures) < event:
        return failures['cte'].max() if len(failures) > 0 else np.inf

    return failures['cte'].iloc[event - 1]


def cut_data_by_date(data: pd.DataFrame, cut_date: float) -> pd.DataFrame:
    """
    Cut trial data at a specific calendar date for analysis.

    Parameters
    ----------
    data : pd.DataFrame
        Trial data from sim_pw_surv()
    cut_date : float
        Calendar date for data cutoff

    Returns
    -------
    pd.DataFrame
        Analysis dataset with tte and event indicator at cutoff
    """
    result = data.copy()
    result = result[result['enroll_time'] <= cut_date].copy()
    result['tte'] = np.minimum(result['cte'], cut_date) - result['enroll_time']
    result['event'] = ((result['fail'] == 1) & (result['cte'] <= cut_date)).astype(int)
    return result
