"""
Simulation functions for event prediction from data.
"""

from typing import Optional, Union
import numpy as np
import pandas as pd
import warnings

from .event_data import EventData, EmptyEventData
from .event_model import EventModel, FromDataSimParam
from .accrual import AccrualGenerator
from .results import FromDataResults, SimQOutput, SingleSimDetails
from .utils import get_ns


def simulate(model: Optional[EventModel] = None,
             data: Optional[EventData] = None,
             sim_params: Optional[FromDataSimParam] = None,
             accrual_generator: Optional[AccrualGenerator] = None,
             n_accrual: int = 0,
             n_sim: int = 10000,
             seed: Optional[int] = None,
             limit: float = 0.05,
             HR: Optional[float] = None,
             r: Optional[float] = None,
             dropout: Optional[dict] = None,
             **kwargs) -> FromDataResults:
    """
    Simulate event predictions from data.
    
    Parameters
    ----------
    model : EventModel, optional
        Fitted event model
    data : EventData, optional
        Event data to use
    sim_params : FromDataSimParam, optional
        Simulation parameters
    accrual_generator : AccrualGenerator, optional
        Generator for additional recruitment
    n_accrual : int
        Number of additional subjects to recruit
    n_sim : int
        Number of simulations
    seed : int, optional
        Random seed for reproducibility
    limit : float
        Quantile limit (default 0.05 for 5%-95% CI)
    HR : float, optional
        Hazard ratio for two-arm simulation
    r : float, optional
        Allocation ratio for two-arm simulation
    dropout : dict, optional
        Dropout parameters
        
    Returns
    -------
    FromDataResults
    """
    # Determine data and sim_params from model if not provided
    if model is not None:
        if data is None:
            data = model.event_data
        if sim_params is None:
            sim_params = model.sim_params
    
    if data is None or sim_params is None:
        raise ValueError("Must provide either model or both data and sim_params")
    
    # Validate arguments
    _validate_simulate_arguments(accrual_generator, n_accrual, n_sim, seed,
                                 limit, HR, r, data)
    
    # Calculate dropout parameters
    if dropout is None:
        dropout_rate = 0.0
        dropout_shape = 1.0
    else:
        prop = dropout.get('proportion', 0)
        time = dropout.get('time', 1)
        dropout_shape = dropout.get('shape', 1.0)
        if prop > 0:
            lambda_val = (-np.log(1 - prop)) ** (1/dropout_shape) / time
            dropout_rate = lambda_val
        else:
            dropout_rate = 0.0
    
    # Set seed
    if seed is not None:
        np.random.seed(seed)
    
    # Get subject data
    indat = data.subject_data.copy()
    n_subjects = len(indat)
    
    # Calculate accrual times
    rec_details = _calculate_accrual_times(n_accrual, n_sim, indat['rand_date'].values,
                                           accrual_generator)
    
    # Calculate quantiles for recruitment
    rec_quantiles = SimQOutput.from_matrix(rec_details, limit, n_sim)
    
    # Get new recruitment times
    if n_accrual > 0:
        new_recs = rec_details[:, -n_accrual:]
    else:
        new_recs = None
    
    # Generate simulation parameters
    single_sim_params = sim_params.generate_parameters(n_sim)
    
    # Perform simulations
    all_event_types = []
    all_event_times = []
    
    for i in range(n_sim):
        params = single_sim_params[i]
        result = _perform_one_simulation(
            params, n_subjects, n_accrual, indat, 
            new_recs[i] if new_recs is not None else None,
            HR, r, dropout_rate, dropout_shape, data.followup, sim_params
        )
        all_event_types.append(result['event_type'])
        all_event_times.append(result['event_times'])
    
    event_type = np.array(all_event_types)
    event_times = np.array(all_event_times)
    
    # Calculate quantiles for events
    event_quantiles = SimQOutput.from_matrix(event_times, limit, n_sim)
    
    # Calculate dropout quantiles (if dropout)
    if dropout_rate > 0:
        dropout_mask = event_type == 1
        dropout_times = np.where(dropout_mask, event_times, np.inf)
        dropout_quantiles = SimQOutput.from_matrix(dropout_times, limit, n_sim)
    else:
        dropout_quantiles = SimQOutput(
            median=np.array([]),
            lower=np.array([]),
            upper=np.array([]),
            limit=limit
        )
    
    # Create dummy accrual generator if not provided
    if accrual_generator is None:
        accrual_generator = AccrualGenerator(
            f=lambda N: None,
            model="NONE",
            text="NONE"
        )
    
    return FromDataResults(
        limit=limit,
        event_quantiles=event_quantiles,
        event_data=data,
        accrual_generator=accrual_generator,
        n_accrual=n_accrual,
        time_pred_data=pd.DataFrame(),
        event_pred_data=pd.DataFrame(),
        rec_quantiles=rec_quantiles,
        dropout_quantiles=dropout_quantiles,
        single_sim_details=SingleSimDetails(
            event_type=event_type.T,
            event_times=event_times.T,
            rec_times=rec_details.T
        ),
        dropout_shape=dropout_shape,
        dropout_rate=dropout_rate,
        sim_params=sim_params
    )


def _validate_simulate_arguments(accrual_generator, n_accrual, n_sim, seed,
                                  limit, HR, r, data):
    """Validate simulation arguments."""
    if (HR is None) != (r is None):
        raise ValueError("Both HR and r must be provided or neither")
    
    if HR is not None:
        if not isinstance(HR, (int, float)) or HR <= 0:
            raise ValueError("Invalid HR")
    
    if r is not None:
        if not isinstance(r, (int, float)) or r <= 0:
            raise ValueError("Invalid r")
    
    if HR is not None:
        df = data.subject_data
        if len(df) > 0 and (df['time'].notna() & (df['time'] != 0)).any():
            raise ValueError("If HR is used, time for each subject must be 0 "
                             "(only recruitment times are used)")
    
    if not 0 <= limit <= 0.5:
        raise ValueError("Invalid limit argument")
    
    if n_accrual < 0:
        raise ValueError("Invalid n_accrual argument")
    
    if n_accrual == 0:
        if accrual_generator is not None:
            warnings.warn("Using accrual generator but n_accrual = 0")
    else:
        if accrual_generator is None:
            raise ValueError("accrual_generator required when n_accrual > 0")
    
    if seed is not None and not isinstance(seed, int):
        warnings.warn("Invalid seed")
    
    if n_sim < 1:
        raise ValueError("Invalid n_sim argument")
    
    if n_accrual + len(data.subject_data) == 0:
        raise ValueError("No subjects!")


def _calculate_accrual_times(n_accrual: int, n_sim: int, rand_dates: np.ndarray,
                              accrual_generator: Optional[AccrualGenerator]) -> np.ndarray:
    """Calculate accrual times for all simulations."""
    # Convert to numeric (days since epoch)
    rand_dates_num = pd.to_datetime(rand_dates).astype(np.int64) // 10**9 // 86400
    
    # Replicate existing recruitment dates
    rs = np.tile(rand_dates_num, (n_sim, 1))
    
    if n_accrual == 0:
        return rs
    
    # Generate new recruitment times
    new_recs = []
    for _ in range(n_sim):
        dates = accrual_generator.generate(n_accrual)
        dates_num = pd.to_datetime(dates).astype(np.int64) // 10**9 // 86400
        new_recs.append(dates_num)
    new_recs = np.array(new_recs)
    
    if len(rand_dates) == 0:
        return new_recs
    
    if np.min(new_recs) < np.max(rand_dates_num):
        warnings.warn("Some new recruited subjects have rand_date earlier than existing subjects")
    
    return np.hstack([rs, new_recs])


def _perform_one_simulation(params: np.ndarray, 
                            n_subjects: int,
                            n_accrual: int,
                            indat: pd.DataFrame,
                            new_recs: Optional[np.ndarray],
                            HR: Optional[float],
                            r: Optional[float],
                            dropout_rate: float,
                            dropout_shape: float,
                            followup: float,
                            sim_params: FromDataSimParam) -> dict:
    """Perform a single simulation."""
    total = n_subjects + n_accrual
    
    # Event types: 0=event, 1=dropout, 2=followup
    event_type = np.zeros(total, dtype=int)
    
    # Event times (as days since epoch)
    event_times = np.full(total, np.nan)
    
    # Subjects who already left the trial (only from existing subjects)
    existing_mask = ((indat['has_event'] == 1) | 
                     (indat['withdrawn'] == 1) | 
                     (indat['censored_at_follow_up'] == 1))
    
    if existing_mask.any():
        # Get indices of existing subjects who have left
        existing_indices = np.where(existing_mask)[0]
        
        # Set their dates and types
        last_dates = (indat.loc[existing_mask, 'rand_date'] + 
                      pd.to_timedelta(indat.loc[existing_mask, 'time'] - 1, unit='D'))
        event_times[existing_indices] = last_dates.astype(np.int64).values // 10**9 // 86400
        
        event_type[existing_indices] = np.where(
            indat.loc[existing_mask, 'has_event'].values == 1, 0,
            np.where(indat.loc[existing_mask, 'censored_at_follow_up'].values == 1, 2, 1)
        )
    
    # Get subjects who need simulation
    sim_mask = ~existing_mask
    sim_indices = np.where(sim_mask)[0].tolist()
    
    # Add accrued subjects
    if n_accrual > 0:
        sim_indices.extend(range(n_subjects, total))
    
    if len(sim_indices) == 0:
        return {'event_type': event_type, 'event_times': event_times}
    
    # Get lower bounds and rand dates for subjects to simulate
    lower_bounds = []
    rand_dates = []
    
    for idx in sim_indices:
        if idx < n_subjects:
            lower_bounds.append(indat.iloc[idx]['time'] if not np.isnan(indat.iloc[idx]['time']) else 0)
            rand_dates.append(pd.to_datetime(indat.iloc[idx]['rand_date']).value // 10**9 // 86400)
        else:
            lower_bounds.append(0)
            rand_dates.append(int(new_recs[idx - n_subjects]))
    
    lower_bounds = np.array(lower_bounds)
    rand_dates = np.array(rand_dates)
    
    # Get HRs for each subject
    if HR is None:
        HRs = np.ones(len(sim_indices))
    else:
        Ns = get_ns(single_arm=False, r=r, N=len(sim_indices))
        HRs = np.concatenate([np.ones(Ns[0]), np.full(Ns[1], HR)])
        np.random.shuffle(HRs)
    
    # Generate event times
    times = sim_params.conditional_sample(lower_bounds, params, HRs)
    
    # Handle dropouts
    if dropout_rate > 0:
        dropout_params = np.array([0, dropout_rate, dropout_shape])
        dropout_times = FromDataSimParam._rcweibull(
            lower_bounds, dropout_rate, dropout_shape, np.ones(len(sim_indices))
        )
        is_dropout = times >= dropout_times
        times = np.minimum(times, dropout_times)
        
        for i, idx in enumerate(sim_indices):
            event_type[idx] = 1 if is_dropout[i] else 0
    
    # Handle followup
    if np.isfinite(followup):
        is_followup = times > followup
        times[is_followup] = followup
        for i, idx in enumerate(sim_indices):
            if is_followup[i]:
                event_type[idx] = 2
    
    # Convert times to dates
    for i, idx in enumerate(sim_indices):
        days_on_study = max(1, round(times[i]))
        event_times[idx] = rand_dates[i] + days_on_study - 1
    
    return {'event_type': event_type, 'event_times': event_times}

