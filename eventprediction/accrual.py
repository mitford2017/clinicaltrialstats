"""
Accrual generation for clinical trial simulations.
"""

from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np
import pandas as pd
from datetime import date, timedelta

from .utils import fix_dates, ml_estimate_k


@dataclass
class AccrualGenerator:
    """
    Class to generate additional subject recruitment.
    
    Attributes
    ----------
    f : Callable
        Function that takes N (number of subjects) and returns recruitment times
    model : str
        Name of the accrual procedure
    text : str
        Summary text describing the accrual procedure
    """
    f: Callable[[int], np.ndarray]
    model: str
    text: str
    
    def generate(self, N: int) -> np.ndarray:
        """Generate N recruitment times."""
        return self.f(N)
    
    def __str__(self) -> str:
        return f"Accrual model: {self.model}\nModel Description:\n{self.text}"


def PoissonAccrual(start_date: str, rate: float) -> AccrualGenerator:
    """
    Create a Poisson process accrual generator.
    
    Parameters
    ----------
    start_date : str or date
        Start date for recruitment
    rate : float
        Rate of the Poisson process (subjects per day)
        
    Returns
    -------
    AccrualGenerator
    """
    if rate <= 0:
        raise ValueError("rate must be positive")
    
    start_date = fix_dates(start_date)
    if isinstance(start_date, pd.Series):
        start_date = start_date.iloc[0]
    
    def f(N: int) -> np.ndarray:
        inter_arrival = np.random.exponential(1 / rate, N)
        arrival_times = np.ceil(np.cumsum(inter_arrival))
        dates = pd.to_datetime(start_date) + pd.to_timedelta(arrival_times, unit='D')
        return dates.values
    
    r_str = f"{rate:.4f}" if rate < 0.01 else f"{rate:.2f}"
    text = f"a Poisson process with rate={r_str} and start date={start_date}."
    
    return AccrualGenerator(f=f, model="Poisson process", text=text)


def PowerLawAccrual(start_date: str, 
                    end_date: str, 
                    k: float, 
                    deterministic: bool = False,
                    rec_start_date: Optional[str] = None) -> AccrualGenerator:
    """
    Create an AccrualGenerator using power law recruitment.
    
    Subjects are accrued according to the c.d.f. G(t) = t^k / B^k
    where k is a parameter, t is time, and B is the recruitment period.
    
    Parameters
    ----------
    start_date : str or date
        Start of subject accrual period
    end_date : str or date
        End of accrual period
    k : float
        Non-uniformity accrual parameter
    deterministic : bool
        If True, use deterministic (non-stochastic) allocation
    rec_start_date : str or date, optional
        If used, subjects follow modified c.d.f.
        
    Returns
    -------
    AccrualGenerator
    """
    start_date = fix_dates(start_date)
    end_date = fix_dates(end_date)
    
    if isinstance(start_date, pd.Series):
        start_date = start_date.iloc[0]
    if isinstance(end_date, pd.Series):
        end_date = end_date.iloc[0]
    
    if rec_start_date is not None:
        rec_start_date = fix_dates(rec_start_date)
        if isinstance(rec_start_date, pd.Series):
            rec_start_date = rec_start_date.iloc[0]
    else:
        rec_start_date = start_date
    
    # Length of recruitment period
    B = (end_date - rec_start_date).days
    L = (start_date - rec_start_date).days
    
    if L < 0 or k <= 0 or B <= L:
        raise ValueError("Invalid arguments")
    
    if deterministic:
        def f(N: int) -> np.ndarray:
            i = np.arange(1, N + 1)
            days = ((i / N) * (B**k - L**k) + L**k) ** (1/k) - L
            dates = pd.to_datetime(start_date) + pd.to_timedelta(days, unit='D')
            return dates.values
        stext = "non-stochastic"
    else:
        def f(N: int) -> np.ndarray:
            u = np.random.uniform(0, 1, N)
            days = ((B**k - L**k) * u + L**k) ** (1/k) - L
            days = np.sort(days)
            dates = pd.to_datetime(start_date) + pd.to_timedelta(days, unit='D')
            return dates.values
        stext = "stochastic"
    
    text = (f"a {stext} allocation following G(t)=t^k/B^k, where k={k:.2f} "
            f"and B={B} days is the recruitment period [{rec_start_date}, {end_date}].")
    
    if L != 0:
        text += f" New subjects will be recruited after {start_date}."
    
    return AccrualGenerator(f=f, model="Power law allocation", text=text)


def estimate_accrual_parameter(event_data: 'EventData',
                               start_date: Optional[str] = None,
                               end_date: Optional[str] = None) -> float:
    """
    Estimate non-uniformity accrual parameter k.
    
    Note: This estimate assumes recruitment has completed.
    
    Parameters
    ----------
    event_data : EventData
        Data with recruitment information
    start_date : str or date, optional
        Start date for recruitment (default: first subject)
    end_date : str or date, optional
        End date for recruitment (default: last subject)
        
    Returns
    -------
    float
        Maximum likelihood estimate of k
    """
    import warnings
    warnings.warn("Note this estimate assumes recruitment has completed")
    
    df = event_data.subject_data
    
    if start_date is None:
        start_date = df['rand_date'].min()
    else:
        start_date = fix_dates(start_date)
        if isinstance(start_date, pd.Series):
            start_date = start_date.iloc[0]
    
    if end_date is None:
        end_date = df['rand_date'].max()
    else:
        end_date = fix_dates(end_date)
        if isinstance(end_date, pd.Series):
            end_date = end_date.iloc[0]
    
    B = (end_date - start_date).days
    if B <= 0:
        raise ValueError("end_date must be after start_date")
    
    recs = (df['rand_date'] - start_date).dt.days.values
    recs = np.where(recs == 0, 0.5, recs)
    
    if np.any(recs < 0) or np.any(recs > B):
        raise ValueError("Subjects recruited outside of recruitment period")
    
    return ml_estimate_k(B, recs)

