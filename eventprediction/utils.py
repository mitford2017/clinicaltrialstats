"""
Utility functions used throughout the eventprediction package.
"""

import numpy as np
from scipy.special import gamma
from scipy.optimize import brentq
from typing import Union, List, Optional
from datetime import date, datetime
import pandas as pd
import csv


def standarddaysinyear() -> float:
    """Return the number of days in a year (default 365.25)."""
    return 365.25


def fix_dates(vec: Union[str, date, List, pd.Series, None]) -> Optional[pd.Series]:
    """
    Convert various date formats to pandas datetime.
    
    Accepts:
    - YYYY-MM-DD
    - DD/MM/YYYY
    - DD Month YYYY
    
    Parameters
    ----------
    vec : str, date, list, or pd.Series
        Date values to convert
        
    Returns
    -------
    pd.Series or None
        Converted dates as pandas datetime
    """
    if vec is None:
        return None
    
    if isinstance(vec, (date, datetime)):
        return pd.to_datetime(vec)
    
    if isinstance(vec, str):
        vec = [vec]
    
    if isinstance(vec, (list, np.ndarray)):
        vec = pd.Series(vec)
    
    if isinstance(vec, pd.Series):
        # Replace empty strings with NaT
        vec = vec.replace('', pd.NaT)
        
        # Try different date formats
        formats = ['%Y-%m-%d', '%d/%m/%Y', '%d %b %Y', '%d %B %Y']
        
        for fmt in formats:
            try:
                result = pd.to_datetime(vec, format=fmt, errors='coerce')
                if result.notna().sum() == vec.notna().sum():
                    return result
            except (ValueError, TypeError):
                continue
        
        # Fall back to pandas' flexible parser
        try:
            return pd.to_datetime(vec, infer_datetime_format=True)
        except (ValueError, TypeError):
            pass
    
    raise ValueError("Error reading date format. It should be of the form "
                     "YYYY-MM-DD, DD/MM/YYYY or DD Month YYYY")


def round_force_output_zeros(number: float, dp: int) -> str:
    """
    Round a number and force trailing zeros to be output.
    
    Parameters
    ----------
    number : float
        Number to round
    dp : int
        Number of decimal places
        
    Returns
    -------
    str
        Formatted number string
    """
    if dp < 1:
        raise ValueError("dp must be positive")
    return f"{number:.{dp}f}"


def get_ns(single_arm: bool, r: float, N: int) -> tuple:
    """
    Calculate the number of subjects on each arm.
    
    Parameters
    ----------
    single_arm : bool
        True if study is single arm
    r : float
        Allocation ratio (control:experimental = 1:r)
    N : int
        Total number of subjects
        
    Returns
    -------
    tuple
        (N_control, N_experimental)
    """
    if single_arm:
        return (N, 0)
    
    n_exp = int(np.floor((r / (r + 1)) * N))
    n_ctrl = N - n_exp
    
    if n_ctrl == 0 or n_exp == 0:
        raise ValueError("Too few subjects recruited for given randomization balance. "
                         "All subjects would be recruited onto the same arm.")
    
    return (n_ctrl, n_exp)


def lambda_calc(median: float, hr: float, shape: float) -> np.ndarray:
    """
    Convert HR and control median into lambda (rate parameter).
    
    Parameters
    ----------
    median : float
        Control median
    hr : float
        Hazard ratio
    shape : float
        Weibull shape parameter
        
    Returns
    -------
    np.ndarray
        Rate parameters [control, experimental]
    """
    control_lambda = np.log(2) ** (1/shape) / median
    if np.isnan(hr):
        return np.array([control_lambda, control_lambda])
    exp_lambda = np.log(2) ** (1/shape) / (median / (hr ** (1/shape)))
    return np.array([control_lambda, exp_lambda])


def csv_sniffer(path: str, delim_options: List[str] = [',', ';', '\t']) -> pd.DataFrame:
    """
    Read a CSV file, automatically detecting the delimiter.
    
    Parameters
    ----------
    path : str
        Path to the CSV file
    delim_options : list
        Possible delimiters to try
        
    Returns
    -------
    pd.DataFrame
        Loaded data
    """
    with open(path, 'r') as f:
        lines = [line for line in f.readlines() if line.strip()]
    
    for delim in delim_options:
        counts = [line.count(delim) for line in lines]
        if len(set(counts)) == 1 and counts[0] > 0:
            return pd.read_csv(path, sep=delim)
    
    raise ValueError("Cannot determine delimiter")


def fit_mixture_model(HR: float, r: float, M: float) -> dict:
    """
    Fit a single Weibull distribution using method of moments.
    
    Given trial assumptions describing an exponential model,
    use the method of moments to fit a single Weibull distribution.
    
    Parameters
    ----------
    HR : float
        Hazard ratio
    r : float
        Randomization balance
    M : float
        Control arm median
        
    Returns
    -------
    dict
        {'rate': rate, 'shape': shape}
    """
    if HR <= 0 or r <= 0 or M <= 0:
        raise ValueError("All arguments must be positive")
    
    ml2 = M / np.log(2)
    mu = (ml2 * (1 + r / HR)) / (r + 1)
    sigma = (ml2 / (1 + r)) * np.sqrt(1 + 2*r - 2*r/HR + (r * (r + 2)) / (HR * HR))
    
    return _root_find_rate_shape(mu, sigma)


def _root_find_rate_shape(mu: float, sigma: float) -> dict:
    """
    Fit Weibull shape and rate given mean and standard deviation.
    
    Parameters
    ----------
    mu : float
        Mean of Weibull
    sigma : float
        Standard deviation
        
    Returns
    -------
    dict
        {'rate': rate, 'shape': shape}
    """
    def f(x):
        return sigma / mu - np.sqrt(gamma(1 + 2/x) - gamma(1 + 1/x)**2) / gamma(1 + 1/x)
    
    try:
        shape = brentq(f, 0.1, 10)
    except ValueError:
        import warnings
        warnings.warn("Error fitting shape, returning NA")
        return {'rate': np.nan, 'shape': np.nan}
    
    rate = gamma(1 + 1/shape) / mu
    return {'rate': rate, 'shape': shape}


def ml_estimate_k(B: float, s_i: np.ndarray) -> float:
    """
    Maximum likelihood estimate of non-uniform accrual parameter k.
    
    Parameters
    ----------
    B : float
        Recruitment period
    s_i : np.ndarray
        Vector of recruitment times
        
    Returns
    -------
    float
        Estimate of k
    """
    if np.any(s_i <= 0) or B <= 0 or np.any(s_i > B):
        raise ValueError("Invalid arguments in ml_estimate_k")
    
    s = np.mean(np.log(s_i))
    return 1 / (np.log(B) - s)


def add_line_breaks(text: str, text_width: int) -> str:
    """
    Add line breaks to text at specified width.
    
    Parameters
    ----------
    text : str
        Text to wrap
    text_width : int
        Target character width
        
    Returns
    -------
    str
        Wrapped text
    """
    import textwrap
    return '\n'.join(textwrap.wrap(text, width=text_width))


def average_rec(N: int, first_date: float, last_date: float) -> float:
    """
    Calculate the average recruitment rate (subjects/day).
    
    Parameters
    ----------
    N : int
        Number of subjects recruited
    first_date : float
        First recruitment date (numeric)
    last_date : float
        Last recruitment date (numeric)
        
    Returns
    -------
    float
        Average recruitment rate
    """
    return N / (last_date - first_date + 1)


def required_events(r: float, alpha: float, power: float, HR: float, N: int) -> float:
    """
    Calculate the required number of events to reject H0: ln(HR)=0.
    
    Parameters
    ----------
    r : float
        Allocation ratio
    alpha : float
        Significance level (already adjusted for one/two-sided)
    power : float
        Target power
    HR : float
        Hazard ratio
    N : int
        Total number of subjects
        
    Returns
    -------
    float
        Required number of events
    """
    from scipy.stats import norm
    
    events_req = ((r + 1) * (norm.ppf(1 - alpha) + norm.ppf(power))) / \
                 (np.sqrt(r) * np.log(HR))
    events_req = events_req ** 2
    
    if events_req > N:
        import warnings
        warnings.warn(f"Given these settings, the required number of events is {round(events_req)}, "
                      f"which is more than there are patients. Please increase trial size!")
    
    return events_req

