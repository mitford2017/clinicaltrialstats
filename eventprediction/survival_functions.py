"""
Survival function classes for event prediction calculations.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional, Union
import numpy as np
from scipy.integrate import quad


@dataclass
class Sfn:
    """
    Survival function class for a single arm.
    
    Used in the integral to calculate event times in predict from parameters.
    
    Attributes
    ----------
    lambda_ : float
        Rate parameter for the arm. For lagged studies, this is for time > T.
    lambdaot : float
        Rate parameter for time < T (lagged studies only), NaN otherwise.
    lag_t : float
        Lag time for the survival function (0 for no lag).
    shape : float
        Weibull shape parameter.
    followup : float
        Follow up time for each subject (inf if no fixed followup).
    dropout_shape : float
        Weibull shape parameter for dropout hazard.
    dropout_lambda : float
        Rate parameter for dropout hazard (0 if no dropout).
    null_f : bool
        True if this represents NULL (e.g., second arm in single arm study).
    """
    lambda_: float
    lambdaot: float = field(default=np.nan)
    lag_t: float = 0.0
    shape: float = 1.0
    followup: float = np.inf
    dropout_shape: float = 1.0
    dropout_lambda: float = 0.0
    null_f: bool = False
    
    def __post_init__(self):
        if self.lag_t == 0:
            self.lambdaot = np.nan
    
    def survival_function_base(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Base survival function without dropouts or fixed follow up."""
        x = np.asarray(x)
        
        if self.lag_t == 0:
            return np.exp(-(self.lambda_ * x) ** self.shape)
        else:
            # Piecewise survival for lagged studies
            before_lag = np.exp(-(self.lambdaot * x) ** self.shape)
            after_lag = np.exp(
                -(self.lambda_ * x) ** self.shape + 
                (self.lambda_ ** self.shape - self.lambdaot ** self.shape) * self.lag_t ** self.shape
            )
            return np.where(x < self.lag_t, before_lag, after_lag)
    
    def hazard_function(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Hazard function."""
        x = np.asarray(x)
        
        if self.lag_t == 0:
            return self.shape * (self.lambda_ ** self.shape) * (x ** (self.shape - 1))
        else:
            before_lag = self.shape * (self.lambdaot ** self.shape) * (x ** (self.shape - 1))
            after_lag = self.shape * (self.lambda_ ** self.shape) * (x ** (self.shape - 1))
            return np.where(x < self.lag_t, before_lag, after_lag)
    
    def survival_function(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Full survival function including dropouts.
        
        S(x) = 0 for x > followup if finite follow up is used.
        """
        if self.null_f:
            return np.zeros_like(np.asarray(x))
        
        x = np.asarray(x)
        S = self.survival_function_base(x)
        
        if self.dropout_lambda != 0:
            S = S * np.exp(-(x * self.dropout_lambda) ** self.dropout_shape)
        
        if np.isfinite(self.followup):
            S = np.where(x <= self.followup, S, 0.0)
        
        return S
    
    def pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Probability density function."""
        if self.null_f:
            return np.zeros_like(np.asarray(x))
        
        x = np.asarray(x)
        S = self.survival_function(x)
        h = self.hazard_function(x)
        
        if self.dropout_lambda != 0:
            dropout_h = (self.dropout_lambda * self.dropout_shape) * \
                        (self.dropout_lambda * x) ** (self.dropout_shape - 1)
            pdf_val = S * (h + dropout_h)
        else:
            pdf_val = S * h
        
        if np.isfinite(self.followup):
            pdf_val = np.where(x <= self.followup, pdf_val, 0.0)
        
        return pdf_val
    
    def sfn(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Function for events integration: 1 - P(had event by time x).
        
        When no dropouts, this equals the survival function.
        """
        if self.null_f:
            return np.zeros_like(np.asarray(x))
        
        x = np.asarray(x)
        scalar_input = x.ndim == 0
        x = np.atleast_1d(x)
        
        f = self.survival_function_base
        h = self.hazard_function
        
        # No dropout case
        if self.dropout_lambda == 0:
            result = np.where(x < self.followup, f(x), f(self.followup))
        else:
            # With dropout - need to integrate
            def f2(t):
                return h(t) * f(t) * np.exp(-(t * self.dropout_lambda) ** self.dropout_shape)
            
            result = []
            for xi in x:
                upper = min(xi, self.followup)
                if upper <= 0:
                    result.append(1.0)
                else:
                    try:
                        integral, _ = quad(f2, 0, upper)
                        result.append(1 - integral)
                    except Exception:
                        result.append(f(xi))
            result = np.array(result)
        
        return result.item() if scalar_input else result


def create_sfn(lambda_: float, 
               lambdaot: float, 
               lag_t: float, 
               shape: float, 
               followup: float,
               dropout_shape: float = 1.0,
               dropout_lambda: float = 0.0) -> Sfn:
    """
    Create a survival function object.
    
    Parameters
    ----------
    lambda_ : float
        Rate parameter for time > T
    lambdaot : float
        Rate parameter for time < T (NA if no lag)
    lag_t : float
        Lag time (0 for no lag)
    shape : float
        Weibull shape parameter
    followup : float
        Follow up time (inf if no fixed followup)
    dropout_shape : float
        Dropout Weibull shape parameter
    dropout_lambda : float
        Dropout rate parameter (0 if no dropout)
        
    Returns
    -------
    Sfn
    """
    return Sfn(
        lambda_=lambda_,
        lambdaot=lambdaot if lag_t != 0 else np.nan,
        lag_t=lag_t,
        shape=shape,
        followup=followup,
        dropout_shape=dropout_shape,
        dropout_lambda=dropout_lambda,
        null_f=False
    )


def null_sfn() -> Sfn:
    """Create a null survival function for single arm trials."""
    return Sfn(
        lambda_=0.0,
        lambdaot=0.0,
        lag_t=0.0,
        shape=0.0,
        followup=0.0,
        dropout_shape=1.0,
        dropout_lambda=0.0,
        null_f=True
    )


def get_survival_functions(lambda_: np.ndarray,
                           lambdaot: np.ndarray,
                           lag_t: float,
                           is_single_arm: bool,
                           shape: float,
                           followup: float,
                           dropout_shape: float,
                           dropout_lambda: np.ndarray) -> list:
    """
    Create survival functions for both arms.
    
    Parameters
    ----------
    lambda_ : np.ndarray
        Rate parameters [control, experimental]
    lambdaot : np.ndarray
        Rate parameters for time < T
    lag_t : float
        Lag time
    is_single_arm : bool
        True if single arm study
    shape : float
        Weibull shape parameter
    followup : float
        Follow up time
    dropout_shape : float
        Dropout Weibull shape
    dropout_lambda : np.ndarray
        Dropout rate parameters
        
    Returns
    -------
    list
        List of Sfn objects [control, experimental]
    """
    range_ = [0] if is_single_arm else [0, 1]
    
    sfns = []
    for i in range_:
        sfn = create_sfn(
            lambda_=lambda_[i],
            lambdaot=lambdaot[i] if not np.isnan(lambdaot[i]) else np.nan,
            lag_t=lag_t,
            shape=shape,
            followup=followup,
            dropout_shape=dropout_shape,
            dropout_lambda=dropout_lambda[i] if i < len(dropout_lambda) else 0.0
        )
        sfns.append(sfn)
    
    if is_single_arm:
        sfns.append(null_sfn())
    
    return sfns


def events_integ(sfn: Sfn, B: float, k: float, t: Union[float, np.ndarray]) -> np.ndarray:
    """
    Calculate expected events using adaptive quadrature.
    
    Calculates integral of s^{k-1}*SurvFn(t-s) ds.
    
    Parameters
    ----------
    sfn : Sfn
        Survival function object
    B : float
        Accrual time
    k : float
        Non-uniformity of accrual
    t : float or np.ndarray
        Prediction times
        
    Returns
    -------
    np.ndarray
        Number of events at given times
    """
    if sfn.null_f:
        return np.zeros_like(np.asarray(t))
    
    t = np.atleast_1d(np.asarray(t))
    
    def myf(s, k, time):
        return (s ** (k - 1)) * sfn.sfn(time - s)
    
    integrated_vals = []
    for ti in t:
        if ti == 0:
            integrated_vals.append(0.0)
        else:
            upper = min(ti, B)
            try:
                # Handle lag discontinuity
                if sfn.lag_t == 0 or upper <= sfn.lag_t:
                    val, _ = quad(myf, 0, upper, args=(k, ti))
                else:
                    split_point = upper - sfn.lag_t
                    val1, _ = quad(myf, 0, split_point, args=(k, ti))
                    val2, _ = quad(myf, split_point, upper, args=(k, ti))
                    val = val1 + val2
                integrated_vals.append(val)
            except Exception:
                integrated_vals.append(0.0)
    
    integrated_vals = np.array(integrated_vals)
    return (np.minimum(t, B) / B) ** k - k * integrated_vals / (B ** k)


def atrisk_integ(sfn: Sfn, B: float, k: float, t: Union[float, np.ndarray]) -> np.ndarray:
    """
    Calculate the expected time at risk.
    
    Parameters
    ----------
    sfn : Sfn
        Survival function object
    B : float
        Accrual time
    k : float
        Non-uniformity of accrual
    t : float or np.ndarray
        Prediction times
        
    Returns
    -------
    np.ndarray
        Time at risk at given times
    """
    if sfn.null_f:
        return np.zeros_like(np.asarray(t))
    
    t = np.atleast_1d(np.asarray(t))
    
    # Time of subjects still on trial
    def myf1(s, k, time):
        return (k * s ** (k - 1) / B ** k) * sfn.survival_function(time - s) * (time - s)
    
    i1 = []
    for ti in t:
        if ti == 0:
            i1.append(0.0)
        else:
            try:
                val, _ = quad(myf1, 0, min(ti, B), args=(k, ti))
                i1.append(val)
            except Exception:
                i1.append(0.0)
    i1 = np.array(i1)
    
    # Time of subjects who had event or dropped out
    def internal_f(s, ti, k, B):
        return (k * s ** (k - 1) / B ** k) * sfn.pdf(ti - s) * (ti - s)
    
    def myf2(ti, k, B):
        upper = min(ti, B)
        lower = max(ti - sfn.followup, 0) if np.isfinite(sfn.followup) else 0
        if lower >= upper:
            return 0.0
        try:
            val, _ = quad(internal_f, lower, upper, args=(ti, k, B))
            return val
        except Exception:
            return 0.0
    
    i2 = []
    for ti in t:
        if ti == 0:
            i2.append(0.0)
        else:
            try:
                val, _ = quad(myf2, 0, ti, args=(k, B))
                i2.append(val)
            except Exception:
                i2.append(0.0)
    i2 = np.array(i2)
    
    # Time for subjects who dropped out at follow up period
    i3 = []
    for ti in t:
        if np.isfinite(sfn.followup) and ti >= sfn.followup:
            val = sfn.survival_function(sfn.followup) * sfn.followup * \
                  (min(B, ti - sfn.followup) ** k) / (B ** k)
            i3.append(val)
        else:
            i3.append(0.0)
    i3 = np.array(i3)
    
    return i1 + i2 + i3

