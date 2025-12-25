"""
Lag effect classes for clinical trial simulations.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class CtrlSpec:
    """Control arm specification."""
    median: float
    text: str
    
    @classmethod
    def from_median(cls, median: float) -> 'CtrlSpec':
        """Create CtrlSpec from median survival time."""
        if np.isnan(median):
            return cls(median=median, text="")
        return cls(median=median, text=f"median survival {median}")
    
    @classmethod
    def from_proportion(cls, time: float, proportion_had_event: float, shape: float) -> 'CtrlSpec':
        """
        Create CtrlSpec from time and proportion.
        
        Parameters
        ----------
        time : float
            Time at which proportion is measured
        proportion_had_event : float
            Proportion who have had an event by time
        shape : float
            Weibull shape parameter
        """
        # Calculate median from proportion
        # S(t) = exp(-(lambda*t)^shape) = 1 - proportion
        # lambda*t = (-log(1-proportion))^(1/shape)
        # lambda = (-log(1-proportion))^(1/shape) / t
        # median: S(m) = 0.5 => lambda*m = log(2)^(1/shape)
        # m = log(2)^(1/shape) / lambda
        
        lambda_val = (-np.log(1 - proportion_had_event)) ** (1/shape) / time
        median = np.log(2) ** (1/shape) / lambda_val
        
        text = f"{proportion_had_event*100:.1f}% had event by {time}"
        return cls(median=median, text=text)


@dataclass
class LagEffect:
    """
    Parameter settings for including a lagged effect.
    
    The lambda and HR in the Study class will be used for time > T.
    
    Attributes
    ----------
    lag_t : float
        Lag time (T)
    ctrl_spec : CtrlSpec
        Control median specification for time period [0,T]
    l_hazard_ratio : float
        Hazard ratio for time period [0,T]
    """
    lag_t: float
    ctrl_spec: CtrlSpec
    l_hazard_ratio: float
    
    def __post_init__(self):
        if self.lag_t < 0:
            raise ValueError("lag_t must be non-negative")
        
        if not np.isnan(self.l_hazard_ratio):
            if self.l_hazard_ratio < 0 or self.l_hazard_ratio > 1:
                raise ValueError("Hazard ratio must be in [0, 1]")
    
    def is_null_lag(self) -> bool:
        """Check if this is a null lag (no lag effect)."""
        return self.lag_t == 0
    
    def __str__(self) -> str:
        if self.is_null_lag():
            return "No Lag"
        
        text = f"{self.lag_t} months of lag during which\n"
        text += f"control group survival {self.ctrl_spec.text}\n"
        if not np.isnan(self.l_hazard_ratio):
            text += f"and the hazard ratio is {self.l_hazard_ratio}"
        return text


def NullLag() -> LagEffect:
    """Create a LagEffect object with no lag."""
    return LagEffect(
        lag_t=0,
        ctrl_spec=CtrlSpec.from_median(np.nan),
        l_hazard_ratio=np.nan
    )


def create_lag_effect(lag_t: float, 
                      l_ctr_median: float = np.nan, 
                      l_hazard_ratio: float = np.nan) -> LagEffect:
    """
    Create a LagEffect object.
    
    Parameters
    ----------
    lag_t : float
        Lag time (T)
    l_ctr_median : float, optional
        Control median for time period [0,T]
    l_hazard_ratio : float, optional
        Hazard ratio for time period [0,T]
        
    Returns
    -------
    LagEffect
    """
    return LagEffect(
        lag_t=lag_t,
        ctrl_spec=CtrlSpec.from_median(l_ctr_median),
        l_hazard_ratio=l_hazard_ratio
    )

