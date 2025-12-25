"""
Study class definitions for clinical trial event prediction.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Union
import numpy as np
from scipy.stats import norm

from .lag_effect import LagEffect, NullLag, CtrlSpec
from .survival_functions import (
    get_survival_functions, events_integ, atrisk_integ, Sfn
)
from .utils import (
    lambda_calc, get_ns, standarddaysinyear, required_events
)


@dataclass
class Study:
    """
    Class defining a clinical trial study.
    
    Attributes
    ----------
    N : int
        Number of subjects to be recruited
    study_duration : float
        Study duration in months
    acc_period : float
        Accrual time in months
    k : float
        Non-uniformity of accrual (1=uniform)
    shape : float
        Weibull shape parameter
    ctrl_median : float
        Control arm median survival time
    HR : float
        Hazard ratio to be detected (NA for single arm)
    alpha : float
        Significance level [0,1]
    power : float
        Power [0,1]
    r : float
        Control:Experimental balance (1:r). 0 for single arm.
    two_sided : bool
        Use two-sided test
    followup : float
        Follow up time (inf if no fixed followup)
    study_type : str
        'Oncology' or 'CRGI'
    dropout : Optional[dict]
        Dropout parameters {'proportion': x, 'time': y, 'shape': z}
    lag_settings : LagEffect
        Lag effect settings
    """
    N: int
    study_duration: float
    acc_period: float
    k: float = 1.0
    shape: float = 1.0
    ctrl_median: float = None
    ctrl_time: float = None
    ctrl_proportion: float = None
    HR: float = None
    alpha: float = None
    power: float = None
    r: float = 1.0
    two_sided: bool = True
    followup: float = np.inf
    study_type: str = 'Oncology'
    dropout: Optional[dict] = None
    lag_settings: LagEffect = field(default_factory=NullLag)
    _ctrl_spec: CtrlSpec = field(init=False, default=None)
    _dropout_specs: List[CtrlSpec] = field(init=False, default_factory=list)
    _dropout_shape: float = field(init=False, default=1.0)
    
    def __post_init__(self):
        self._validate()
        self._setup_ctrl_spec()
        self._setup_dropout()
    
    def _validate(self):
        """Validate study parameters."""
        if self.shape <= 0:
            raise ValueError("Invalid shape")
        if self.acc_period <= 0:
            raise ValueError("Invalid acc_period")
        if self.k <= 0:
            raise ValueError("Invalid k")
        if self.N <= 0 or not isinstance(self.N, int):
            raise ValueError("Invalid N")
        if self.study_duration <= 0:
            raise ValueError("Invalid study_duration")
        if self.study_duration <= self.acc_period:
            raise ValueError("acc_period must be < study_duration")
        if self.followup <= 0:
            raise ValueError("Invalid followup")
        if self.study_type not in ['Oncology', 'CRGI']:
            raise ValueError("Invalid study_type")
        
        if self.r == 0:  # Single arm
            if self.HR is not None and not np.isnan(self.HR):
                raise ValueError("HR must be NA/None for single arm study")
            if self.power is not None and not np.isnan(self.power):
                raise ValueError("power must be NA/None for single arm study")
            if self.alpha is not None and not np.isnan(self.alpha):
                raise ValueError("alpha must be NA/None for single arm study")
        else:
            if self.HR is None or np.isnan(self.HR) or self.HR >= 1 or self.HR <= 0:
                raise ValueError("Invalid HR")
            if self.alpha is None or self.alpha >= 1 or self.alpha <= 0:
                raise ValueError("Invalid alpha")
            if self.power is None or self.power >= 1 or self.power <= 0:
                raise ValueError("Invalid power")
    
    def _setup_ctrl_spec(self):
        """Set up control arm specification."""
        if self.ctrl_median is not None:
            self._ctrl_spec = CtrlSpec.from_median(self.ctrl_median)
        elif self.ctrl_time is not None and self.ctrl_proportion is not None:
            self._ctrl_spec = CtrlSpec.from_proportion(
                self.ctrl_time, self.ctrl_proportion, self.shape
            )
        else:
            raise ValueError("Must specify either ctrl_median or (ctrl_time and ctrl_proportion)")
    
    def _setup_dropout(self):
        """Set up dropout specifications."""
        if self.dropout is None:
            self._dropout_shape = 1.0
            # Create infinite median specs (no dropout)
            spec = CtrlSpec(median=np.inf, text="do not drop out")
            self._dropout_specs = [spec, spec] if self.r != 0 else [spec]
        else:
            prop = self.dropout.get('proportion', 0)
            time = self.dropout.get('time', 1)
            self._dropout_shape = self.dropout.get('shape', 1.0)
            
            if prop <= 0:
                spec = CtrlSpec(median=np.inf, text="do not drop out")
            else:
                # Calculate median from proportion and time
                # At time t, proportion p have dropped out
                # S(t) = 1 - p = exp(-(lambda*t)^shape)
                # lambda*t = (-log(1-p))^(1/shape)
                # median: S(m) = 0.5 => lambda*m = log(2)^(1/shape)
                lambda_val = (-np.log(1 - prop)) ** (1/self._dropout_shape) / time
                median = np.log(2) ** (1/self._dropout_shape) / lambda_val
                spec = CtrlSpec(
                    median=median,
                    text=f"{prop*100:.1f}% would drop out by {time}"
                )
            
            self._dropout_specs = [spec, spec] if self.r != 0 else [spec]
    
    def is_single_arm(self) -> bool:
        """Check if this is a single arm study."""
        return self.r == 0
    
    def get_dropout_lambda(self) -> np.ndarray:
        """Get dropout rate parameters."""
        range_ = [0] if self.is_single_arm() else [0, 1]
        return np.array([
            lambda_calc(self._dropout_specs[i].median, 1, self._dropout_shape)[0]
            for i in range_
        ])
    
    def predict(self, 
                time_pred: Optional[List[float]] = None,
                event_pred: Optional[List[int]] = None,
                step_size: float = 0.5) -> 'AnalysisResults':
        """
        Predict events or times for the study.
        
        Parameters
        ----------
        time_pred : list of float, optional
            Times at which to predict number of events
        event_pred : list of int, optional
            Number of events at which to predict times
        step_size : float
            Resolution of the grid for plotting
            
        Returns
        -------
        AnalysisResults
        """
        from .results import AnalysisResults
        
        self._validate_predict_args(time_pred, event_pred, step_size)
        
        # Times for plotting
        grid_times = np.arange(0, self.study_duration + step_size, step_size)
        
        # Calculate rate parameters
        lagged = self.lag_settings
        lambda_ = lambda_calc(self._ctrl_spec.median, self.HR, self.shape)
        
        if lagged.is_null_lag():
            lambdaot = np.array([np.nan, np.nan])
        else:
            lambdaot = lambda_calc(lagged.ctrl_spec.median, lagged.l_hazard_ratio, self.shape)
        
        dropout_lambda = self.get_dropout_lambda()
        
        # Create survival functions
        sfns = get_survival_functions(
            lambda_, lambdaot, lagged.lag_t, self.is_single_arm(), self.shape,
            self.followup, self._dropout_shape, dropout_lambda
        )
        
        # Calculate event details for plot
        grid = self._calculate_events_given_times(grid_times, sfns, calc_at_risk=False)
        
        # Calculate average HR
        if lagged.is_null_lag() or self.is_single_arm():
            av_hr = self.HR if self.HR is not None else np.nan
        else:
            av_hr = self._calculate_average_hr(
                sfns, lagged.lag_t, lagged.l_hazard_ratio, lambdaot, lambda_
            )
        
        # Calculate critical value
        critical_data = self._calculate_critical_value(sfns, grid, av_hr)
        
        # Calculate events at given times
        if time_pred is not None:
            predict_data = self._calculate_events_given_times(np.array(time_pred), sfns)
        else:
            predict_data = {}
        
        # Calculate times for given number of events
        if event_pred is not None:
            event_data = self._calculate_times_given_events(np.array(event_pred), sfns, grid)
            if predict_data:
                for key in predict_data:
                    predict_data[key] = np.concatenate([predict_data[key], event_data[key]])
            else:
                predict_data = event_data
        
        return AnalysisResults(
            critical_hr=critical_data['chr'],
            critical_data=critical_data['critical_data'],
            critical_events_req=critical_data['critical_events_req'],
            av_hr=av_hr,
            grid=grid,
            predict_data=predict_data,
            study=self,
            sfns=sfns
        )
    
    def _validate_predict_args(self, time_pred, event_pred, step_size):
        """Validate prediction arguments."""
        if time_pred is not None:
            if any(t < 0 or t > self.study_duration for t in time_pred):
                raise ValueError("Invalid time_pred argument")
        
        if event_pred is not None:
            if any(e < 0 or not isinstance(e, int) for e in event_pred):
                raise ValueError("Invalid event_pred argument")
        
        if step_size <= 0 or step_size > self.study_duration:
            raise ValueError("Invalid step_size argument")
    
    def _calculate_events_given_times(self, times: np.ndarray, sfns: list, 
                                      calc_at_risk: bool = True) -> dict:
        """Calculate expected events at given times."""
        Ns = get_ns(self.is_single_arm(), self.r, self.N)
        
        # Recruitment
        m = np.minimum(times, self.acc_period)
        rec_rate = (m ** self.k) / (self.acc_period ** self.k)
        recruit = np.outer(rec_rate, Ns)
        recruit_tot = recruit.sum(axis=1)
        
        # Events
        events_cdf = [events_integ(sfn, self.acc_period, self.k, times) for sfn in sfns]
        events = np.column_stack([events_cdf[0] * Ns[0], events_cdf[1] * Ns[1]])
        events_tot = events.sum(axis=1)
        rounded_events_tot = np.floor(events[:, 0]) + np.floor(events[:, 1])
        
        result = {
            'time': times,
            'events1': events[:, 0],
            'events2': events[:, 1],
            'events_tot': events_tot,
            'recruit_tot': recruit_tot,
            'rounded_events_tot': rounded_events_tot,
            'time_pred': np.ones(len(times), dtype=bool)
        }
        
        if calc_at_risk:
            atrisk_cdf = [atrisk_integ(sfn, self.acc_period, self.k, times) for sfn in sfns]
            atrisk = np.column_stack([atrisk_cdf[0] * Ns[0], atrisk_cdf[1] * Ns[1]])
            atrisk_tot = atrisk.sum(axis=1)
            result.update({
                'at_risk1': atrisk[:, 0],
                'at_risk2': atrisk[:, 1],
                'atrisk_tot': atrisk_tot
            })
        
        return result
    
    def _calculate_times_given_events(self, event_pred: np.ndarray, sfns: list, 
                                       grid: dict) -> dict:
        """Calculate expected times for given events."""
        rounded_events = grid['rounded_events_tot']
        
        ans_times = []
        for ev in event_pred:
            start_idx = np.sum(rounded_events < ev)
            if start_idx >= len(grid['time']) - 1:
                import warnings
                warnings.warn("event_pred argument too large")
                ans_times.append(np.nan)
            else:
                exact_time = self._subdivide(
                    grid['time'][start_idx],
                    grid['time'][start_idx + 1],
                    ev, sfns
                )
                ans_times.append(exact_time)
        
        ans_times = np.array(ans_times)
        valid = ~np.isnan(ans_times)
        
        if not np.any(valid):
            return {'time': np.array([]), 'events_tot': np.array([])}
        
        result = self._calculate_events_given_times(ans_times[valid], sfns)
        result['time_pred'] = np.zeros(len(result['time']), dtype=bool)
        return result
    
    def _subdivide(self, low: float, high: float, aim: int, sfns: list) -> float:
        """Find exact time when target events occur."""
        while not np.isclose(low, high):
            mid = (low + high) / 2
            ans = self._calculate_events_given_times(
                np.array([mid]), sfns, calc_at_risk=False
            )['rounded_events_tot'][0]
            
            if ans >= aim:
                high = mid
            else:
                low = mid
        
        return high
    
    def _calculate_critical_value(self, sfns: list, grid: dict, av_hr: float) -> dict:
        """Calculate critical value for the study."""
        if self.is_single_arm():
            return {
                'chr': np.nan,
                'critical_data': {},
                'critical_events_req': np.nan
            }
        
        alpha = self.alpha / 2 if self.two_sided else self.alpha
        events_req = required_events(self.r, alpha, self.power, av_hr, self.N)
        
        # Critical HR with 50% power
        chr_ = np.exp(-1 * ((self.r + 1) * (norm.ppf(1 - alpha) + norm.ppf(0.5))) / 
                      np.sqrt(self.r * events_req))
        
        critical_data = self._calculate_times_given_events(
            np.array([int(np.round(events_req))]), sfns, grid
        )
        
        return {
            'chr': chr_,
            'critical_data': critical_data,
            'critical_events_req': events_req
        }
    
    def _calculate_average_hr(self, sfns: list, lag_t: float, lag_hr: float,
                              lambdaot: np.ndarray, lambda_: np.ndarray) -> float:
        """Calculate average HR for lagged study."""
        # Create survival functions for lag period only
        sfns_lag = get_survival_functions(
            lambdaot, np.array([0.0, 0.0]), 0, self.is_single_arm(),
            self.shape, lag_t, self._dropout_shape, self.get_dropout_lambda()
        )
        
        Ns = get_ns(self.is_single_arm(), self.r, self.N)
        
        # Events during lag period
        pot = np.array([
            events_integ(sfn, self.acc_period, self.k, np.array([self.study_duration]))[0]
            for sfn in sfns_lag
        ])
        
        # Events after lag period
        if lag_t < self.study_duration:
            ptts = np.array([
                events_integ(sfn, self.acc_period, self.k, np.array([self.study_duration]))[0]
                for sfn in sfns
            ]) - pot
        else:
            ptts = np.array([0.0, 0.0])
        
        w1 = np.sum(pot * Ns)
        w2 = np.sum(ptts * Ns)
        
        return np.exp((w1 * np.log(lag_hr) + w2 * np.log(self.HR)) / (w1 + w2))
    
    def __str__(self) -> str:
        lines = ["Study definition:"]
        lines.append(f"Number of Patients (N): {self.N}")
        lines.append(f"Study duration: {self.study_duration} months")
        lines.append(f"Accrual period: {self.acc_period} months")
        lines.append(f"Accrual uniformity (k): {self.k}")
        
        if not self.is_single_arm():
            lines.append(f"Control arm survival: {self._ctrl_spec.text}")
            lines.append(f"Hazard Ratio: {self.HR}")
            lines.append(f"Ratio of control to experimental 1:{self.r}")
            side = "(two sided)" if self.two_sided else "(one sided)"
            lines.append(f"alpha: {self.alpha} {side}")
            lines.append(f"Power: {self.power}")
        else:
            lines.append(f"Survival: {self._ctrl_spec.text}")
            lines.append("Single Arm trial")
        
        if self.shape == 1:
            lines.append("Exponential survival")
        else:
            lines.append(f"Weibull survival with shape parameter {self.shape}")
        
        if np.isfinite(self.followup):
            lines.append(f"Subject follow up period: {self.followup} months")
        
        lines.append(str(self.lag_settings))
        
        return '\n'.join(lines)


def SingleArmStudy(N: int, study_duration: float, ctrl_median: float, k: float, 
                   acc_period: float, shape: float = 1.0, 
                   dropout: Optional[dict] = None,
                   lag_settings: LagEffect = None) -> Study:
    """Create a single arm study."""
    if lag_settings is None:
        lag_settings = NullLag()
    
    return Study(
        N=N,
        study_duration=study_duration,
        acc_period=acc_period,
        k=k,
        shape=shape,
        ctrl_median=ctrl_median,
        HR=np.nan,
        alpha=np.nan,
        power=np.nan,
        r=0,
        two_sided=False,
        followup=np.inf,
        study_type='Oncology',
        dropout=dropout,
        lag_settings=lag_settings
    )


def CRGIStudy(alpha: float, power: float, HR: float, r: float, N: int,
              study_duration: float, ctrl_time: float, ctrl_proportion: float,
              k: float, acc_period: float, two_sided: bool, 
              shape: float = 1.0, followup: float = np.inf,
              dropout: Optional[dict] = None,
              lag_settings: LagEffect = None) -> Study:
    """Create a CRGI type study."""
    if lag_settings is None:
        lag_settings = NullLag()
    
    return Study(
        N=N,
        study_duration=study_duration,
        acc_period=acc_period,
        k=k,
        shape=shape,
        ctrl_time=ctrl_time,
        ctrl_proportion=ctrl_proportion,
        HR=HR,
        alpha=alpha,
        power=power,
        r=r,
        two_sided=two_sided,
        followup=followup,
        study_type='CRGI',
        dropout=dropout,
        lag_settings=lag_settings
    )


def SingleArmCRGIStudy(N: int, study_duration: float, ctrl_time: float,
                       ctrl_proportion: float, k: float, acc_period: float,
                       shape: float = 1.0, followup: float = np.inf,
                       dropout: Optional[dict] = None,
                       lag_settings: LagEffect = None) -> Study:
    """Create a single arm CRGI type study."""
    if lag_settings is None:
        lag_settings = NullLag()
    
    return Study(
        N=N,
        study_duration=study_duration,
        acc_period=acc_period,
        k=k,
        shape=shape,
        ctrl_time=ctrl_time,
        ctrl_proportion=ctrl_proportion,
        HR=np.nan,
        alpha=np.nan,
        power=np.nan,
        r=0,
        two_sided=False,
        followup=followup,
        study_type='CRGI',
        dropout=dropout,
        lag_settings=lag_settings
    )

