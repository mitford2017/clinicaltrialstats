"""
Results classes for event prediction.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date

from .utils import standarddaysinyear, add_line_breaks


@dataclass
class SimQOutput:
    """
    Quantile output from simulations.
    
    Attributes
    ----------
    median : np.ndarray
        Median values (dates or counts)
    lower : np.ndarray
        Lower quantile values
    upper : np.ndarray
        Upper quantile values
    limit : float
        Quantile limit (e.g., 0.05 for 5%-95%)
    """
    median: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    limit: float = 0.05
    
    @classmethod
    def from_matrix(cls, matrix: np.ndarray, limit: float, n_sim: int) -> 'SimQOutput':
        """
        Create SimQOutput from simulation matrix.
        
        Parameters
        ----------
        matrix : np.ndarray
            Matrix with shape (n_sim, n_items)
        limit : float
            Quantile limit
        n_sim : int
            Number of simulations
            
        Returns
        -------
        SimQOutput
        """
        # Sort each column
        sorted_matrix = np.sort(matrix, axis=0)
        
        # Calculate quantile indices
        lower_idx = int(np.floor(limit * n_sim)) - 1
        upper_idx = int(np.ceil((1 - limit) * n_sim)) - 1
        median_idx = int(np.floor(0.5 * n_sim)) - 1
        
        lower_idx = max(0, lower_idx)
        upper_idx = min(n_sim - 1, upper_idx)
        median_idx = max(0, min(n_sim - 1, median_idx))
        
        return cls(
            median=sorted_matrix[median_idx, :],
            lower=sorted_matrix[lower_idx, :],
            upper=sorted_matrix[upper_idx, :],
            limit=limit
        )


@dataclass
class SingleSimDetails:
    """
    Details from individual simulations.
    
    Attributes
    ----------
    event_type : np.ndarray
        Event type for each subject in each simulation
    event_times : np.ndarray
        Event times for each subject in each simulation
    rec_times : np.ndarray
        Recruitment times for each subject in each simulation
    """
    event_type: np.ndarray
    event_times: np.ndarray
    rec_times: np.ndarray


@dataclass
class AnalysisResults:
    """
    Results from predict function of Study object.
    
    Attributes
    ----------
    critical_hr : float
        Critical hazard ratio
    critical_data : dict
        Data for critical number of events
    critical_events_req : float
        Number of events required for target power
    av_hr : float
        Average hazard ratio
    grid : dict
        Event and recruitment details at grid times
    predict_data : dict
        Predicted values at requested times/events
    study : Study
        The study object used
    sfns : list
        Survival function objects
    """
    critical_hr: float
    critical_data: dict
    critical_events_req: float
    av_hr: float
    grid: dict
    predict_data: dict
    study: 'Study'
    sfns: list
    
    def __str__(self) -> str:
        lines = [str(self.study)]
        
        if not np.isnan(self.av_hr):
            lines.append(f"Average HR: {self.av_hr:.2f}")
        
        if self.predict_data and len(self.predict_data.get('time', [])) > 0:
            lines.append("\nPredicted Values:")
            df = pd.DataFrame(self.predict_data)
            lines.append(df.to_string())
        
        if self.critical_data and len(self.critical_data.get('time', [])) > 0:
            lines.append("\nCritical Number of Events:")
            df = pd.DataFrame(self.critical_data)
            lines.append(df.to_string())
        
        if not np.isnan(self.critical_hr):
            lines.append(f"Critical HR: {self.critical_hr:.2f}")
        
        return '\n'.join(lines)
    
    def plot(self, figsize: tuple = (10, 6), **kwargs) -> plt.Figure:
        """
        Plot the prediction results.
        
        Parameters
        ----------
        figsize : tuple
            Figure size
        **kwargs
            Additional matplotlib arguments
            
        Returns
        -------
        matplotlib.Figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        times = self.grid['time']
        events_tot = self.grid['events_tot']
        recruit_tot = self.grid['recruit_tot']
        
        ax.plot(times, events_tot, 'b-', linewidth=2, label='Expected Events')
        ax.plot(times, recruit_tot, 'k-', linewidth=2, label='Recruitment')
        
        if not np.isnan(self.critical_events_req):
            ax.axhline(y=self.critical_events_req, color='r', linestyle='--', 
                       label=f'Required Events ({self.critical_events_req:.0f})')
        
        ax.set_xlabel('Time (months)')
        ax.set_ylabel('Number')
        ax.set_title('Event Prediction from Study Parameters')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


@dataclass 
class FromDataResults:
    """
    Results from simulating event predictions from data.
    
    Attributes
    ----------
    limit : float
        Confidence interval width
    event_quantiles : SimQOutput
        Dates for median/CI of each event
    event_data : EventData
        Original event data
    accrual_generator : AccrualGenerator
        Accrual generator used
    n_accrual : int
        Number of additional subjects recruited
    time_pred_data : pd.DataFrame
        Expected events at target dates
    event_pred_data : pd.DataFrame
        Expected dates for target events
    rec_quantiles : SimQOutput
        Dates for recruitment quantiles
    dropout_quantiles : SimQOutput
        Dates for dropout quantiles
    single_sim_details : SingleSimDetails
        Subject-level simulation details
    dropout_shape : float
        Weibull dropout shape
    dropout_rate : float
        Weibull dropout rate
    sim_params : FromDataSimParam
        Simulation parameters
    """
    limit: float
    event_quantiles: SimQOutput
    event_data: 'EventData'
    accrual_generator: 'AccrualGenerator'
    n_accrual: int
    time_pred_data: pd.DataFrame
    event_pred_data: pd.DataFrame
    rec_quantiles: SimQOutput
    dropout_quantiles: SimQOutput
    single_sim_details: SingleSimDetails
    dropout_shape: float
    dropout_rate: float
    sim_params: 'FromDataSimParam'
    
    def predict(self, 
                time_pred: Optional[List[str]] = None,
                event_pred: Optional[List[int]] = None) -> 'FromDataResults':
        """
        Make predictions for given times or events.
        
        Parameters
        ----------
        time_pred : list of str/date, optional
            Dates to predict events at
        event_pred : list of int, optional
            Event counts to predict times for
            
        Returns
        -------
        FromDataResults
            Updated results with predictions
        """
        from .utils import fix_dates
        
        if time_pred is None and event_pred is None:
            raise ValueError("No predictions requested! Provide time_pred or event_pred")
        
        max_events = len(self.event_quantiles.median)
        
        if event_pred is not None:
            if any(e <= 0 or e > max_events for e in event_pred):
                raise ValueError(f"Invalid event_pred: must be positive and <= {max_events}")
        
        result = FromDataResults(
            limit=self.limit,
            event_quantiles=self.event_quantiles,
            event_data=self.event_data,
            accrual_generator=self.accrual_generator,
            n_accrual=self.n_accrual,
            time_pred_data=self.time_pred_data.copy(),
            event_pred_data=self.event_pred_data.copy(),
            rec_quantiles=self.rec_quantiles,
            dropout_quantiles=self.dropout_quantiles,
            single_sim_details=self.single_sim_details,
            dropout_shape=self.dropout_shape,
            dropout_rate=self.dropout_rate,
            sim_params=self.sim_params
        )
        
        if time_pred is not None:
            dates = fix_dates(time_pred)
            new_data = self._predict_given_dates(dates)
            result.time_pred_data = pd.concat([result.time_pred_data, new_data], 
                                               ignore_index=True)
        
        if event_pred is not None:
            new_data = self._predict_given_events(event_pred)
            result.event_pred_data = pd.concat([result.event_pred_data, new_data],
                                                ignore_index=True)
        
        return result
    
    def _predict_given_dates(self, dates: pd.Series) -> pd.DataFrame:
        """Predict events at given dates."""
        # Convert dates to ordinal for comparison
        event_times = self.single_sim_details.event_times
        
        results = []
        for d in dates:
            # Count events before this date in each simulation
            counts = np.sum(event_times <= np.datetime64(d), axis=1)
            median = int(np.median(counts))
            lower = int(np.percentile(counts, self.limit * 100))
            upper = int(np.percentile(counts, (1 - self.limit) * 100))
            
            results.append({
                'time': d,
                'event': median,
                'CI_low': lower,
                'CI_high': upper,
                'daysatrisk': 0  # TODO: Calculate properly
            })
        
        return pd.DataFrame(results)
    
    def _predict_given_events(self, event_counts: List[int]) -> pd.DataFrame:
        """Predict dates for given event counts."""
        results = []
        
        for n in event_counts:
            idx = n - 1  # 0-indexed
            
            if idx < len(self.event_quantiles.median):
                # Convert from days since epoch to datetime
                median = self.event_quantiles.median[idx]
                lower = self.event_quantiles.lower[idx]
                upper = self.event_quantiles.upper[idx]
                
                # Convert numeric days to datetime
                median_date = pd.Timestamp('1970-01-01') + pd.Timedelta(days=int(median))
                lower_date = pd.Timestamp('1970-01-01') + pd.Timedelta(days=int(lower))
                upper_date = pd.Timestamp('1970-01-01') + pd.Timedelta(days=int(upper))
                
                results.append({
                    'time': median_date,
                    'event': n,
                    'CI_low': lower_date,
                    'CI_high': upper_date,
                    'daysatrisk': 0
                })
        
        return pd.DataFrame(results)
    
    def summary(self, round_method: str = 'None', text_width: int = 60,
                show_predictions: bool = True, show_at_risk: bool = True) -> str:
        """Generate summary text."""
        daysinyear = standarddaysinyear()
        
        data = self.event_data
        df = data.subject_data
        n_subjects = len(df)
        
        title = ""
        if n_subjects > 0:
            last_event = df.loc[df['has_event'] == 1, 'rand_date'].max() if data.n_events > 0 else None
            
            title = f"{n_subjects} patients recruited where last patient in on {df['rand_date'].max()}"
            
            if last_event is not None:
                title += f" and last event observed at {last_event} ({data.n_events} events). "
            else:
                title += " and no events observed. "
        
        if self.n_accrual > 0:
            title += (f" Out of {n_subjects + self.n_accrual} patients, {self.n_accrual} "
                      f"were simulated using {self.accrual_generator.text} ")
        
        title += f"Using {self.sim_params.type_} survival model. "
        
        if len(self.event_pred_data) > 0 and show_predictions:
            for _, row in self.event_pred_data.iterrows():
                title += (f"The time at which {int(row['event'])} events have occurred is "
                          f"predicted to be {row['time']} [{row['CI_low']}, {row['CI_high']}]. ")
        
        if len(self.time_pred_data) > 0 and show_predictions:
            for _, row in self.time_pred_data.iterrows():
                title += (f"On {row['time']} the predicted number of events is "
                          f"{int(row['event'])} [{int(row['CI_low'])}, {int(row['CI_high'])}]. ")
        
        return add_line_breaks(title, text_width)
    
    def __str__(self) -> str:
        return self.summary()
    
    def plot(self, figsize: tuple = (12, 8), show_title: bool = False,
             include_dropouts: bool = True, **kwargs) -> plt.Figure:
        """
        Plot the simulation results.
        
        Parameters
        ----------
        figsize : tuple
            Figure size
        show_title : bool
            Show title with summary
        include_dropouts : bool
            Include dropout information
        **kwargs
            Additional arguments
            
        Returns
        -------
        matplotlib.Figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Event data
        df = self.event_data.subject_data
        event_df = df[df['has_event'] == 1].copy()
        if len(event_df) > 0:
            event_df['last_date'] = event_df['rand_date'] + pd.to_timedelta(
                event_df['time'] - 1, unit='D'
            )
            event_df = event_df.sort_values('last_date')
            ax.plot(event_df['last_date'], range(1, len(event_df) + 1),
                    'purple', linewidth=2, label='Observed Events')
        
        # Predicted events
        median_dates = pd.to_datetime(self.event_quantiles.median)
        ax.plot(median_dates, range(1, len(median_dates) + 1),
                'b-', linewidth=2, label='Predicted Events')
        
        lower_dates = pd.to_datetime(self.event_quantiles.lower)
        upper_dates = pd.to_datetime(self.event_quantiles.upper)
        ax.plot(lower_dates, range(1, len(lower_dates) + 1),
                'r--', linewidth=1, label=f'CI [{self.limit:.0%}, {1-self.limit:.0%}]')
        ax.plot(upper_dates, range(1, len(upper_dates) + 1),
                'r--', linewidth=1)
        
        # Recruitment
        rec_median = pd.to_datetime(self.rec_quantiles.median)
        ax.plot(rec_median, range(1, len(rec_median) + 1),
                'k-', linewidth=2, label='Recruitment')
        
        ax.set_xlabel('Date')
        ax.set_ylabel('N')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        if show_title:
            ax.set_title(self.summary(text_width=100), fontsize=9, wrap=True)
        
        plt.tight_layout()
        return fig

