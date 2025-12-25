"""
Plotting functions for survival analysis and event prediction.

This module provides:
- Kaplan-Meier survival curves
- Event rate / cumulative event curves  
- Weibull diagnostic plots
- Prediction result visualizations

Example Usage:
    from eventprediction import EventData_from_dataframe, plot_survival_curve
    from eventprediction import plot_event_curve, plot_cumulative_events
    
    event_data = EventData_from_dataframe(...)
    
    # Survival curve
    fig = plot_survival_curve(event_data)
    
    # Event rate curve
    fig = plot_event_curve(event_data)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Union, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from .event_data import EventData
    from .event_model import EventModel
    from .results import AnalysisResults

# =============================================================================
# Color Palette (consistent across all plots)
# =============================================================================

COLORS = {
    'primary': '#2E86AB',      # Blue - main data
    'secondary': '#E94F37',    # Red - fitted models
    'tertiary': '#A23B72',     # Purple - secondary data
    'accent': '#F18F01',       # Orange - highlights
    'neutral': '#5C5C5C',      # Gray - grid/annotations
    'ci_fill': '#2E86AB',      # Same as primary for CI fill
}


# =============================================================================
# Utility Functions
# =============================================================================

def _get_time_scale(units: str) -> float:
    """Get scale factor for time units."""
    from .utils import standarddaysinyear
    daysinyear = standarddaysinyear()
    
    scales = {
        'Days': 1,
        'Months': daysinyear / 12,
        'Years': daysinyear
    }
    
    if units not in scales:
        raise ValueError(f"units must be one of {list(scales.keys())}")
    
    return scales[units]


def _style_axis(ax: plt.Axes, title: str = '', xlabel: str = '', ylabel: str = ''):
    """Apply consistent styling to an axis."""
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# =============================================================================
# Kaplan-Meier Estimation
# =============================================================================

@dataclass
class KaplanMeierResult:
    """
    Result from Kaplan-Meier estimation.
    
    Attributes
    ----------
    time : np.ndarray
        Event times
    survival : np.ndarray
        Survival probability at each time
    n_risk : np.ndarray
        Number at risk at each time
    n_event : np.ndarray
        Number of events at each time
    n_censor : np.ndarray
        Number censored at each time
    ci_lower : np.ndarray
        Lower confidence interval
    ci_upper : np.ndarray
        Upper confidence interval
    """
    time: np.ndarray
    survival: np.ndarray
    n_risk: np.ndarray
    n_event: np.ndarray
    n_censor: np.ndarray
    ci_lower: Optional[np.ndarray] = None
    ci_upper: Optional[np.ndarray] = None
    
    @property
    def cumulative_events(self) -> np.ndarray:
        """Cumulative number of events."""
        return np.cumsum(self.n_event)
    
    @property
    def event_rate(self) -> np.ndarray:
        """Instantaneous event rate (events per time unit at each point)."""
        # Rate = events / at_risk at each time point
        with np.errstate(divide='ignore', invalid='ignore'):
            rate = np.where(self.n_risk > 0, self.n_event / self.n_risk, 0)
        return rate


def kaplan_meier_estimate(times: np.ndarray, 
                          events: np.ndarray,
                          conf_level: float = 0.95) -> KaplanMeierResult:
    """
    Calculate Kaplan-Meier survival estimates.
    
    Parameters
    ----------
    times : np.ndarray
        Time to event or censoring for each subject
    events : np.ndarray
        Event indicator (1=event, 0=censored)
    conf_level : float
        Confidence level for intervals (default 0.95)
        
    Returns
    -------
    KaplanMeierResult
        Kaplan-Meier estimates with confidence intervals
        
    Example
    -------
    >>> times = np.array([5, 10, 15, 20, 25])
    >>> events = np.array([1, 0, 1, 1, 0])
    >>> km = kaplan_meier_estimate(times, events)
    >>> print(km.survival)
    """
    from scipy.stats import norm
    
    # Get unique event times
    unique_times = np.unique(times[events == 1])
    unique_times = np.sort(unique_times)
    
    n_total = len(times)
    all_times = np.concatenate([[0], unique_times])
    n_times = len(all_times)
    
    # Initialize arrays
    survival = np.ones(n_times)
    n_risk = np.zeros(n_times)
    n_event = np.zeros(n_times)
    n_censor = np.zeros(n_times)
    var_sum = np.zeros(n_times)
    
    n_risk[0] = n_total
    
    # Calculate KM estimates at each event time
    for i, t in enumerate(all_times[1:], 1):
        n_risk[i] = np.sum(times >= t)
        n_event[i] = np.sum((times == t) & (events == 1))
        n_censor[i] = np.sum((times == t) & (events == 0))
        
        if n_risk[i] > 0:
            survival[i] = survival[i-1] * (1 - n_event[i] / n_risk[i])
            
            # Greenwood's variance formula
            if n_event[i] > 0 and n_risk[i] > n_event[i]:
                var_sum[i] = var_sum[i-1] + n_event[i] / (n_risk[i] * (n_risk[i] - n_event[i]))
            else:
                var_sum[i] = var_sum[i-1]
        else:
            survival[i] = survival[i-1]
            var_sum[i] = var_sum[i-1]
    
    # Calculate confidence intervals (log-log transformation)
    z = norm.ppf((1 + conf_level) / 2)
    survival_safe = np.clip(survival, 1e-10, 1 - 1e-10)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        log_log_se = np.sqrt(var_sum) / np.abs(np.log(survival_safe))
        ci_lower = survival_safe ** np.exp(z * log_log_se)
        ci_upper = survival_safe ** np.exp(-z * log_log_se)
    
    ci_lower = np.clip(ci_lower, 0, 1)
    ci_upper = np.clip(ci_upper, 0, 1)
    
    return KaplanMeierResult(
        time=all_times,
        survival=survival,
        n_risk=n_risk.astype(int),
        n_event=n_event.astype(int),
        n_censor=n_censor.astype(int),
        ci_lower=ci_lower,
        ci_upper=ci_upper
    )


# =============================================================================
# Survival Curve Plotting
# =============================================================================

def plot_survival_curve(event_data: 'EventData',
                        event_model: Optional['EventModel'] = None,
                        units: str = 'Days',
                        show_ci: bool = True,
                        show_censored: bool = True,
                        show_risk_table: bool = True,
                        figsize: Tuple[int, int] = (10, 8),
                        title: str = 'Kaplan-Meier Survival Curve',
                        ax: Optional[plt.Axes] = None) -> plt.Figure:
    """
    Plot Kaplan-Meier survival curve.
    
    Parameters
    ----------
    event_data : EventData
        Event data object
    event_model : EventModel, optional
        Fitted model to overlay on the curve
    units : str
        Time units: 'Days', 'Months', or 'Years'
    show_ci : bool
        Show 95% confidence intervals
    show_censored : bool
        Show censoring tick marks
    show_risk_table : bool
        Show number at risk table
    figsize : tuple
        Figure size (width, height)
    title : str
        Plot title
    ax : plt.Axes, optional
        Existing axes to use
        
    Returns
    -------
    plt.Figure
    
    Example
    -------
    >>> from eventprediction import EventData_from_dataframe, plot_survival_curve
    >>> event_data = EventData_from_dataframe(df, ...)
    >>> fig = plot_survival_curve(event_data, units='Months')
    >>> plt.show()
    """
    scale = _get_time_scale(units)
    
    df = event_data.subject_data
    times = df['time'].values / scale
    events = df['has_event'].values
    
    km = kaplan_meier_estimate(times, events)
    
    # Create figure
    if ax is None:
        if show_risk_table:
            fig, (ax_main, ax_table) = plt.subplots(
                2, 1, figsize=figsize, 
                gridspec_kw={'height_ratios': [4, 1]},
                sharex=True
            )
            fig.subplots_adjust(hspace=0.05)
        else:
            fig, ax_main = plt.subplots(figsize=figsize)
            ax_table = None
    else:
        ax_main = ax
        fig = ax.figure
        ax_table = None
    
    # Plot KM step curve
    ax_main.step(km.time, km.survival, where='post', 
                 color=COLORS['primary'], linewidth=2, 
                 label='Kaplan-Meier estimate')
    
    # Confidence intervals
    if show_ci:
        ax_main.fill_between(km.time, km.ci_lower, km.ci_upper, 
                             step='post', alpha=0.2, color=COLORS['ci_fill'],
                             label='95% CI')
    
    # Censoring marks
    if show_censored:
        censor_mask = events == 0
        if censor_mask.any():
            censor_times = times[censor_mask]
            censor_surv = [km.survival[max(0, np.searchsorted(km.time, ct, side='right') - 1)] 
                           for ct in censor_times]
            ax_main.scatter(censor_times, censor_surv, marker='|', 
                           color=COLORS['primary'], s=50, zorder=5, label='Censored')
    
    # Overlay fitted model
    if event_model is not None:
        _overlay_fitted_survival(ax_main, event_model, times, scale, units)
    
    # Styling
    ax_main.set_ylim(0, 1.05)
    ax_main.set_xlim(0, None)
    ax_main.legend(loc='upper right', framealpha=0.9)
    _style_axis(ax_main, title=title, ylabel='Survival Probability')
    
    if not show_risk_table:
        ax_main.set_xlabel(f'Time ({units})')
    
    # Risk table
    if show_risk_table and ax_table is not None:
        _add_risk_table(ax_table, km, times, units)
    
    plt.tight_layout()
    return fig


def _overlay_fitted_survival(ax: plt.Axes, model: 'EventModel', 
                              times: np.ndarray, scale: float, units: str):
    """Overlay fitted survival curve on KM plot."""
    t_plot = np.linspace(0, max(times) * 1.1, 200)
    
    if model.dist == 'weibull':
        surv_fitted = np.exp(-(model.rate * t_plot * scale) ** model.shape)
    else:  # loglogistic
        surv_fitted = 1 / (1 + (model.rate * t_plot * scale) ** model.shape)
    
    ax.plot(t_plot, surv_fitted, color=COLORS['secondary'], linewidth=2, 
            linestyle='--', label=f'Fitted {model.dist}')


def _add_risk_table(ax: plt.Axes, km: KaplanMeierResult, 
                    times: np.ndarray, units: str):
    """Add number at risk table below survival plot."""
    max_time = km.time[-1]
    n_points = min(6, len(km.time))
    risk_times = np.linspace(0, max_time, n_points)
    
    risk_counts = [int(np.sum(times >= rt)) for rt in risk_times]
    
    ax.set_xlim(ax.get_xlim())
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    ax.text(-0.05, 0.5, 'At risk:', transform=ax.transAxes,
            fontsize=10, fontweight='bold', ha='right', va='center')
    
    for t, n in zip(risk_times, risk_counts):
        ax.text(t, 0.5, str(n), ha='center', va='center', fontsize=10)
    
    ax.set_xlabel(f'Time ({units})', fontsize=12)


# =============================================================================
# Event Rate / Cumulative Event Curves
# =============================================================================

def plot_event_curve(event_data: 'EventData',
                     curve_type: str = 'cumulative',
                     units: str = 'Days',
                     show_ci: bool = True,
                     figsize: Tuple[int, int] = (10, 6),
                     title: Optional[str] = None,
                     ax: Optional[plt.Axes] = None) -> plt.Figure:
    """
    Plot event curves over time.
    
    Parameters
    ----------
    event_data : EventData
        Event data object
    curve_type : str
        Type of curve:
        - 'cumulative': Cumulative events (1 - S(t))
        - 'hazard': Cumulative hazard (-log(S(t)))
        - 'incidence': Cumulative incidence rate
    units : str
        Time units: 'Days', 'Months', or 'Years'
    show_ci : bool
        Show confidence intervals
    figsize : tuple
        Figure size
    title : str, optional
        Plot title (auto-generated if None)
    ax : plt.Axes, optional
        Existing axes
        
    Returns
    -------
    plt.Figure
    
    Example
    -------
    >>> fig = plot_event_curve(event_data, curve_type='cumulative', units='Months')
    """
    scale = _get_time_scale(units)
    
    df = event_data.subject_data
    times = df['time'].values / scale
    events = df['has_event'].values
    
    km = kaplan_meier_estimate(times, events)
    
    # Calculate y-values based on curve type
    if curve_type == 'cumulative':
        y_values = 1 - km.survival
        y_lower = 1 - km.ci_upper if km.ci_upper is not None else None
        y_upper = 1 - km.ci_lower if km.ci_lower is not None else None
        ylabel = 'Cumulative Event Probability'
        default_title = 'Cumulative Event Curve'
    elif curve_type == 'hazard':
        with np.errstate(divide='ignore'):
            y_values = -np.log(np.clip(km.survival, 1e-10, 1))
            y_lower = -np.log(np.clip(km.ci_upper, 1e-10, 1)) if km.ci_upper is not None else None
            y_upper = -np.log(np.clip(km.ci_lower, 1e-10, 1)) if km.ci_lower is not None else None
        ylabel = 'Cumulative Hazard'
        default_title = 'Cumulative Hazard Curve'
    elif curve_type == 'incidence':
        y_values = km.cumulative_events / len(times)
        y_lower = y_upper = None  # No CI for this
        ylabel = 'Cumulative Incidence Rate'
        default_title = 'Cumulative Incidence Curve'
    else:
        raise ValueError("curve_type must be 'cumulative', 'hazard', or 'incidence'")
    
    # Create plot
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    ax.step(km.time, y_values, where='post', color=COLORS['primary'], 
            linewidth=2, label='Observed')
    
    if show_ci and y_lower is not None and y_upper is not None:
        ax.fill_between(km.time, y_lower, y_upper, step='post', 
                        alpha=0.2, color=COLORS['ci_fill'], label='95% CI')
    
    _style_axis(ax, title=title or default_title, 
                xlabel=f'Time ({units})', ylabel=ylabel)
    ax.set_xlim(0, None)
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    return fig


def plot_cumulative_events(event_data: 'EventData',
                           units: str = 'Days',
                           show_recruitment: bool = True,
                           figsize: Tuple[int, int] = (10, 6),
                           title: str = 'Cumulative Events Over Time') -> plt.Figure:
    """
    Plot cumulative number of events over calendar time.
    
    Shows actual count of events (not probability) on the y-axis.
    
    Parameters
    ----------
    event_data : EventData
        Event data object
    units : str
        Time units for x-axis
    show_recruitment : bool
        Also show cumulative recruitment
    figsize : tuple
        Figure size
    title : str
        Plot title
        
    Returns
    -------
    plt.Figure
    
    Example
    -------
    >>> fig = plot_cumulative_events(event_data, show_recruitment=True)
    """
    scale = _get_time_scale(units)
    df = event_data.subject_data.copy()
    
    # Calculate event date for subjects with events
    df['last_date'] = df['rand_date'] + pd.to_timedelta(df['time'] - 1, unit='D')
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot cumulative events
    events_df = df[df['has_event'] == 1].sort_values('last_date')
    if len(events_df) > 0:
        ax.step(events_df['last_date'], range(1, len(events_df) + 1),
                where='post', color=COLORS['primary'], linewidth=2,
                label=f'Events (n={len(events_df)})')
    
    # Plot cumulative recruitment
    if show_recruitment:
        rec_df = df.sort_values('rand_date')
        ax.step(rec_df['rand_date'], range(1, len(rec_df) + 1),
                where='post', color=COLORS['neutral'], linewidth=2,
                linestyle='--', label=f'Recruitment (n={len(rec_df)})')
    
    _style_axis(ax, title=title, xlabel='Date', ylabel='Cumulative Count')
    ax.legend(loc='upper left')
    
    # Format x-axis for dates
    fig.autofmt_xdate()
    
    plt.tight_layout()
    return fig


def plot_event_rate(event_data: 'EventData',
                    time_unit: str = 'month',
                    rate_per: int = 100,
                    figsize: Tuple[int, int] = (10, 6),
                    title: Optional[str] = None) -> plt.Figure:
    """
    Plot event rate over time periods.
    
    Shows events per N person-time at risk in each period.
    
    Parameters
    ----------
    event_data : EventData
        Event data object
    time_unit : str
        Grouping period: 'month', 'quarter', 'week'
    rate_per : int
        Calculate rate per this many person-days (default 100)
    figsize : tuple
        Figure size
    title : str, optional
        Plot title
        
    Returns
    -------
    plt.Figure
    
    Example
    -------
    >>> fig = plot_event_rate(event_data, time_unit='month', rate_per=100)
    """
    df = event_data.subject_data.copy()
    df['last_date'] = df['rand_date'] + pd.to_timedelta(df['time'] - 1, unit='D')
    
    # Determine period for each event
    period_map = {'month': 'M', 'quarter': 'Q', 'week': 'W'}
    if time_unit not in period_map:
        raise ValueError(f"time_unit must be one of {list(period_map.keys())}")
    
    events_df = df[df['has_event'] == 1].copy()
    events_df['period'] = events_df['last_date'].dt.to_period(period_map[time_unit])
    
    # Count events per period
    event_counts = events_df.groupby('period').size()
    
    # Calculate person-time at risk per period (simplified)
    # This is approximate - proper calculation would track exact at-risk time
    all_periods = pd.period_range(
        df['rand_date'].min(), df['last_date'].max(), freq=period_map[time_unit]
    )
    
    rates = []
    periods = []
    for period in all_periods:
        # Subjects at risk during this period
        period_start = period.start_time
        period_end = period.end_time
        
        at_risk = ((df['rand_date'] <= period_end) & 
                   (df['last_date'] >= period_start))
        n_at_risk = at_risk.sum()
        
        n_events = event_counts.get(period, 0)
        
        if n_at_risk > 0:
            # Approximate person-days at risk
            person_days = n_at_risk * (period_end - period_start).days
            rate = (n_events / person_days) * rate_per if person_days > 0 else 0
        else:
            rate = 0
        
        rates.append(rate)
        periods.append(period)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = range(len(periods))
    ax.bar(x, rates, color=COLORS['primary'], alpha=0.8)
    
    ax.set_xticks(x[::max(1, len(x)//10)])  # Show ~10 labels
    ax.set_xticklabels([str(p) for p in periods[::max(1, len(x)//10)]], 
                        rotation=45, ha='right')
    
    default_title = f'Event Rate per {rate_per} Person-Days'
    _style_axis(ax, title=title or default_title,
                xlabel=f'Time ({time_unit.title()})',
                ylabel=f'Events per {rate_per} Person-Days')
    
    plt.tight_layout()
    return fig


# =============================================================================
# Diagnostic Plots
# =============================================================================

def plot_weibull_diagnostic(event_data: 'EventData',
                            figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
    """
    Plot Weibull diagnostic (log-log survival plot).
    
    If data follows Weibull distribution, points should form a straight line.
    The slope estimates the shape parameter.
    
    Parameters
    ----------
    event_data : EventData
        Event data object
    figsize : tuple
        Figure size
        
    Returns
    -------
    plt.Figure
    """
    df = event_data.subject_data
    times = df['time'].values
    events = df['has_event'].values
    
    km = kaplan_meier_estimate(times, events)
    
    valid = (km.survival > 0) & (km.survival < 1) & (km.time > 0)
    if valid.sum() < 2:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'Insufficient data for diagnostic plot',
                ha='center', va='center', transform=ax.transAxes)
        return fig
    
    x = np.log(km.time[valid])
    y = np.log(-np.log(km.survival[valid]))
    
    slope, intercept = np.polyfit(x, y, 1)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.scatter(x, y, color=COLORS['primary'], s=50, alpha=0.7, label='Data')
    
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, color=COLORS['secondary'], 
            linewidth=2, label=f'Fitted (shape={slope:.2f})')
    
    _style_axis(ax, title='Weibull Diagnostic Plot',
                xlabel='log(time)', ylabel='log(-log(S(t)))')
    ax.legend()
    
    plt.tight_layout()
    return fig


def plot_prediction_curve(results: 'AnalysisResults',
                          show_recruitment: bool = True,
                          show_required: bool = True,
                          figsize: Tuple[int, int] = (10, 6),
                          title: str = 'Event Prediction from Parameters') -> plt.Figure:
    """
    Plot event prediction from Study.predict() results.
    
    Shows expected events and recruitment over time.
    
    Parameters
    ----------
    results : AnalysisResults
        Results from Study.predict()
    show_recruitment : bool
        Show recruitment curve
    show_required : bool
        Show required events line
    figsize : tuple
        Figure size
    title : str
        Plot title
        
    Returns
    -------
    plt.Figure
    
    Example
    -------
    >>> from eventprediction import Study, plot_prediction_curve
    >>> study = Study(...)
    >>> results = study.predict(event_pred=[200])
    >>> fig = plot_prediction_curve(results)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    times = results.grid['time']
    events = results.grid['events_tot']
    recruitment = results.grid['recruit_tot']
    
    # Events curve
    ax.plot(times, events, color=COLORS['primary'], linewidth=2, 
            label='Expected Events')
    
    # Recruitment curve
    if show_recruitment:
        ax.plot(times, recruitment, color=COLORS['neutral'], linewidth=2,
                linestyle='--', label='Recruitment')
    
    # Required events line
    if show_required and not np.isnan(results.critical_events_req):
        ax.axhline(y=results.critical_events_req, color=COLORS['secondary'], 
                   linestyle=':', linewidth=2,
                   label=f'Required Events ({results.critical_events_req:.0f})')
        
        # Mark intersection
        if 'time' in results.critical_data and len(results.critical_data['time']) > 0:
            crit_time = results.critical_data['time'][0]
            ax.axvline(x=crit_time, color=COLORS['accent'], linestyle=':',
                       alpha=0.7)
            ax.plot(crit_time, results.critical_events_req, 'o', 
                    color=COLORS['accent'], markersize=8)
    
    _style_axis(ax, title=title, xlabel='Time (months)', 
                ylabel='Number of Subjects/Events')
    ax.legend(loc='upper left')
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    
    plt.tight_layout()
    return fig


def plot_events_vs_time(event_data: 'EventData',
                        time_unit: str = 'month',
                        figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot bar chart of event counts by time period.
    
    Parameters
    ----------
    event_data : EventData
        Event data object
    time_unit : str
        Grouping: 'month', 'quarter', 'week'
    figsize : tuple
        Figure size
        
    Returns
    -------
    plt.Figure
    """
    df = event_data.subject_data.copy()
    df['last_date'] = df['rand_date'] + pd.to_timedelta(df['time'] - 1, unit='D')
    
    events = df[df['has_event'] == 1].copy()
    
    if len(events) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No events to display', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
        return fig
    
    period_map = {'month': 'M', 'quarter': 'Q', 'week': 'W'}
    if time_unit not in period_map:
        raise ValueError(f"time_unit must be one of {list(period_map.keys())}")
    
    events['period'] = events['last_date'].dt.to_period(period_map[time_unit])
    event_counts = events.groupby('period').size()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = range(len(event_counts))
    ax.bar(x, event_counts.values, color=COLORS['primary'], alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels([str(p) for p in event_counts.index], rotation=45, ha='right')
    
    _style_axis(ax, title='Events Over Time',
                xlabel='Time Period', ylabel='Number of Events')
    
    plt.tight_layout()
    return fig
