"""
Plotting functions for survival curves and diagnostics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Union
from dataclasses import dataclass


@dataclass
class KaplanMeierResult:
    """Result from Kaplan-Meier estimation."""
    time: np.ndarray
    survival: np.ndarray
    n_risk: np.ndarray
    n_event: np.ndarray
    n_censor: np.ndarray
    ci_lower: Optional[np.ndarray] = None
    ci_upper: Optional[np.ndarray] = None


def kaplan_meier_estimate(times: np.ndarray, 
                          events: np.ndarray,
                          conf_level: float = 0.95) -> KaplanMeierResult:
    """
    Calculate Kaplan-Meier survival estimates.
    
    Parameters
    ----------
    times : np.ndarray
        Time to event or censoring
    events : np.ndarray
        Event indicator (1=event, 0=censored)
    conf_level : float
        Confidence level for confidence intervals
        
    Returns
    -------
    KaplanMeierResult
        Kaplan-Meier survival estimates
    """
    from scipy.stats import norm
    
    # Get unique event times and sort
    unique_times = np.unique(times[events == 1])
    unique_times = np.sort(unique_times)
    
    n_total = len(times)
    
    # Add time 0
    all_times = np.concatenate([[0], unique_times])
    
    survival = np.ones(len(all_times))
    n_risk = np.zeros(len(all_times))
    n_event = np.zeros(len(all_times))
    n_censor = np.zeros(len(all_times))
    var_sum = np.zeros(len(all_times))
    
    n_risk[0] = n_total
    
    for i, t in enumerate(all_times[1:], 1):
        # Number at risk at time t
        n_risk[i] = np.sum(times >= t)
        
        # Number of events at time t
        n_event[i] = np.sum((times == t) & (events == 1))
        
        # Number censored at time t
        n_censor[i] = np.sum((times == t) & (events == 0))
        
        if n_risk[i] > 0:
            # Kaplan-Meier estimate
            survival[i] = survival[i-1] * (1 - n_event[i] / n_risk[i])
            
            # Greenwood's formula for variance
            if n_event[i] > 0:
                var_sum[i] = var_sum[i-1] + n_event[i] / (n_risk[i] * (n_risk[i] - n_event[i]))
            else:
                var_sum[i] = var_sum[i-1]
        else:
            survival[i] = survival[i-1]
            var_sum[i] = var_sum[i-1]
    
    # Calculate confidence intervals using log-log transformation
    z = norm.ppf((1 + conf_level) / 2)
    
    # Avoid log(0) issues
    survival_safe = np.clip(survival, 1e-10, 1 - 1e-10)
    log_log_se = np.sqrt(var_sum) / np.abs(np.log(survival_safe))
    
    ci_lower = survival_safe ** np.exp(z * log_log_se)
    ci_upper = survival_safe ** np.exp(-z * log_log_se)
    
    # Clip to [0, 1]
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
        If provided, overlay the fitted survival curve
    units : str
        Time units ('Days', 'Months', or 'Years')
    show_ci : bool
        Show confidence intervals
    show_censored : bool
        Show tick marks for censored observations
    show_risk_table : bool
        Show number at risk table below the plot
    figsize : tuple
        Figure size
    title : str
        Plot title
    ax : plt.Axes, optional
        Existing axes to plot on
        
    Returns
    -------
    plt.Figure
    """
    from .utils import standarddaysinyear
    
    daysinyear = standarddaysinyear()
    
    # Get scale factor
    if units == 'Days':
        scale = 1
    elif units == 'Months':
        scale = daysinyear / 12
    elif units == 'Years':
        scale = daysinyear
    else:
        raise ValueError("units must be 'Days', 'Months', or 'Years'")
    
    df = event_data.subject_data
    times = df['time'].values / scale
    events = df['has_event'].values
    
    # Calculate Kaplan-Meier
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
    
    # Plot KM curve (step function)
    ax_main.step(km.time, km.survival, where='post', color='#2E86AB', 
                 linewidth=2, label='Kaplan-Meier estimate')
    
    # Plot confidence intervals
    if show_ci:
        ax_main.fill_between(km.time, km.ci_lower, km.ci_upper, 
                             step='post', alpha=0.2, color='#2E86AB',
                             label='95% CI')
    
    # Show censored observations
    if show_censored:
        censor_mask = events == 0
        if censor_mask.any():
            censor_times = times[censor_mask]
            # Get survival probability at censoring times
            censor_surv = []
            for ct in censor_times:
                idx = np.searchsorted(km.time, ct, side='right') - 1
                idx = max(0, min(idx, len(km.survival) - 1))
                censor_surv.append(km.survival[idx])
            
            ax_main.scatter(censor_times, censor_surv, marker='|', 
                           color='#2E86AB', s=50, zorder=5, label='Censored')
    
    # Overlay fitted model if provided
    if event_model is not None:
        t_plot = np.linspace(0, max(times) * 1.1, 200)
        
        # Calculate fitted survival
        if event_model.dist == 'weibull':
            # S(t) = exp(-(lambda*t)^shape)
            surv_fitted = np.exp(-(event_model.rate * t_plot * scale) ** event_model.shape)
        else:  # loglogistic
            # S(t) = 1 / (1 + (lambda*t)^shape)
            surv_fitted = 1 / (1 + (event_model.rate * t_plot * scale) ** event_model.shape)
        
        ax_main.plot(t_plot, surv_fitted, color='#E94F37', linewidth=2, 
                     linestyle='--', label=f'Fitted {event_model.dist}')
    
    # Styling
    ax_main.set_ylim(0, 1.05)
    ax_main.set_xlim(0, None)
    ax_main.set_ylabel('Survival Probability', fontsize=12)
    ax_main.set_title(title, fontsize=14, fontweight='bold')
    ax_main.legend(loc='upper right', framealpha=0.9)
    ax_main.grid(True, alpha=0.3)
    ax_main.spines['top'].set_visible(False)
    ax_main.spines['right'].set_visible(False)
    
    if not show_risk_table:
        ax_main.set_xlabel(f'Time ({units})', fontsize=12)
    
    # Add risk table
    if show_risk_table and ax_table is not None:
        # Choose time points for risk table
        max_time = km.time[-1]
        n_points = min(6, len(km.time))
        risk_times = np.linspace(0, max_time, n_points)
        
        risk_counts = []
        for rt in risk_times:
            idx = np.searchsorted(km.time, rt, side='right') - 1
            idx = max(0, min(idx, len(km.n_risk) - 1))
            # Recalculate as subjects with time >= rt
            risk_counts.append(np.sum(times >= rt))
        
        ax_table.set_xlim(ax_main.get_xlim())
        ax_table.set_ylim(0, 1)
        ax_table.axis('off')
        
        # Add "At risk" label
        ax_table.text(-0.05, 0.5, 'At risk:', transform=ax_table.transAxes,
                      fontsize=10, fontweight='bold', ha='right', va='center')
        
        for i, (t, n) in enumerate(zip(risk_times, risk_counts)):
            ax_table.text(t, 0.5, str(int(n)), ha='center', va='center', fontsize=10)
        
        ax_table.set_xlabel(f'Time ({units})', fontsize=12)
    
    plt.tight_layout()
    return fig


def plot_weibull_diagnostic(event_data: 'EventData',
                            figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
    """
    Plot Weibull diagnostic (log-log plot).
    
    If the data follows a Weibull distribution, points should lie on a straight line.
    
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
    
    # Calculate KM
    km = kaplan_meier_estimate(times, events)
    
    # Filter for valid points (0 < S < 1)
    valid = (km.survival > 0) & (km.survival < 1) & (km.time > 0)
    
    x = np.log(km.time[valid])
    y = np.log(-np.log(km.survival[valid]))
    
    # Fit linear regression
    slope, intercept = np.polyfit(x, y, 1)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.scatter(x, y, color='#2E86AB', s=50, alpha=0.7)
    
    # Plot fitted line
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, color='#E94F37', linewidth=2, 
            label=f'Fitted line (shape={slope:.2f})')
    
    ax.set_xlabel('log(t)', fontsize=12)
    ax.set_ylabel('log(-log(S(t)))', fontsize=12)
    ax.set_title('Weibull Diagnostic Plot', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_events_vs_time(event_data: 'EventData',
                        time_unit: str = 'month',
                        figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot bar chart of events over time.
    
    Parameters
    ----------
    event_data : EventData
        Event data object
    time_unit : str
        Time unit for grouping ('month', 'quarter', 'week')
    figsize : tuple
        Figure size
        
    Returns
    -------
    plt.Figure
    """
    df = event_data.subject_data.copy()
    
    # Calculate last date for each subject
    df['last_date'] = df['rand_date'] + pd.to_timedelta(df['time'] - 1, unit='D')
    
    # Filter to events only
    events = df[df['has_event'] == 1].copy()
    
    if len(events) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No events to display', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
        return fig
    
    # Group by time period
    if time_unit == 'month':
        events['period'] = events['last_date'].dt.to_period('M')
    elif time_unit == 'quarter':
        events['period'] = events['last_date'].dt.to_period('Q')
    elif time_unit == 'week':
        events['period'] = events['last_date'].dt.to_period('W')
    else:
        raise ValueError("time_unit must be 'month', 'quarter', or 'week'")
    
    event_counts = events.groupby('period').size()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = range(len(event_counts))
    ax.bar(x, event_counts.values, color='#2E86AB', alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels([str(p) for p in event_counts.index], rotation=45, ha='right')
    ax.set_ylabel('Number of Events', fontsize=12)
    ax.set_xlabel('Time Period', fontsize=12)
    ax.set_title('Events Over Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig

