"""
Event Prediction in Clinical Trials with Time-to-Event Outcomes

This Python package implements methods to predict either the required number of events 
to achieve a target or the expected time at which you will reach the required number of events.
You can use this package in the design phase of clinical trials or in the reporting phase.

Based on the R package 'eventPrediction' by Daniel Dalevi, Nik Burkoff, and contributors.
"""

__version__ = "1.0.0"
__author__ = "Python port of eventPrediction R package"

from .study import Study, SingleArmStudy, CRGIStudy, SingleArmCRGIStudy
from .event_data import EventData, EmptyEventData, CutData, EventData_from_dataframe
from .event_model import EventModel, FromDataSimParam
from .accrual import AccrualGenerator, PoissonAccrual, PowerLawAccrual, estimate_accrual_parameter
from .simulation import simulate
from .survival_functions import Sfn
from .lag_effect import LagEffect, NullLag, create_lag_effect
from .display_options import DisplayOptions
from .results import AnalysisResults, FromDataResults
from .utils import standarddaysinyear, csv_sniffer, fit_mixture_model
from .plotting import (
    plot_survival_curve, plot_weibull_diagnostic, plot_events_vs_time,
    kaplan_meier_estimate, KaplanMeierResult
)
from .clinicaltrials import (
    fetch_trial_info, create_study_from_nct, search_trials, TrialInfo
)

__all__ = [
    # Study classes
    'Study', 'SingleArmStudy', 'CRGIStudy', 'SingleArmCRGIStudy',
    # Data classes  
    'EventData', 'EventData_from_dataframe', 'EmptyEventData', 'CutData', 
    'EventModel', 'FromDataSimParam',
    # Accrual
    'AccrualGenerator', 'PoissonAccrual', 'PowerLawAccrual', 'estimate_accrual_parameter',
    # Simulation
    'simulate',
    # Survival functions
    'Sfn',
    # Lag effects
    'LagEffect', 'NullLag', 'create_lag_effect',
    # Display
    'DisplayOptions',
    # Results
    'AnalysisResults', 'FromDataResults',
    # Utilities
    'standarddaysinyear', 'csv_sniffer', 'fit_mixture_model',
]

