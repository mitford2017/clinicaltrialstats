"""
Clinical Trial Predictor

A Python tool for predicting Phase 3 clinical trial outcomes using
simulation, group sequential designs, and AI-powered benchmark analysis.

Inspired by the Merck simtrial R package.
"""

from .predictor import TrialPredictor, quick_predict, TrialOutcomePrediction
from .api.clinicaltrials import ClinicalTrialsAPI, StudyData
from .analysis.benchmark import BenchmarkAnalyzer
from .analysis.group_sequential import GroupSequentialDesign

__version__ = "1.0.0"
__all__ = [
    "TrialPredictor",
    "quick_predict",
    "TrialOutcomePrediction",
    "ClinicalTrialsAPI",
    "StudyData",
    "BenchmarkAnalyzer",
    "GroupSequentialDesign",
]
