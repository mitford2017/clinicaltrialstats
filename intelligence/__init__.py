"""
Clinical Trial Intelligence Module.

Provides research-driven analysis combining:
- Benchmark research (competitor trials, SOC data)
- Protocol design extraction
- Multi-scenario analysis
- Reverse solve capabilities
"""

from .benchmark_research import (
    BenchmarkResearchAgent,
    BenchmarkLandscape,
    CompetitorTrial,
    ProtocolDesign,
)
from .scenario_analyzer import (
    ScenarioAnalyzer,
    ScenarioResult,
    ReverseSolveResult,
    MultiScenarioAnalysis,
)

__all__ = [
    "BenchmarkResearchAgent",
    "BenchmarkLandscape",
    "CompetitorTrial",
    "ProtocolDesign",
    "ScenarioAnalyzer",
    "ScenarioResult",
    "ReverseSolveResult",
    "MultiScenarioAnalysis",
]
