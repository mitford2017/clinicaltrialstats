"""
Scenario Analyzer - Multi-scenario analysis and reverse solve capabilities.

Provides:
- Reverse solve: Given a date, what HR to expect?
- Multi-scenario comparison: Bull/Base/Bear cases
- Competitor-based scenario analysis
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, List, Dict
from datetime import date, timedelta
from scipy.stats import norm
from scipy.optimize import brentq


@dataclass
class ScenarioResult:
    """Results for a single scenario."""
    scenario_name: str
    hazard_ratio: float
    median_date: date
    q05_date: date
    q95_date: date
    probability_of_success: float
    events_at_analysis: int
    critical_hr: float
    description: str


@dataclass
class ReverseSolveResult:
    """Result of reverse solving for HR given a date."""
    target_date: date
    expected_events: int
    events_range: tuple[int, int]  # 90% CI
    implied_hr: float
    critical_hr: float
    probability_of_crossing: float
    maturity: float  # % of final events
    interpretation: str


@dataclass
class MultiScenarioAnalysis:
    """Complete multi-scenario analysis."""
    trial_name: str
    scenarios: List[ScenarioResult]
    base_case: ScenarioResult
    bull_case: ScenarioResult
    bear_case: ScenarioResult
    competitor_scenarios: List[ScenarioResult]
    summary_table: pd.DataFrame


class ScenarioAnalyzer:
    """
    Analyzes clinical trials under multiple scenarios.
    """

    def __init__(self):
        pass

    def reverse_solve(
        self,
        target_date: date,
        start_date: date,
        sample_size: int,
        enroll_duration: float,
        ctrl_median: float,
        final_events: int,
        alpha: float = 0.025,
        allocation_ratio: float = 1.0,
    ) -> ReverseSolveResult:
        """
        Given a target date, solve for expected HR and probability of success.

        This answers: "If the trial reads out on date X, what should we expect?"
        """
        # Time in months from start
        days = (target_date - start_date).days
        if days <= 0:
            raise ValueError("Target date must be after start date")

        t_months = days / 30.44

        # Calculate expected events at target date
        events = self._expected_events_at_time(
            t_months, sample_size, enroll_duration, ctrl_median, hr=0.75
        )

        # Events range (approximate 90% CI)
        events_std = np.sqrt(events * 0.3)  # Rough approximation
        events_lower = max(1, int(events - 1.645 * events_std))
        events_upper = int(events + 1.645 * events_std)

        # Calculate critical HR at this event count
        r = allocation_ratio
        z = norm.ppf(1 - alpha / 2)  # Two-sided
        critical_hr = np.exp(-z * (r + 1) / np.sqrt(r * events))

        # Maturity (% of final events)
        maturity = events / final_events

        # For implied HR, we need to solve backwards
        # This is tricky - we'll use the expected event rate
        implied_hr = self._solve_hr_for_events(
            events, t_months, sample_size, enroll_duration, ctrl_median
        )

        # Probability of crossing at this maturity (if true HR = implied HR)
        prob_crossing = self._probability_of_crossing(
            implied_hr, events, critical_hr
        )

        # Interpretation
        if maturity >= 1.0:
            interp = "Full maturity reached - final analysis likely"
        elif maturity >= 0.66:
            interp = "High maturity - could support interim analysis"
        elif maturity >= 0.5:
            interp = "Moderate maturity - early interim possible"
        else:
            interp = "Low maturity - limited power for meaningful analysis"

        if prob_crossing >= 0.8:
            interp += f". Strong chance of positive result if HR={implied_hr:.2f}"
        elif prob_crossing >= 0.5:
            interp += f". Moderate chance of success"
        else:
            interp += f". Challenging to achieve significance"

        return ReverseSolveResult(
            target_date=target_date,
            expected_events=int(events),
            events_range=(events_lower, events_upper),
            implied_hr=implied_hr,
            critical_hr=critical_hr,
            probability_of_crossing=prob_crossing,
            maturity=maturity,
            interpretation=interp,
        )

    def _expected_events_at_time(
        self,
        t: float,
        sample_size: int,
        enroll_duration: float,
        ctrl_median: float,
        hr: float,
    ) -> float:
        """Calculate expected events at calendar time t."""
        ctrl_rate = np.log(2) / ctrl_median
        avg_rate = ctrl_rate * (1 + hr) / 2

        enrolled = min(t * (sample_size / enroll_duration), sample_size)

        if t <= enroll_duration:
            avg_followup = t / 2
        else:
            min_followup = t - enroll_duration
            max_followup = t
            avg_followup = (min_followup + max_followup) / 2

        if avg_followup > 0:
            p_event = 1 - np.exp(-avg_rate * avg_followup)
        else:
            p_event = 0

        return enrolled * p_event

    def _solve_hr_for_events(
        self,
        target_events: int,
        t: float,
        sample_size: int,
        enroll_duration: float,
        ctrl_median: float,
    ) -> float:
        """Solve for HR that produces target events at time t."""
        def objective(hr):
            events = self._expected_events_at_time(
                t, sample_size, enroll_duration, ctrl_median, hr
            )
            return events - target_events

        try:
            hr = brentq(objective, 0.1, 2.0)
            return hr
        except ValueError:
            # If no solution, return default
            return 0.75

    def _probability_of_crossing(
        self,
        true_hr: float,
        events: int,
        critical_hr: float,
    ) -> float:
        """Calculate probability of observing HR < critical_hr given true HR."""
        if events <= 0:
            return 0.0

        log_true = np.log(true_hr)
        log_crit = np.log(critical_hr)
        se = 2 / np.sqrt(events)

        z = (log_crit - log_true) / se
        return norm.cdf(z)

    def multi_scenario_analysis(
        self,
        sample_size: int,
        enroll_duration: float,
        ctrl_median: float,
        start_date: date,
        final_events: int,
        design_hr: float,
        scenarios: Dict[str, float],  # name -> HR
        alpha: float = 0.025,
        n_sim: int = 1000,
    ) -> MultiScenarioAnalysis:
        """
        Run analysis under multiple HR scenarios.

        scenarios: Dict mapping scenario names to hazard ratios
        E.g., {"Bull (VISION-like)": 0.62, "Base": 0.75, "Bear": 0.90}
        """
        try:
            from ..analysis.event_prediction import predict_analysis_dates
        except ImportError:
            from analysis.event_prediction import predict_analysis_dates

        results = []

        for name, hr in scenarios.items():
            pred = predict_analysis_dates(
                sample_size=sample_size,
                enroll_duration=enroll_duration,
                ctrl_median=ctrl_median,
                scenario_hr=hr,
                design_hr=design_hr,
                start_date=start_date,
                final_events=final_events,
                n_sim=n_sim,
            )

            # Calculate probability of success
            critical_hr = pred['final']['critical_hr']
            prob_success = self._probability_of_crossing(hr, final_events, critical_hr)

            result = ScenarioResult(
                scenario_name=name,
                hazard_ratio=hr,
                median_date=pred['final']['median_date'],
                q05_date=pred['final']['q05_date'],
                q95_date=pred['final']['q95_date'],
                probability_of_success=prob_success,
                events_at_analysis=final_events,
                critical_hr=critical_hr,
                description=self._describe_scenario(name, hr, prob_success),
            )
            results.append(result)

        # Identify base/bull/bear
        sorted_results = sorted(results, key=lambda x: x.hazard_ratio)
        bull = sorted_results[0] if sorted_results else results[0]
        bear = sorted_results[-1] if sorted_results else results[-1]
        base = next((r for r in results if "base" in r.scenario_name.lower()), results[len(results)//2])

        # Build summary table
        table_data = []
        for r in results:
            table_data.append({
                "Scenario": r.scenario_name,
                "HR": r.hazard_ratio,
                "Median Date": r.median_date.strftime("%Y-%m-%d"),
                "P(Success)": f"{r.probability_of_success:.0%}",
                "Critical HR": f"{r.critical_hr:.3f}",
            })

        return MultiScenarioAnalysis(
            trial_name="",
            scenarios=results,
            base_case=base,
            bull_case=bull,
            bear_case=bear,
            competitor_scenarios=[],
            summary_table=pd.DataFrame(table_data),
        )

    def _describe_scenario(self, name: str, hr: float, prob: float) -> str:
        """Generate description for a scenario."""
        if prob >= 0.9:
            outcome = "Very likely to succeed"
        elif prob >= 0.7:
            outcome = "Good chance of success"
        elif prob >= 0.5:
            outcome = "Moderate chance of success"
        elif prob >= 0.3:
            outcome = "Challenging, but possible"
        else:
            outcome = "Unlikely to meet primary endpoint"

        return f"{name}: HR={hr:.2f} → {outcome} ({prob:.0%})"

    def competitor_scenario(
        self,
        competitor_name: str,
        competitor_hr: float,
        sample_size: int,
        enroll_duration: float,
        ctrl_median: float,
        start_date: date,
        final_events: int,
        alpha: float = 0.025,
        n_sim: int = 500,
    ) -> ScenarioResult:
        """
        Analyze: "What if this trial performs like [competitor]?"
        """
        try:
            from ..analysis.event_prediction import predict_analysis_dates
        except ImportError:
            from analysis.event_prediction import predict_analysis_dates

        pred = predict_analysis_dates(
            sample_size=sample_size,
            enroll_duration=enroll_duration,
            ctrl_median=ctrl_median,
            scenario_hr=competitor_hr,
            design_hr=competitor_hr,
            start_date=start_date,
            final_events=final_events,
            n_sim=n_sim,
        )

        critical_hr = pred['final']['critical_hr']
        prob_success = self._probability_of_crossing(competitor_hr, final_events, critical_hr)

        return ScenarioResult(
            scenario_name=f"Like {competitor_name}",
            hazard_ratio=competitor_hr,
            median_date=pred['final']['median_date'],
            q05_date=pred['final']['q05_date'],
            q95_date=pred['final']['q95_date'],
            probability_of_success=prob_success,
            events_at_analysis=final_events,
            critical_hr=critical_hr,
            description=f"If performance matches {competitor_name} (HR={competitor_hr:.2f})",
        )

    def events_to_date(
        self,
        target_events: int,
        sample_size: int,
        enroll_duration: float,
        ctrl_median: float,
        start_date: date,
        hr: float = 0.75,
    ) -> date:
        """
        Calculate when a target number of events will be reached.
        """
        def objective(t):
            events = self._expected_events_at_time(
                t, sample_size, enroll_duration, ctrl_median, hr
            )
            return events - target_events

        try:
            t_months = brentq(objective, 1, 200)
            days = int(t_months * 30.44)
            return start_date + timedelta(days=days)
        except ValueError:
            # If can't solve, return far future date
            return start_date + timedelta(days=3650)

    def critical_hr_at_events(
        self,
        events: int,
        alpha: float = 0.025,
        allocation_ratio: float = 1.0,
    ) -> float:
        """Calculate the critical HR threshold at a given event count."""
        r = allocation_ratio
        z = norm.ppf(1 - alpha / 2)
        return np.exp(-z * (r + 1) / np.sqrt(r * events))
