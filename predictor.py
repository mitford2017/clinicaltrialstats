"""
Clinical Trial Prediction Engine.
Orchestrates all components to predict trial outcomes and timing.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

try:
    from .api.clinicaltrials import ClinicalTrialsAPI, StudyData
    from .analysis.benchmark import BenchmarkAnalyzer, TrialPrediction
    from .analysis.group_sequential import (
        GroupSequentialDesign,
        obrien_fleming_spending,
        calculate_sample_size,
    )
    from .analysis.event_prediction import predict_analysis_dates, calc_probability_of_success
    from .simulation.distributions import PiecewiseExponential
    from .simulation.survival import (
        EnrollmentPattern,
        TrialArm,
        TrialConfig,
        simulate_trial,
        simulate_trials,
    )
except ImportError:
    from api.clinicaltrials import ClinicalTrialsAPI, StudyData
    from analysis.benchmark import BenchmarkAnalyzer, TrialPrediction
    from analysis.group_sequential import (
        GroupSequentialDesign,
        obrien_fleming_spending,
        calculate_sample_size,
    )
    from analysis.event_prediction import predict_analysis_dates, calc_probability_of_success
    from simulation.distributions import PiecewiseExponential
    from simulation.survival import (
        EnrollmentPattern,
        TrialArm,
        TrialConfig,
        simulate_trial,
        simulate_trials,
    )


@dataclass
class TimelineEvent:
    """A predicted event in the trial timeline."""
    event_type: str  # "interim_analysis", "final_analysis", "enrollment_complete"
    predicted_date: datetime
    calendar_months: float
    events_at_time: int
    information_fraction: float
    critical_value: float
    probability_of_crossing: float


@dataclass
class HazardRatioPrediction:
    """Predicted hazard ratio at a specific time point."""
    calendar_date: datetime
    calendar_months: float
    events: int
    expected_hr: float
    hr_95_ci: tuple[float, float]
    probability_of_success: float
    z_statistic: float
    critical_value: float


@dataclass
class TrialOutcomePrediction:
    """Complete prediction for a clinical trial."""
    study: StudyData
    benchmark_analysis: TrialPrediction
    design: GroupSequentialDesign
    timeline: list[TimelineEvent]
    hr_trajectory: list[HazardRatioPrediction]
    overall_probability_of_success: float
    expected_final_date: datetime
    expected_hr_at_final: float
    simulation_summary: dict


class TrialPredictor:
    """
    Main prediction engine for clinical trial outcomes.
    """

    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        n_simulations: int = 1000,
    ):
        """
        Initialize the trial predictor.

        Args:
            gemini_api_key: API key for Gemini (benchmark analysis)
            n_simulations: Number of simulations to run
        """
        self.ct_api = ClinicalTrialsAPI()
        self.benchmark_analyzer = BenchmarkAnalyzer(api_key=gemini_api_key)
        self.n_simulations = n_simulations

    def predict(
        self,
        nct_id: str,
        target_date: Optional[datetime] = None,
        custom_hr: Optional[float] = None,
        custom_control_median: Optional[float] = None,
    ) -> TrialOutcomePrediction:
        """
        Generate predictions for a clinical trial.

        Args:
            nct_id: ClinicalTrials.gov NCT ID
            target_date: Optional specific date to predict HR at
            custom_hr: Optional custom hazard ratio to use
            custom_control_median: Optional custom control median to use

        Returns:
            TrialOutcomePrediction with comprehensive predictions
        """
        # Fetch study data
        study = self.ct_api.get_study(nct_id)

        # Get benchmark analysis
        benchmark = self.benchmark_analyzer.analyze_trial(study)

        # Override with custom values if provided
        hazard_ratio = custom_hr or benchmark.recommended_hr
        control_median = custom_control_median or benchmark.control_median

        # Create group sequential design
        design = GroupSequentialDesign.create(
            total_events=benchmark.expected_events,
            information_fractions=benchmark.interim_analyses,
            alpha=benchmark.recommended_alpha,
            power=benchmark.recommended_power,
            spending_function=obrien_fleming_spending,
        )

        # Calculate timeline
        timeline = self._calculate_timeline(study, benchmark, design, hazard_ratio)

        # Calculate HR trajectory
        hr_trajectory = self._calculate_hr_trajectory(
            study, benchmark, design, hazard_ratio, control_median, target_date
        )

        # Run simulations
        sim_summary = self._run_simulations(
            study, benchmark, design, hazard_ratio, control_median
        )

        # Calculate overall probability of success
        pos = self._calculate_probability_of_success(
            design, hazard_ratio, sim_summary
        )

        # Expected final analysis date from model prediction
        final_event = next(
            (e for e in timeline if e.event_type == "final_analysis"),
            None
        )
        expected_final_date = final_event.predicted_date if final_event else datetime.now()

        # Expected HR at final
        final_hr = hr_trajectory[-1] if hr_trajectory else None
        expected_hr_at_final = final_hr.expected_hr if final_hr else hazard_ratio

        return TrialOutcomePrediction(
            study=study,
            benchmark_analysis=benchmark,
            design=design,
            timeline=timeline,
            hr_trajectory=hr_trajectory,
            overall_probability_of_success=pos,
            expected_final_date=expected_final_date,
            expected_hr_at_final=expected_hr_at_final,
            simulation_summary=sim_summary,
        )

    def _calculate_timeline(
        self,
        study: StudyData,
        benchmark: TrialPrediction,
        design: GroupSequentialDesign,
        hazard_ratio: float,
    ) -> list[TimelineEvent]:
        """Calculate predicted timeline of trial events."""
        events = []

        # Get trial start date
        start_date = study.dates.start_date or datetime.now()

        # Estimate enrollment parameters
        enrollment_duration = self._estimate_enrollment_duration(study, benchmark)
        enrollment_rate = study.design.enrollment / enrollment_duration

        for analysis in design.analyses:
            # Calculate time to reach target events
            time_to_events = design.time_to_events(
                target_events=analysis.target_events,
                enrollment_rate=enrollment_rate,
                enrollment_duration=enrollment_duration,
                median_survival=benchmark.control_median,
                hazard_ratio=hazard_ratio,
            )

            predicted_date = start_date + timedelta(days=time_to_events * 30.44)

            # Calculate probability of crossing boundary at this analysis
            # Using normal approximation
            expected_z = -np.log(hazard_ratio) * np.sqrt(analysis.target_events) / 2
            prob_crossing = 1 - stats.norm.cdf(analysis.critical_value - expected_z)

            event_type = "final_analysis" if analysis.information_fraction == 1.0 else "interim_analysis"

            events.append(TimelineEvent(
                event_type=event_type,
                predicted_date=predicted_date,
                calendar_months=time_to_events,
                events_at_time=analysis.target_events,
                information_fraction=analysis.information_fraction,
                critical_value=analysis.critical_value,
                probability_of_crossing=prob_crossing,
            ))

        # Add enrollment complete event (only if it occurs before final analysis)
        final_analysis_time = max(e.calendar_months for e in events) if events else 0
        if enrollment_duration < final_analysis_time:
            enrollment_complete = TimelineEvent(
                event_type="enrollment_complete",
                predicted_date=start_date + timedelta(days=enrollment_duration * 30.44),
                calendar_months=enrollment_duration,
                events_at_time=0,
                information_fraction=0.0,
                critical_value=0.0,
                probability_of_crossing=0.0,
            )
            events.append(enrollment_complete)

        return sorted(events, key=lambda x: x.calendar_months)

    def _calculate_hr_trajectory(
        self,
        study: StudyData,
        benchmark: TrialPrediction,
        design: GroupSequentialDesign,
        hazard_ratio: float,
        control_median: float,
        target_date: Optional[datetime],
    ) -> list[HazardRatioPrediction]:
        """Calculate predicted HR at various time points."""
        trajectory = []

        start_date = study.dates.start_date or datetime.now()
        enrollment_duration = self._estimate_enrollment_duration(study, benchmark)
        enrollment_rate = study.design.enrollment / enrollment_duration

        # Generate time points
        max_time = design.time_to_events(
            design.total_events,
            enrollment_rate,
            enrollment_duration,
            control_median,
            hazard_ratio,
        )

        time_points = np.linspace(enrollment_duration, max_time, 12)

        # Add target date if provided
        if target_date:
            target_months = (target_date - start_date).days / 30.44
            if target_months > 0:
                time_points = np.append(time_points, target_months)
                time_points = np.sort(np.unique(time_points))

        # Add primary completion date if available
        if study.dates.primary_completion_date:
            pcd_months = (study.dates.primary_completion_date - start_date).days / 30.44
            if pcd_months > 0:
                time_points = np.append(time_points, pcd_months)
                time_points = np.sort(np.unique(time_points))

        for t in time_points:
            events = int(design.expected_events_at_time(
                t, enrollment_rate, enrollment_duration, control_median, hazard_ratio
            ))

            if events < 10:
                continue

            # Find critical value for current information fraction
            info_frac = events / design.total_events
            critical_value = self._interpolate_critical_value(design, info_frac)

            # Expected Z-statistic
            expected_z = -np.log(hazard_ratio) * np.sqrt(events) / 2

            # Probability of success at this time
            prob_success = 1 - stats.norm.cdf(critical_value - expected_z)

            # HR confidence interval (assuming true HR)
            se_log_hr = 2 / np.sqrt(events)
            hr_lower = hazard_ratio * np.exp(-1.96 * se_log_hr)
            hr_upper = hazard_ratio * np.exp(1.96 * se_log_hr)

            trajectory.append(HazardRatioPrediction(
                calendar_date=start_date + timedelta(days=t * 30.44),
                calendar_months=t,
                events=events,
                expected_hr=hazard_ratio,
                hr_95_ci=(hr_lower, hr_upper),
                probability_of_success=prob_success,
                z_statistic=expected_z,
                critical_value=critical_value,
            ))

        return trajectory

    def _interpolate_critical_value(
        self,
        design: GroupSequentialDesign,
        info_frac: float,
    ) -> float:
        """Interpolate critical value for an information fraction."""
        # Find bracketing analyses
        for i, analysis in enumerate(design.analyses):
            if analysis.information_fraction >= info_frac:
                if i == 0:
                    return analysis.critical_value
                prev = design.analyses[i - 1]
                # Linear interpolation
                frac = (info_frac - prev.information_fraction) / (
                    analysis.information_fraction - prev.information_fraction
                )
                return prev.critical_value + frac * (
                    analysis.critical_value - prev.critical_value
                )

        return design.analyses[-1].critical_value

    def _run_simulations(
        self,
        study: StudyData,
        benchmark: TrialPrediction,
        design: GroupSequentialDesign,
        hazard_ratio: float,
        control_median: float,
    ) -> dict:
        """Run Monte Carlo simulations to estimate trial outcomes."""
        enrollment_duration = self._estimate_enrollment_duration(study, benchmark)

        # Create survival distributions
        control_dist = PiecewiseExponential.from_median(control_median)

        # Treatment with potential delayed effect for immunotherapy
        is_immunotherapy = self._is_immunotherapy(study)
        if is_immunotherapy:
            delay = 2.0  # 2-month delay for immunotherapy
            control_dist, treatment_dist = PiecewiseExponential.with_delayed_effect(
                control_median, hazard_ratio, delay
            )
        else:
            treatment_dist = PiecewiseExponential.from_median(
                control_median / hazard_ratio
            )

        # Create trial config
        dropout_dist = PiecewiseExponential(
            durations=[np.inf],
            hazard_rates=[0.02]  # ~2% annual dropout
        )

        config = TrialConfig(
            enrollment=EnrollmentPattern.ramp_up(
                ramp_duration=3.0,
                steady_duration=enrollment_duration - 3.0,
                total_n=study.design.enrollment,
            ),
            arms=[
                TrialArm("Control", 1.0, control_dist, dropout_dist),
                TrialArm("Treatment", 1.0, treatment_dist, dropout_dist),
            ],
            total_n=study.design.enrollment,
            seed=42,
        )

        # Run simulations
        trials = simulate_trials(config, self.n_simulations, seed=42)

        # Analyze outcomes
        successes = 0
        hr_estimates = []
        final_times = []

        for trial in trials:
            cut_data, cut_time = trial.cut_at_events(design.total_events)
            final_times.append(cut_time)

            # Simple HR estimate
            try:
                from .analysis.statistical_tests import weighted_logrank
            except ImportError:
                from analysis.statistical_tests import weighted_logrank
            try:
                result = weighted_logrank(cut_data)
                hr_estimates.append(result.hazard_ratio)

                # Check if trial would be positive
                if result.z_statistic >= design.analyses[-1].critical_value:
                    successes += 1
            except Exception:
                hr_estimates.append(1.0)

        return {
            "n_simulations": self.n_simulations,
            "success_rate": successes / self.n_simulations,
            "mean_hr": np.mean(hr_estimates),
            "median_hr": np.median(hr_estimates),
            "hr_std": np.std(hr_estimates),
            "hr_percentiles": {
                "5%": np.percentile(hr_estimates, 5),
                "25%": np.percentile(hr_estimates, 25),
                "50%": np.percentile(hr_estimates, 50),
                "75%": np.percentile(hr_estimates, 75),
                "95%": np.percentile(hr_estimates, 95),
            },
            "mean_final_time": np.mean(final_times),
            "final_time_std": np.std(final_times),
        }

    def _calculate_probability_of_success(
        self,
        design: GroupSequentialDesign,
        hazard_ratio: float,
        sim_summary: dict,
    ) -> float:
        """Calculate overall probability of trial success."""
        # Use simulation results as primary estimate
        if sim_summary and "success_rate" in sim_summary:
            return sim_summary["success_rate"]

        # Fallback to analytical calculation
        return design.power_at_hr(hazard_ratio)

    def _estimate_enrollment_duration(
        self,
        study: StudyData,
        benchmark: TrialPrediction,
    ) -> float:
        """Estimate enrollment duration in months."""
        if study.dates.start_date and study.dates.primary_completion_date:
            # Use actual dates if available
            total_duration = (
                study.dates.primary_completion_date - study.dates.start_date
            ).days / 30.44

            # For survival trials, enrollment is typically 50-70% of total duration
            # The remaining time is for follow-up to accumulate events
            # Use 60% as a reasonable estimate
            return total_duration * 0.6

        # Default estimate based on sample size
        # Assume 30-50 patients/month for large Phase 3 oncology trials
        enrollment_rate = 40
        return study.design.enrollment / enrollment_rate

    def _get_expected_final_analysis_date(
        self,
        study: StudyData,
        calculated_date: datetime,
    ) -> datetime:
        """
        Get the expected final analysis date, preferring registry data.

        If the registry has a primary completion date, use it as a sanity check.
        """
        if study.dates.primary_completion_date:
            registry_date = study.dates.primary_completion_date

            # If our calculated date is significantly earlier than registry,
            # the registry date is more reliable (they have actual enrollment data)
            if calculated_date < registry_date - timedelta(days=180):
                return registry_date

        return calculated_date

    def _is_immunotherapy(self, study: StudyData) -> bool:
        """Check if the trial involves immunotherapy."""
        immunotherapy_keywords = [
            "immunotherapy", "checkpoint", "pd-1", "pd-l1", "ctla-4",
            "pembrolizumab", "nivolumab", "atezolizumab", "durvalumab",
            "ipilimumab", "avelumab", "cemiplimab", "car-t", "car t",
        ]

        text_to_search = (
            study.brief_summary.lower() +
            study.detailed_description.lower() +
            " ".join(arm.description.lower() for arm in study.arms) +
            " ".join(
                " ".join(arm.interventions).lower() for arm in study.arms
            )
        )

        return any(keyword in text_to_search for keyword in immunotherapy_keywords)


def quick_predict(nct_id: str, gemini_api_key: Optional[str] = None) -> TrialOutcomePrediction:
    """
    Quick helper function to generate predictions for a trial.

    Args:
        nct_id: ClinicalTrials.gov NCT ID
        gemini_api_key: Optional Gemini API key

    Returns:
        TrialOutcomePrediction
    """
    predictor = TrialPredictor(
        gemini_api_key=gemini_api_key,
        n_simulations=500,  # Fewer sims for quick analysis
    )
    return predictor.predict(nct_id)
