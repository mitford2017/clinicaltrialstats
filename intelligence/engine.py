"""
Clinical Trial Intelligence Engine - Main orchestrator.

This is the primary interface for the intelligence system.
Given an NCT ID, it:
1. Gathers trial data from CT.gov
2. Researches competitive landscape via Gemini
3. Extracts protocol design parameters
4. Runs multi-scenario simulations
5. Provides predictions and reverse solve capabilities
"""

import os
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional, List, Dict

import numpy as np
import pandas as pd

try:
    from ..api.clinicaltrials import ClinicalTrialsAPI, StudyData
    from ..analysis.event_prediction import predict_analysis_dates, calc_probability_of_success
    from ..analysis.improved_calculations import (
        analyze_trial, schoenfeld_events, critical_hr,
        probability_of_success, calculate_trial_events
    )
    from .benchmark_research import BenchmarkResearchAgent, BenchmarkLandscape, ProtocolDesign
    from .scenario_analyzer import ScenarioAnalyzer, MultiScenarioAnalysis, ReverseSolveResult
except ImportError:
    from api.clinicaltrials import ClinicalTrialsAPI, StudyData
    from analysis.event_prediction import predict_analysis_dates, calc_probability_of_success
    from analysis.improved_calculations import (
        analyze_trial, schoenfeld_events, critical_hr,
        probability_of_success, calculate_trial_events
    )
    from intelligence.benchmark_research import BenchmarkResearchAgent, BenchmarkLandscape, ProtocolDesign
    from intelligence.scenario_analyzer import ScenarioAnalyzer, MultiScenarioAnalysis, ReverseSolveResult


@dataclass
class IntelligenceReport:
    """Complete intelligence report for a clinical trial."""
    # Trial basics
    nct_id: str
    trial_name: str
    sponsor: str
    indication: str
    drug_name: str
    comparator: str
    primary_endpoint: str
    enrollment: int
    status: str
    start_date: Optional[date]
    primary_completion_date: Optional[date]

    # Design parameters (from research)
    design_hr: float
    power: float
    alpha: float
    interim_events: int
    final_events: int

    # Competitive landscape
    landscape: Optional[BenchmarkLandscape]
    soc_median: float
    best_in_class_hr: Optional[float]

    # Predictions
    interim_date: date
    interim_date_range: tuple[date, date]
    final_date: date
    final_date_range: tuple[date, date]

    # Multi-scenario analysis
    scenario_analysis: MultiScenarioAnalysis
    probability_of_success: float  # Base case

    # Critical thresholds
    interim_critical_hr: float
    final_critical_hr: float

    # Key insights
    key_insights: List[str]


class TrialIntelligenceEngine:
    """
    Main intelligence engine for clinical trial analysis.

    Usage:
        engine = TrialIntelligenceEngine(gemini_api_key="...")
        report = engine.analyze("NCT06925737")
        print(report.scenario_analysis.summary_table)
    """

    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        n_simulations: int = 1000,
    ):
        self.api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")
        self.n_sim = n_simulations

        self.ct_api = ClinicalTrialsAPI()
        self.benchmark_agent = BenchmarkResearchAgent(api_key=self.api_key)
        self.scenario_analyzer = ScenarioAnalyzer()

    def analyze(
        self,
        nct_id: str,
        custom_scenarios: Optional[Dict[str, float]] = None,
    ) -> IntelligenceReport:
        """
        Generate comprehensive intelligence report for a trial.

        Args:
            nct_id: ClinicalTrials.gov NCT ID
            custom_scenarios: Optional dict of {scenario_name: hazard_ratio}

        Returns:
            IntelligenceReport with complete analysis
        """
        # 1. Fetch trial data
        study = self.ct_api.get_study(nct_id)

        # Extract key info
        indication = ", ".join(study.conditions[:2]) if study.conditions else "Unknown"
        drug_name = self._extract_drug_name(study)
        comparator = self._extract_comparator(study)
        primary_endpoint = study.get_primary_endpoint_type() or "OS"

        # 2. Research competitive landscape
        landscape = self.benchmark_agent.research_competitive_landscape(
            nct_id=nct_id,
            indication=indication,
            drug_name=drug_name,
            line_of_therapy=self._infer_line_of_therapy(study),
            primary_endpoint=primary_endpoint,
        )

        # 3. Extract or infer design parameters
        protocol = self.benchmark_agent.extract_protocol_design(
            nct_id=nct_id,
            trial_title=study.title,
            sponsor=study.sponsor,
        )

        # Use researched values or defaults
        if landscape:
            design_hr = landscape.recommended_design_hr
            soc_median = landscape.soc_median_survival
            best_hr = landscape.best_in_class_hr
        else:
            design_hr = 0.75
            soc_median = self._estimate_control_median(indication, primary_endpoint)
            best_hr = None

        if protocol:
            power = protocol.power
            alpha = protocol.alpha
        else:
            power = 0.90
            alpha = 0.025

        # Estimate enrollment duration
        enroll_duration = self._estimate_enrollment_duration(study)
        start_date_dt = study.dates.start_date or datetime.now()
        start_date_d = start_date_dt.date() if isinstance(start_date_dt, datetime) else start_date_dt

        # Use IMPROVED trial-specific event calculations
        trial_analysis = analyze_trial(
            sample_size=study.design.enrollment,
            control_median=soc_median,
            design_hr=design_hr,
            power=power,
            alpha=alpha,
            enroll_duration=enroll_duration,
            min_followup=12.0,  # Assume 12 month minimum follow-up
        )

        # Use protocol events if available, otherwise use calculated
        if protocol and protocol.final_events:
            final_events = protocol.final_events
            interim_events = protocol.interim_events or int(final_events * 0.66)
        else:
            final_events = trial_analysis.final_events
            interim_events = trial_analysis.interim_events

        # 4. Build scenarios
        scenarios = custom_scenarios or self._build_scenarios(landscape, design_hr)

        # 5. Run multi-scenario analysis
        scenario_analysis = self.scenario_analyzer.multi_scenario_analysis(
            sample_size=study.design.enrollment,
            enroll_duration=enroll_duration,
            ctrl_median=soc_median,
            start_date=start_date_d,
            final_events=final_events,
            design_hr=design_hr,
            scenarios=scenarios,
            alpha=alpha,
            n_sim=self.n_sim,
        )

        # If registry has primary completion date, use it as a sanity check
        # The registry date is typically more accurate for ongoing trials
        registry_final_date = None
        if study.dates.primary_completion_date:
            pcd = study.dates.primary_completion_date
            registry_final_date = pcd.date() if hasattr(pcd, 'date') else pcd

        # 6. Get base case predictions
        base_pred = predict_analysis_dates(
            sample_size=study.design.enrollment,
            enroll_duration=enroll_duration,
            ctrl_median=soc_median,
            scenario_hr=design_hr,
            design_hr=design_hr,
            start_date=start_date_d,
            interim_events=interim_events,
            final_events=final_events,
            n_sim=self.n_sim,
        )

        # Calibrate model predictions using registry dates when available
        # The registry primary completion date is based on sponsor's actual data
        # and is more reliable than pure model predictions
        if registry_final_date and start_date_d:
            registry_months = (registry_final_date - start_date_d).days / 30.44
            model_months = (base_pred['final']['median_date'] - start_date_d).days / 30.44

            # If model is significantly faster than registry, calibrate
            if model_months > 0 and registry_months > model_months * 1.2:
                # Calculate scale factor
                scale = registry_months / model_months

                # Apply scaling to predictions
                from datetime import timedelta

                def scale_date(d, ref_date, factor):
                    if d and ref_date:
                        days_from_start = (d - ref_date).days
                        scaled_days = int(days_from_start * factor)
                        return ref_date + timedelta(days=scaled_days)
                    return d

                base_pred['interim']['median_date'] = scale_date(
                    base_pred['interim']['median_date'], start_date_d, scale
                )
                base_pred['interim']['q05_date'] = scale_date(
                    base_pred['interim']['q05_date'], start_date_d, scale
                )
                base_pred['interim']['q95_date'] = scale_date(
                    base_pred['interim']['q95_date'], start_date_d, scale
                )
                base_pred['final']['median_date'] = scale_date(
                    base_pred['final']['median_date'], start_date_d, scale
                )
                base_pred['final']['q05_date'] = scale_date(
                    base_pred['final']['q05_date'], start_date_d, scale
                )
                base_pred['final']['q95_date'] = scale_date(
                    base_pred['final']['q95_date'], start_date_d, scale
                )

        # 7. Generate insights
        insights = self._generate_insights(
            study, landscape, scenario_analysis, base_pred
        )

        return IntelligenceReport(
            nct_id=nct_id,
            trial_name=study.brief_title,
            sponsor=study.sponsor,
            indication=indication,
            drug_name=drug_name,
            comparator=comparator,
            primary_endpoint=primary_endpoint,
            enrollment=study.design.enrollment,
            status=study.status,
            start_date=start_date_d,
            primary_completion_date=study.dates.primary_completion_date.date() if study.dates.primary_completion_date else None,
            design_hr=design_hr,
            power=power,
            alpha=alpha,
            interim_events=interim_events,
            final_events=final_events,
            landscape=landscape,
            soc_median=soc_median,
            best_in_class_hr=best_hr,
            interim_date=base_pred['interim']['median_date'],
            interim_date_range=(base_pred['interim']['q05_date'], base_pred['interim']['q95_date']),
            final_date=base_pred['final']['median_date'],
            final_date_range=(base_pred['final']['q05_date'], base_pred['final']['q95_date']),
            scenario_analysis=scenario_analysis,
            probability_of_success=scenario_analysis.base_case.probability_of_success,
            interim_critical_hr=base_pred['interim']['critical_hr'],
            final_critical_hr=base_pred['final']['critical_hr'],
            key_insights=insights,
        )

    def reverse_solve(
        self,
        nct_id: str,
        target_date: date,
    ) -> ReverseSolveResult:
        """
        Given a target date, solve for what to expect.

        Args:
            nct_id: Trial NCT ID
            target_date: Date to analyze

        Returns:
            ReverseSolveResult with expected HR, events, probability
        """
        study = self.ct_api.get_study(nct_id)

        start_date_dt = study.dates.start_date or datetime.now()
        start_date_d = start_date_dt.date() if isinstance(start_date_dt, datetime) else start_date_dt

        enroll_duration = self._estimate_enrollment_duration(study)
        ctrl_median = 14.0  # Default, could be researched

        final_events = self._calc_events(0.75, 0.90, 0.025)

        return self.scenario_analyzer.reverse_solve(
            target_date=target_date,
            start_date=start_date_d,
            sample_size=study.design.enrollment,
            enroll_duration=enroll_duration,
            ctrl_median=ctrl_median,
            final_events=final_events,
        )

    def what_if(
        self,
        nct_id: str,
        competitor_name: str,
        competitor_hr: float,
    ):
        """
        Analyze: "What if this trial performs like [competitor]?"

        Args:
            nct_id: Trial NCT ID
            competitor_name: Name of competitor trial
            competitor_hr: HR achieved by competitor
        """
        study = self.ct_api.get_study(nct_id)

        start_date_dt = study.dates.start_date or datetime.now()
        start_date_d = start_date_dt.date() if isinstance(start_date_dt, datetime) else start_date_dt

        enroll_duration = self._estimate_enrollment_duration(study)
        final_events = self._calc_events(0.75, 0.90, 0.025)

        return self.scenario_analyzer.competitor_scenario(
            competitor_name=competitor_name,
            competitor_hr=competitor_hr,
            sample_size=study.design.enrollment,
            enroll_duration=enroll_duration,
            ctrl_median=14.0,
            start_date=start_date_d,
            final_events=final_events,
            n_sim=self.n_sim,
        )

    def _extract_drug_name(self, study: StudyData) -> str:
        """Extract experimental drug name from study."""
        for arm in study.arms:
            arm_type_lower = arm.arm_type.lower() if arm.arm_type else ""
            if "experimental" in arm_type_lower:
                if arm.interventions:
                    return arm.interventions[0]
                return arm.label
        return "Unknown"

    def _extract_comparator(self, study: StudyData) -> str:
        """Extract comparator from study."""
        for arm in study.arms:
            arm_type_lower = arm.arm_type.lower() if arm.arm_type else ""
            if "comparator" in arm_type_lower or "placebo" in arm_type_lower:
                if arm.interventions:
                    return arm.interventions[0]
                return arm.label
        return "Standard of Care"

    def _infer_line_of_therapy(self, study: StudyData) -> str:
        """Infer line of therapy from eligibility criteria."""
        criteria = study.eligibility_criteria.lower()
        if "first-line" in criteria or "first line" in criteria or "1l" in criteria:
            return "1L"
        elif "second-line" in criteria or "second line" in criteria or "2l" in criteria:
            return "2L"
        elif "third" in criteria or "3l" in criteria:
            return "3L+"
        elif "previously treated" in criteria:
            return "2L+"
        return "Unknown"

    def _estimate_control_median(self, indication: str, endpoint: str) -> float:
        """
        Estimate control arm median survival based on indication.

        Returns median in months. This is a key parameter that
        significantly affects event prediction.
        """
        indication_lower = indication.lower()

        # Prostate cancer benchmarks (2024 data)
        if "prostate" in indication_lower:
            if "castration-resistant" in indication_lower or "crpc" in indication_lower:
                # mCRPC varies by line of therapy
                return 14.0  # Post-taxane: ~12mo, Pre-taxane: ~16mo, average ~14mo
            else:
                return 20.0  # mHSPC: ~20-24 months

        # Lung cancer
        if "lung" in indication_lower or "nsclc" in indication_lower:
            if "small cell" in indication_lower:
                return 10.0  # SCLC
            else:
                return 12.0  # NSCLC 1L: 12-15mo

        # Breast cancer
        if "breast" in indication_lower:
            if "triple" in indication_lower and "negative" in indication_lower:
                return 12.0  # TNBC
            else:
                return 18.0  # HR+ metastatic

        # GI cancers
        if "pancrea" in indication_lower:
            return 8.0  # Pancreatic
        if "colorectal" in indication_lower or "colon" in indication_lower:
            return 18.0
        if "gastric" in indication_lower or "stomach" in indication_lower:
            return 12.0
        if "hepatocellular" in indication_lower or "liver" in indication_lower or "hcc" in indication_lower:
            return 13.0

        # Other solid tumors
        if "melanoma" in indication_lower:
            return 15.0
        if "renal" in indication_lower or "kidney" in indication_lower:
            return 20.0
        if "bladder" in indication_lower or "urothelial" in indication_lower:
            return 12.0
        if "ovarian" in indication_lower:
            return 14.0

        # Default for unknown indications
        return 14.0

    def _estimate_enrollment_duration(self, study: StudyData) -> float:
        """
        Estimate enrollment duration in months.

        Uses registry dates when available, otherwise estimates based on
        realistic enrollment rates for Phase 3 oncology trials.
        """
        if study.dates.start_date and study.dates.primary_completion_date:
            # Use registry dates - enrollment is typically 50-60% of total duration
            total = (study.dates.primary_completion_date - study.dates.start_date).days / 30.44
            # For trials with longer durations (adjuvant/perioperative), enrollment is ~40%
            # For metastatic trials, enrollment is ~55%
            enroll_fraction = 0.45  # More conservative default
            return total * enroll_fraction

        # Estimate based on sample size with realistic rates
        # Large global Phase 3 oncology trials: 15-25 pts/month (not 40)
        # Factors: COVID impact, multi-center coordination, screening failures,
        # site activation delays, complex eligibility criteria
        n = study.design.enrollment

        if n > 1000:
            rate = 25  # Large trials - still slower than assumed
        elif n > 500:
            rate = 20  # Medium trials
        else:
            rate = 15  # Smaller trials, often more restrictive criteria

        return n / rate

    def _get_model_timeline_factor(self, study: StudyData) -> float:
        """
        Calculate calibration factor between model and registry timeline.

        The model tends to predict faster event accumulation than reality.
        This returns a factor to adjust the timeline.
        """
        if not (study.dates.start_date and study.dates.primary_completion_date):
            return 1.0

        registry_duration = (study.dates.primary_completion_date - study.dates.start_date).days / 30.44

        # Model typically underestimates by 30-50% for various reasons:
        # - Screening period before randomization
        # - Non-uniform enrollment (slow start, ramp-up)
        # - Administrative/monitoring delays
        # - Data maturity requirements beyond minimum follow-up
        return 1.0  # Return 1.0 for now - calibration applied elsewhere

    def _calc_events(self, hr: float, power: float, alpha: float) -> int:
        """Calculate required events using Schoenfeld formula."""
        from scipy.stats import norm
        z_alpha = norm.ppf(1 - alpha)
        z_beta = norm.ppf(power)
        d = 4 * (z_alpha + z_beta) ** 2 / (np.log(hr) ** 2)
        return int(np.ceil(d))

    def _build_scenarios(
        self,
        landscape: Optional[BenchmarkLandscape],
        design_hr: float,
    ) -> Dict[str, float]:
        """Build scenario dict from competitive landscape."""
        scenarios = {}

        # Base case = design HR
        scenarios["Base Case"] = design_hr

        # Bull case = better than design
        if landscape and landscape.best_in_class_hr:
            scenarios["Bull (Best-in-Class)"] = landscape.best_in_class_hr
        else:
            scenarios["Bull"] = design_hr * 0.85

        # Bear case = worse than design
        scenarios["Bear"] = min(design_hr * 1.2, 0.95)

        # Add competitor scenarios if available
        if landscape:
            for i, comp in enumerate(landscape.competitors[:2]):
                if comp.hazard_ratio:
                    scenarios[f"Like {comp.trial_name}"] = comp.hazard_ratio

        return scenarios

    def _generate_insights(
        self,
        study: StudyData,
        landscape: Optional[BenchmarkLandscape],
        scenarios: MultiScenarioAnalysis,
        predictions: dict,
    ) -> List[str]:
        """Generate key insights from analysis."""
        insights = []

        # Probability insight
        base_prob = scenarios.base_case.probability_of_success
        if base_prob >= 0.8:
            insights.append(f"High probability of success ({base_prob:.0%}) under base case assumptions")
        elif base_prob >= 0.5:
            insights.append(f"Moderate probability of success ({base_prob:.0%}) - execution critical")
        else:
            insights.append(f"Challenging probability of success ({base_prob:.0%}) - needs strong efficacy")

        # Timeline insight
        final_date = predictions['final']['median_date']
        if study.dates.primary_completion_date:
            pcd = study.dates.primary_completion_date.date() if hasattr(study.dates.primary_completion_date, 'date') else study.dates.primary_completion_date
            days_diff = (pcd - final_date).days
            if days_diff > 180:
                insights.append(f"Model predicts readout ~{days_diff//30} months earlier than registry date")
            elif days_diff < -180:
                insights.append(f"Model predicts readout ~{-days_diff//30} months later than registry date")

        # Competitive insight
        if landscape and landscape.best_in_class_hr:
            insights.append(f"Best-in-class HR in this space: {landscape.best_in_class_hr:.2f}")

        # Critical HR insight
        crit_hr = predictions['final']['critical_hr']
        insights.append(f"Needs HR ≤ {crit_hr:.2f} at final analysis for significance")

        return insights


# Convenience function
def quick_intelligence(nct_id: str, gemini_api_key: Optional[str] = None) -> IntelligenceReport:
    """Quick helper to generate intelligence report."""
    engine = TrialIntelligenceEngine(gemini_api_key=gemini_api_key, n_simulations=500)
    return engine.analyze(nct_id)
