"""
Benchmark analysis using Gemini AI to identify relevant historical data
and propose trial parameters.
"""

import json
import os
from dataclasses import dataclass
from typing import Optional

import google.generativeai as genai

try:
    from ..api.clinicaltrials import StudyData
except ImportError:
    from api.clinicaltrials import StudyData


@dataclass
class BenchmarkData:
    """Historical benchmark data for a specific indication/endpoint."""
    indication: str
    line_of_therapy: str
    endpoint: str
    control_arm: str
    median_survival_months: float
    hazard_ratio_range: tuple[float, float]
    source_trials: list[str]
    notes: str


@dataclass
class TrialPrediction:
    """Prediction and recommendations for a trial."""
    recommended_hr: float
    hr_justification: str
    control_median: float
    expected_events: int
    recommended_power: float
    recommended_alpha: float
    interim_analyses: list[float]  # Information fractions
    benchmark_sources: list[BenchmarkData]
    probability_of_success: float
    risk_factors: list[str]
    assumptions: list[str]


class BenchmarkAnalyzer:
    """
    Analyzes clinical trials and identifies relevant benchmarks using AI.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the benchmark analyzer.

        Args:
            api_key: Gemini API key (default: from GEMINI_API_KEY env var)
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel("gemini-2.0-flash")
        else:
            self.model = None

    def analyze_trial(self, study: StudyData) -> TrialPrediction:
        """
        Analyze a clinical trial and generate predictions.

        Args:
            study: Study data from ClinicalTrials.gov

        Returns:
            TrialPrediction with recommendations
        """
        if self.model is None:
            return self._generate_default_prediction(study)

        # Create prompt for Gemini
        prompt = self._create_analysis_prompt(study)

        try:
            response = self.model.generate_content(prompt)
            return self._parse_prediction_response(response.text, study)
        except Exception as e:
            print(f"Gemini API error: {e}")
            return self._generate_default_prediction(study)

    def _create_analysis_prompt(self, study: StudyData) -> str:
        """Create the prompt for Gemini analysis."""
        conditions = ", ".join(study.conditions)
        primary_outcomes = "\n".join([
            f"- {o.title}: {o.description} (timeframe: {o.time_frame})"
            for o in study.primary_outcomes
        ])
        arms = "\n".join([
            f"- {a.label} ({a.arm_type}): {', '.join(a.interventions)}"
            for a in study.arms
        ])

        prompt = f"""You are an expert clinical trial statistician and oncologist. Analyze the following Phase 3 clinical trial and provide benchmark data and predictions.

TRIAL INFORMATION:
- NCT ID: {study.nct_id}
- Title: {study.title}
- Conditions: {conditions}
- Phase: {', '.join(study.design.phases)}
- Enrollment: {study.design.enrollment} patients
- Status: {study.status}

PRIMARY OUTCOMES:
{primary_outcomes}

STUDY ARMS:
{arms}

BRIEF SUMMARY:
{study.brief_summary}

Please provide a detailed analysis in the following JSON format:
{{
    "indication": "specific cancer type and stage",
    "line_of_therapy": "1L, 2L, 3L+, etc.",
    "endpoint_type": "OS, PFS, DFS, etc.",
    "control_arm_treatment": "standard of care description",
    "control_median_months": <expected median survival/PFS in control arm>,
    "recommended_hazard_ratio": <HR needed for clinically meaningful benefit>,
    "hr_range": [<lower_bound>, <upper_bound>],
    "hr_justification": "explanation of why this HR is appropriate",
    "benchmark_trials": [
        {{
            "trial_name": "trial identifier",
            "median_survival": <months>,
            "hazard_ratio": <if applicable>,
            "notes": "relevant context"
        }}
    ],
    "recommended_power": <0.8 to 0.95>,
    "recommended_alpha": <one-sided alpha, typically 0.025>,
    "interim_analyses": [<information fractions, e.g., 0.5, 0.75, 1.0>],
    "probability_of_success": <0.0 to 1.0>,
    "risk_factors": ["list of factors that could reduce success probability"],
    "assumptions": ["key assumptions in this analysis"]
}}

Be specific and base your recommendations on published clinical trial data for similar indications and patient populations. If this is an immunotherapy or targeted therapy trial, consider delayed treatment effects."""

        return prompt

    def _parse_prediction_response(self, response_text: str, study: StudyData) -> TrialPrediction:
        """Parse Gemini response into TrialPrediction."""
        try:
            # Extract JSON from response
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                data = json.loads(json_str)
            else:
                return self._generate_default_prediction(study)

            # Parse benchmark trials
            benchmarks = []
            for trial in data.get("benchmark_trials", []):
                benchmarks.append(BenchmarkData(
                    indication=data.get("indication", ""),
                    line_of_therapy=data.get("line_of_therapy", ""),
                    endpoint=data.get("endpoint_type", ""),
                    control_arm=data.get("control_arm_treatment", ""),
                    median_survival_months=trial.get("median_survival", 12.0),
                    hazard_ratio_range=tuple(data.get("hr_range", [0.6, 0.8])),
                    source_trials=[trial.get("trial_name", "")],
                    notes=trial.get("notes", ""),
                ))

            control_median = data.get("control_median_months", 12.0)
            recommended_hr = data.get("recommended_hazard_ratio", 0.7)

            # Calculate expected events based on sample size and median
            # Approximate using the Schoenfeld formula
            from .group_sequential import calculate_sample_size
            expected_events = calculate_sample_size(
                alpha=data.get("recommended_alpha", 0.025),
                power=data.get("recommended_power", 0.9),
                hazard_ratio=recommended_hr,
            )

            return TrialPrediction(
                recommended_hr=recommended_hr,
                hr_justification=data.get("hr_justification", ""),
                control_median=control_median,
                expected_events=expected_events,
                recommended_power=data.get("recommended_power", 0.9),
                recommended_alpha=data.get("recommended_alpha", 0.025),
                interim_analyses=data.get("interim_analyses", [0.6, 1.0]),
                benchmark_sources=benchmarks,
                probability_of_success=data.get("probability_of_success", 0.5),
                risk_factors=data.get("risk_factors", []),
                assumptions=data.get("assumptions", []),
            )

        except json.JSONDecodeError:
            return self._generate_default_prediction(study)

    def _generate_default_prediction(self, study: StudyData) -> TrialPrediction:
        """Generate default prediction when AI analysis is unavailable."""
        # Default values based on typical Phase 3 oncology trials
        endpoint_type = study.get_primary_endpoint_type() or "Time-to-Event"

        # Set defaults based on endpoint type
        if endpoint_type == "Overall Survival":
            control_median = 12.0
            recommended_hr = 0.75
        elif endpoint_type == "Progression-Free Survival":
            control_median = 8.0
            recommended_hr = 0.70
        else:
            control_median = 10.0
            recommended_hr = 0.70

        from .group_sequential import calculate_sample_size
        expected_events = calculate_sample_size(
            alpha=0.025,
            power=0.9,
            hazard_ratio=recommended_hr,
        )

        return TrialPrediction(
            recommended_hr=recommended_hr,
            hr_justification=f"Default assumption for {endpoint_type} endpoint",
            control_median=control_median,
            expected_events=expected_events,
            recommended_power=0.9,
            recommended_alpha=0.025,
            interim_analyses=[0.6, 1.0],
            benchmark_sources=[],
            probability_of_success=0.5,
            risk_factors=[
                "No AI-based benchmark analysis available",
                "Using default assumptions",
            ],
            assumptions=[
                f"Assumed control arm median: {control_median} months",
                f"Assumed hazard ratio: {recommended_hr}",
                "Assuming proportional hazards",
            ],
        )

    def get_historical_benchmarks(
        self,
        indication: str,
        endpoint: str,
        line_of_therapy: str = "any"
    ) -> list[BenchmarkData]:
        """
        Query for historical benchmarks for a specific indication.

        Args:
            indication: Disease indication (e.g., "NSCLC", "breast cancer")
            endpoint: Endpoint type (e.g., "OS", "PFS")
            line_of_therapy: Line of therapy (e.g., "1L", "2L")

        Returns:
            List of relevant benchmark data
        """
        if self.model is None:
            return []

        prompt = f"""Provide historical clinical trial benchmark data for:
- Indication: {indication}
- Endpoint: {endpoint}
- Line of therapy: {line_of_therapy}

Return a JSON array of benchmark trials with the following structure:
[
    {{
        "trial_name": "trial identifier",
        "indication": "specific indication",
        "line_of_therapy": "1L/2L/3L+",
        "endpoint": "OS/PFS/DFS",
        "control_arm": "treatment description",
        "median_months": <median survival/PFS>,
        "hazard_ratio": <if experimental arm reported>,
        "sample_size": <number of patients>,
        "year": <publication year>,
        "notes": "relevant context"
    }}
]

Include only well-established, published Phase 3 trials from the past 10 years."""

        try:
            response = self.model.generate_content(prompt)
            json_start = response.text.find("[")
            json_end = response.text.rfind("]") + 1
            if json_start >= 0 and json_end > json_start:
                data = json.loads(response.text[json_start:json_end])
                return [
                    BenchmarkData(
                        indication=d.get("indication", indication),
                        line_of_therapy=d.get("line_of_therapy", line_of_therapy),
                        endpoint=d.get("endpoint", endpoint),
                        control_arm=d.get("control_arm", ""),
                        median_survival_months=d.get("median_months", 0),
                        hazard_ratio_range=(
                            d.get("hazard_ratio", 1.0) * 0.9,
                            d.get("hazard_ratio", 1.0) * 1.1
                        ) if d.get("hazard_ratio") else (0.6, 0.8),
                        source_trials=[d.get("trial_name", "")],
                        notes=d.get("notes", ""),
                    )
                    for d in data
                ]
        except Exception as e:
            print(f"Error fetching benchmarks: {e}")

        return []
