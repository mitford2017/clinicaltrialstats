"""
Benchmark Research Agent - Uses Gemini to gather competitive intelligence.

This module searches for and extracts:
- Competitor trials in the same indication/line of therapy
- Published efficacy data (HR, median OS/PFS, ORR)
- Standard of care benchmarks
- Protocol design assumptions
"""

import json
import os
from dataclasses import dataclass, field
from typing import Optional, List
from datetime import date

try:
    from google import genai
    from google.genai import types
    NEW_SDK = True
except ImportError:
    import google.generativeai as genai
    NEW_SDK = False


@dataclass
class CompetitorTrial:
    """Data from a competitor or benchmark trial."""
    trial_name: str
    nct_id: Optional[str]
    drug_name: str
    indication: str
    line_of_therapy: str
    comparator: str
    primary_endpoint: str
    hazard_ratio: Optional[float]
    hazard_ratio_ci: Optional[tuple[float, float]]
    median_experimental: Optional[float]  # months
    median_control: Optional[float]  # months
    sample_size: Optional[int]
    events: Optional[int]
    p_value: Optional[float]
    publication_year: Optional[int]
    source: str  # URL or citation
    notes: str


@dataclass
class BenchmarkLandscape:
    """Complete competitive landscape for a trial."""
    indication: str
    line_of_therapy: str
    standard_of_care: str
    soc_median_survival: float
    competitors: List[CompetitorTrial]
    best_in_class_hr: Optional[float]
    typical_hr_range: tuple[float, float]
    recommended_design_hr: float
    recommended_scenario_hr: float
    rationale: str


@dataclass
class ProtocolDesign:
    """Extracted protocol design parameters."""
    design_hr: float
    power: float
    alpha: float
    interim_alpha: Optional[float]
    final_alpha: Optional[float]
    interim_events: Optional[int]
    final_events: Optional[int]
    interim_info_fraction: Optional[float]
    alpha_spending: str  # "OBF", "Pocock", etc.
    primary_endpoint: str
    secondary_endpoints: List[str]
    enrollment_target: int
    enrollment_duration_months: Optional[float]
    source: str


class BenchmarkResearchAgent:
    """
    Agent that uses Gemini with web search to gather competitive intelligence.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-pro-preview-05-06"):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model_name = model_name

        if self.api_key and NEW_SDK:
            # Use new client-based SDK with Google Search grounding
            self.client = genai.Client(api_key=self.api_key)
            self.grounding_tool = types.Tool(google_search=types.GoogleSearch())
            self.config = types.GenerateContentConfig(tools=[self.grounding_tool])
            self.model = None  # Not used in new SDK
        elif self.api_key:
            # Fallback to legacy SDK
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(model_name)
            self.client = None
            self.config = None
        else:
            self.model = None
            self.client = None
            self.config = None

    def _generate(self, prompt: str) -> Optional[str]:
        """Generate content using available SDK."""
        if self.client:
            # New SDK with Google Search grounding
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=self.config,
            )
            return response.text
        elif self.model:
            # Legacy SDK
            response = self.model.generate_content(prompt)
            return response.text
        return None

    def research_competitive_landscape(
        self,
        nct_id: str,
        indication: str,
        drug_name: str,
        line_of_therapy: str,
        primary_endpoint: str,
    ) -> Optional[BenchmarkLandscape]:
        """
        Research the competitive landscape for a trial.

        Uses web search to find:
        - Competitor Phase 3 trials in same indication/line
        - Published efficacy data
        - Standard of care benchmarks
        """
        if not self.model:
            return None

        prompt = f"""You are a clinical trial analyst researching the competitive landscape for a Phase 3 trial.

CRITICAL DISTINCTION:
- For the TRIAL BEING ANALYZED ({nct_id}): Do NOT use its own results. We want to PREDICT this trial.
- For COMPETITOR trials: You CAN use their published results as benchmarks.
- The recommended_design_hr should be what a sponsor would typically assume (0.70-0.80), NOT actual observed results.

TRIAL BEING ANALYZED:
- NCT ID: {nct_id}
- Drug: {drug_name}
- Indication: {indication}
- Line of therapy: {line_of_therapy}
- Primary endpoint: {primary_endpoint}

TASK: Search for and compile data on:

1. STANDARD OF CARE: What is the current standard treatment for this patient population? What is the typical median {primary_endpoint}?

2. COMPETITOR TRIALS: Find OTHER Phase 3 trials (NOT {nct_id}) testing similar drugs in this indication/line. For each, extract:
   - Trial name and NCT ID
   - Drug and comparator
   - Hazard ratio (with 95% CI if available)
   - Median {primary_endpoint} for both arms
   - Sample size and events
   - Publication year

3. BENCHMARK ANALYSIS (based on competitors, NOT the trial being analyzed):
   - What is the best-in-class HR achieved by OTHER trials so far?
   - What is the typical range of HRs in this space?
   - What HR would be considered clinically meaningful?

Return your findings as JSON:
{{
    "indication": "{indication}",
    "line_of_therapy": "{line_of_therapy}",
    "standard_of_care": "treatment name",
    "soc_median_months": <number>,
    "competitors": [
        {{
            "trial_name": "name",
            "nct_id": "NCT...",
            "drug_name": "name",
            "comparator": "name",
            "hazard_ratio": <number or null>,
            "hr_lower_ci": <number or null>,
            "hr_upper_ci": <number or null>,
            "median_experimental_months": <number or null>,
            "median_control_months": <number or null>,
            "sample_size": <number or null>,
            "events": <number or null>,
            "p_value": <number or null>,
            "publication_year": <number or null>,
            "source": "URL or citation",
            "notes": "key context"
        }}
    ],
    "best_in_class_hr": <number>,
    "typical_hr_range": [<lower>, <upper>],
    "recommended_design_hr": <number between 0.70-0.80 based on typical protocol assumptions>,
    "recommended_scenario_hr": <number based on competitor benchmarks>,
    "rationale": "explanation based on competitor landscape, NOT on this trial's own results"
}}

IMPORTANT: Do NOT include {nct_id} in the competitors list. Focus on 3-5 OTHER relevant trials."""

        try:
            response_text = self._generate(prompt)
            if response_text:
                return self._parse_landscape_response(response_text, indication, line_of_therapy)
            return None
        except Exception as e:
            print(f"Benchmark research error: {e}")
            return None

    def _parse_landscape_response(
        self,
        response_text: str,
        indication: str,
        line_of_therapy: str,
    ) -> Optional[BenchmarkLandscape]:
        """Parse Gemini response into BenchmarkLandscape."""
        try:
            # Extract JSON
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            if json_start < 0 or json_end <= json_start:
                return None

            data = json.loads(response_text[json_start:json_end])

            competitors = []
            for c in data.get("competitors", []):
                hr_ci = None
                if c.get("hr_lower_ci") and c.get("hr_upper_ci"):
                    hr_ci = (c["hr_lower_ci"], c["hr_upper_ci"])

                competitors.append(CompetitorTrial(
                    trial_name=c.get("trial_name", ""),
                    nct_id=c.get("nct_id"),
                    drug_name=c.get("drug_name", ""),
                    indication=indication,
                    line_of_therapy=line_of_therapy,
                    comparator=c.get("comparator", ""),
                    primary_endpoint=data.get("primary_endpoint", "OS"),
                    hazard_ratio=c.get("hazard_ratio"),
                    hazard_ratio_ci=hr_ci,
                    median_experimental=c.get("median_experimental_months"),
                    median_control=c.get("median_control_months"),
                    sample_size=c.get("sample_size"),
                    events=c.get("events"),
                    p_value=c.get("p_value"),
                    publication_year=c.get("publication_year"),
                    source=c.get("source", ""),
                    notes=c.get("notes", ""),
                ))

            hr_range = data.get("typical_hr_range", [0.65, 0.85])
            if isinstance(hr_range, list) and len(hr_range) == 2:
                hr_range = tuple(hr_range)
            else:
                hr_range = (0.65, 0.85)

            return BenchmarkLandscape(
                indication=indication,
                line_of_therapy=line_of_therapy,
                standard_of_care=data.get("standard_of_care", ""),
                soc_median_survival=data.get("soc_median_months", 12.0),
                competitors=competitors,
                best_in_class_hr=data.get("best_in_class_hr"),
                typical_hr_range=hr_range,
                recommended_design_hr=data.get("recommended_design_hr", 0.75),
                recommended_scenario_hr=data.get("recommended_scenario_hr", 0.75),
                rationale=data.get("rationale", ""),
            )

        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            return None

    def extract_protocol_design(
        self,
        nct_id: str,
        trial_title: str,
        sponsor: str,
    ) -> Optional[ProtocolDesign]:
        """
        Extract protocol design parameters using web search.

        Searches for:
        - Statistical design assumptions (HR, power, alpha)
        - Interim analysis plan
        - Event targets
        """
        if not self.model:
            return None

        prompt = f"""You are a biostatistician extracting PROSPECTIVE protocol design parameters for a clinical trial.

CRITICAL: You must find the DESIGN ASSUMPTIONS from the protocol/SAP, NOT the actual observed results.
- The design HR is what sponsors ASSUMED when calculating sample size (typically 0.70-0.80)
- Do NOT use any observed/actual hazard ratios from trial results
- If the trial has reported results, IGNORE those - we want the original design assumptions

TRIAL:
- NCT ID: {nct_id}
- Title: {trial_title}
- Sponsor: {sponsor}

TASK: Search for the ORIGINAL protocol, SAP, or trial design publications (NOT results). Extract:

1. SAMPLE SIZE ASSUMPTIONS (from protocol, not results):
   - Design hazard ratio (HR assumed for power calculation, typically 0.70-0.80)
   - Power (typically 80-90%)
   - Alpha (typically 0.025 one-sided or 0.05 two-sided)

2. INTERIM ANALYSIS PLAN:
   - Number of interim analyses
   - Information fraction at interim (e.g., 50%, 66%)
   - Alpha spending function (O'Brien-Fleming, Pocock, Lan-DeMets)
   - Alpha allocated to interim vs final

3. EVENT TARGETS:
   - Events required for interim analysis
   - Events required for final analysis

4. ENDPOINTS:
   - Primary endpoint (OS, PFS, rPFS, etc.)
   - Key secondary endpoints

Return as JSON:
{{
    "design_hr": <number>,
    "power": <number 0-1>,
    "alpha": <number>,
    "interim_alpha": <number or null>,
    "final_alpha": <number or null>,
    "interim_events": <number or null>,
    "final_events": <number or null>,
    "interim_info_fraction": <number 0-1 or null>,
    "alpha_spending": "OBF" or "Pocock" or "Lan-DeMets" or "unknown",
    "primary_endpoint": "OS" or "PFS" or "rPFS" or etc,
    "secondary_endpoints": ["list", "of", "endpoints"],
    "enrollment_target": <number>,
    "enrollment_duration_months": <number or null>,
    "source": "URL or citation where you found this"
}}

If you cannot find specific values, use null. Be specific about your sources."""

        try:
            response_text = self._generate(prompt)
            if response_text:
                return self._parse_protocol_response(response_text)
            return None
        except Exception as e:
            print(f"Protocol extraction error: {e}")
            return None

    def _parse_protocol_response(self, response_text: str) -> Optional[ProtocolDesign]:
        """Parse protocol design from Gemini response."""
        try:
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            if json_start < 0 or json_end <= json_start:
                return None

            data = json.loads(response_text[json_start:json_end])

            return ProtocolDesign(
                design_hr=data.get("design_hr", 0.75),
                power=data.get("power", 0.90),
                alpha=data.get("alpha", 0.025),
                interim_alpha=data.get("interim_alpha"),
                final_alpha=data.get("final_alpha"),
                interim_events=data.get("interim_events"),
                final_events=data.get("final_events"),
                interim_info_fraction=data.get("interim_info_fraction"),
                alpha_spending=data.get("alpha_spending", "OBF"),
                primary_endpoint=data.get("primary_endpoint", "OS"),
                secondary_endpoints=data.get("secondary_endpoints", []),
                enrollment_target=data.get("enrollment_target", 0),
                enrollment_duration_months=data.get("enrollment_duration_months"),
                source=data.get("source", ""),
            )

        except json.JSONDecodeError:
            return None

    def get_scenario_from_competitor(
        self,
        competitor_trial: str,
        nct_id: str,
    ) -> Optional[dict]:
        """
        Get scenario parameters based on a specific competitor trial.

        E.g., "What if this trial performs like VISION?" or "like PSMAfore?"
        """
        if not self.model:
            return None

        prompt = f"""Search for the results of the {competitor_trial} clinical trial.

Extract:
1. Primary endpoint hazard ratio
2. Median survival/PFS in experimental arm
3. Median survival/PFS in control arm
4. Sample size and events

Return as JSON:
{{
    "trial_name": "{competitor_trial}",
    "hazard_ratio": <number>,
    "median_experimental_months": <number>,
    "median_control_months": <number>,
    "events": <number>,
    "sample_size": <number>,
    "source": "citation"
}}"""

        try:
            response_text = self._generate(prompt)
            if response_text:
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    return json.loads(response_text[json_start:json_end])
        except Exception as e:
            print(f"Competitor lookup error: {e}")

        return None
