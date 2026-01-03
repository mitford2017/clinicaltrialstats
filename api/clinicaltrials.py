"""
ClinicalTrials.gov API client for retrieving study data.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import requests


@dataclass
class OutcomeMeasure:
    """Represents a trial outcome measure."""
    title: str
    description: str
    time_frame: str
    outcome_type: str  # "primary", "secondary", "other"


@dataclass
class StudyArm:
    """Represents a study arm/treatment group."""
    label: str
    arm_type: str  # "Experimental", "Active Comparator", "Placebo Comparator", etc.
    description: str
    interventions: list[str] = field(default_factory=list)


@dataclass
class StudyDesign:
    """Study design characteristics."""
    study_type: str
    phases: list[str]
    allocation: Optional[str]
    intervention_model: Optional[str]
    masking: Optional[str]
    primary_purpose: Optional[str]
    enrollment: int
    enrollment_type: str  # "Actual" or "Anticipated"


@dataclass
class StudyDates:
    """Key study dates."""
    start_date: Optional[datetime]
    primary_completion_date: Optional[datetime]
    completion_date: Optional[datetime]
    first_posted: Optional[datetime]
    last_update: Optional[datetime]


@dataclass
class StudyData:
    """Complete study data extracted from ClinicalTrials.gov."""
    nct_id: str
    title: str
    brief_title: str
    status: str
    sponsor: str
    conditions: list[str]
    design: StudyDesign
    dates: StudyDates
    arms: list[StudyArm]
    primary_outcomes: list[OutcomeMeasure]
    secondary_outcomes: list[OutcomeMeasure]
    eligibility_criteria: str
    brief_summary: str
    detailed_description: str

    def get_primary_endpoint_type(self) -> Optional[str]:
        """Infer the primary endpoint type from outcome measures."""
        if not self.primary_outcomes:
            return None

        title = self.primary_outcomes[0].title.lower()
        description = self.primary_outcomes[0].description.lower() if self.primary_outcomes[0].description else ""
        combined = f"{title} {description}"

        if any(term in combined for term in ["overall survival", "os", "death"]):
            return "Overall Survival"
        elif any(term in combined for term in ["progression-free", "pfs", "progression free"]):
            return "Progression-Free Survival"
        elif any(term in combined for term in ["disease-free", "dfs", "disease free", "recurrence"]):
            return "Disease-Free Survival"
        elif any(term in combined for term in ["event-free", "efs", "event free"]):
            return "Event-Free Survival"
        elif any(term in combined for term in ["response", "orr", "objective response"]):
            return "Objective Response Rate"
        elif any(term in combined for term in ["duration of response", "dor"]):
            return "Duration of Response"
        elif any(term in combined for term in ["time to progression", "ttp"]):
            return "Time to Progression"
        else:
            return "Time-to-Event"

    def is_time_to_event_endpoint(self) -> bool:
        """Check if the primary endpoint is a time-to-event endpoint."""
        endpoint_type = self.get_primary_endpoint_type()
        tte_endpoints = [
            "Overall Survival", "Progression-Free Survival",
            "Disease-Free Survival", "Event-Free Survival",
            "Time to Progression", "Time-to-Event"
        ]
        return endpoint_type in tte_endpoints


class ClinicalTrialsAPI:
    """Client for ClinicalTrials.gov API v2."""

    BASE_URL = "https://clinicaltrials.gov/api/v2"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "ClinicalTrialPredictor/1.0"
        })

    def get_study(self, nct_id: str) -> StudyData:
        """Retrieve study data by NCT ID."""
        nct_id = self._normalize_nct_id(nct_id)

        url = f"{self.BASE_URL}/studies/{nct_id}"
        response = self.session.get(url)
        response.raise_for_status()

        data = response.json()
        return self._parse_study(data)

    def search_studies(
        self,
        condition: Optional[str] = None,
        intervention: Optional[str] = None,
        phase: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 10
    ) -> list[StudyData]:
        """Search for studies matching criteria."""
        params = {"pageSize": limit}

        query_parts = []
        if condition:
            query_parts.append(f"AREA[Condition]{condition}")
        if intervention:
            query_parts.append(f"AREA[Intervention]{intervention}")
        if phase:
            query_parts.append(f"AREA[Phase]{phase}")
        if status:
            query_parts.append(f"AREA[OverallStatus]{status}")

        if query_parts:
            params["query.term"] = " AND ".join(query_parts)

        url = f"{self.BASE_URL}/studies"
        response = self.session.get(url, params=params)
        response.raise_for_status()

        data = response.json()
        studies = []
        for study_data in data.get("studies", []):
            try:
                studies.append(self._parse_study(study_data))
            except Exception:
                continue

        return studies

    def _normalize_nct_id(self, nct_id: str) -> str:
        """Normalize NCT ID format."""
        nct_id = nct_id.strip().upper()
        if not nct_id.startswith("NCT"):
            nct_id = f"NCT{nct_id}"

        match = re.match(r"NCT(\d+)", nct_id)
        if match:
            number = match.group(1).zfill(8)
            return f"NCT{number}"

        return nct_id

    def _parse_study(self, data: dict) -> StudyData:
        """Parse API response into StudyData."""
        protocol = data.get("protocolSection", {})

        # Identification
        id_module = protocol.get("identificationModule", {})
        nct_id = id_module.get("nctId", "")
        title = id_module.get("officialTitle", "")
        brief_title = id_module.get("briefTitle", "")

        # Status
        status_module = protocol.get("statusModule", {})
        status = status_module.get("overallStatus", "")

        # Sponsor
        sponsor_module = protocol.get("sponsorCollaboratorsModule", {})
        lead_sponsor = sponsor_module.get("leadSponsor", {})
        sponsor = lead_sponsor.get("name", "")

        # Conditions
        conditions_module = protocol.get("conditionsModule", {})
        conditions = conditions_module.get("conditions", [])

        # Design
        design_module = protocol.get("designModule", {})
        design = self._parse_design(design_module)

        # Dates
        dates = self._parse_dates(status_module)

        # Arms and Interventions
        arms_module = protocol.get("armsInterventionsModule", {})
        arms = self._parse_arms(arms_module)

        # Outcomes
        outcomes_module = protocol.get("outcomesModule", {})
        primary_outcomes = self._parse_outcomes(
            outcomes_module.get("primaryOutcomes", []), "primary"
        )
        secondary_outcomes = self._parse_outcomes(
            outcomes_module.get("secondaryOutcomes", []), "secondary"
        )

        # Eligibility
        eligibility_module = protocol.get("eligibilityModule", {})
        eligibility_criteria = eligibility_module.get("eligibilityCriteria", "")

        # Description
        description_module = protocol.get("descriptionModule", {})
        brief_summary = description_module.get("briefSummary", "")
        detailed_description = description_module.get("detailedDescription", "")

        return StudyData(
            nct_id=nct_id,
            title=title,
            brief_title=brief_title,
            status=status,
            sponsor=sponsor,
            conditions=conditions,
            design=design,
            dates=dates,
            arms=arms,
            primary_outcomes=primary_outcomes,
            secondary_outcomes=secondary_outcomes,
            eligibility_criteria=eligibility_criteria,
            brief_summary=brief_summary,
            detailed_description=detailed_description,
        )

    def _parse_design(self, design_module: dict) -> StudyDesign:
        """Parse study design information."""
        study_type = design_module.get("studyType", "")
        phases = design_module.get("phases", [])

        design_info = design_module.get("designInfo", {})
        allocation = design_info.get("allocation")
        intervention_model = design_info.get("interventionModel")

        masking_info = design_info.get("maskingInfo", {})
        masking = masking_info.get("masking")

        primary_purpose = design_info.get("primaryPurpose")

        enrollment_info = design_module.get("enrollmentInfo", {})
        enrollment = enrollment_info.get("count", 0)
        enrollment_type = enrollment_info.get("type", "")

        return StudyDesign(
            study_type=study_type,
            phases=phases,
            allocation=allocation,
            intervention_model=intervention_model,
            masking=masking,
            primary_purpose=primary_purpose,
            enrollment=enrollment,
            enrollment_type=enrollment_type,
        )

    def _parse_dates(self, status_module: dict) -> StudyDates:
        """Parse study dates."""
        def parse_date(date_struct: Optional[dict]) -> Optional[datetime]:
            if not date_struct:
                return None
            date_str = date_struct.get("date")
            if not date_str:
                return None
            try:
                # Handle various date formats
                for fmt in ["%Y-%m-%d", "%Y-%m", "%B %d, %Y", "%B %Y"]:
                    try:
                        return datetime.strptime(date_str, fmt)
                    except ValueError:
                        continue
            except Exception:
                pass
            return None

        return StudyDates(
            start_date=parse_date(status_module.get("startDateStruct")),
            primary_completion_date=parse_date(status_module.get("primaryCompletionDateStruct")),
            completion_date=parse_date(status_module.get("completionDateStruct")),
            first_posted=parse_date(status_module.get("studyFirstPostDateStruct")),
            last_update=parse_date(status_module.get("lastUpdatePostDateStruct")),
        )

    def _parse_arms(self, arms_module: dict) -> list[StudyArm]:
        """Parse study arms and interventions."""
        arms_data = arms_module.get("armGroups", [])
        interventions_data = arms_module.get("interventions", [])

        # Map interventions to arms
        intervention_map: dict[str, list[str]] = {}
        for intervention in interventions_data:
            name = intervention.get("name", "")
            arm_labels = intervention.get("armGroupLabels", [])
            for label in arm_labels:
                if label not in intervention_map:
                    intervention_map[label] = []
                intervention_map[label].append(name)

        arms = []
        for arm_data in arms_data:
            label = arm_data.get("label", "")
            arms.append(StudyArm(
                label=label,
                arm_type=arm_data.get("type", ""),
                description=arm_data.get("description", ""),
                interventions=intervention_map.get(label, []),
            ))

        return arms

    def _parse_outcomes(
        self, outcomes_data: list[dict], outcome_type: str
    ) -> list[OutcomeMeasure]:
        """Parse outcome measures."""
        outcomes = []
        for outcome in outcomes_data:
            outcomes.append(OutcomeMeasure(
                title=outcome.get("measure", ""),
                description=outcome.get("description", ""),
                time_frame=outcome.get("timeFrame", ""),
                outcome_type=outcome_type,
            ))
        return outcomes
