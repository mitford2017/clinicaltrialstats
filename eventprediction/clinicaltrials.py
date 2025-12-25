"""
ClinicalTrials.gov API integration for fetching trial information.
"""

import requests
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import date, datetime
import warnings


@dataclass
class TrialInfo:
    """
    Information about a clinical trial from ClinicalTrials.gov.
    
    Attributes
    ----------
    nct_id : str
        NCT identifier (e.g., 'NCT12345678')
    title : str
        Official title of the study
    brief_title : str
        Brief title
    status : str
        Overall recruitment status
    start_date : Optional[date]
        Study start date
    primary_completion_date : Optional[date]
        Primary completion date
    completion_date : Optional[date]
        Study completion date
    enrollment : Optional[int]
        Target or actual enrollment
    phase : str
        Study phase
    study_type : str
        Type of study (Interventional, Observational, etc.)
    conditions : List[str]
        Conditions being studied
    interventions : List[str]
        Interventions being studied
    sponsor : str
        Lead sponsor
    primary_outcome_measures : List[str]
        Primary outcome measures
    secondary_outcome_measures : List[str]
        Secondary outcome measures
    raw_data : Dict[str, Any]
        Raw API response data
    """
    nct_id: str
    title: str = ""
    brief_title: str = ""
    status: str = ""
    start_date: Optional[date] = None
    primary_completion_date: Optional[date] = None
    completion_date: Optional[date] = None
    enrollment: Optional[int] = None
    phase: str = ""
    study_type: str = ""
    conditions: List[str] = field(default_factory=list)
    interventions: List[str] = field(default_factory=list)
    sponsor: str = ""
    primary_outcome_measures: List[str] = field(default_factory=list)
    secondary_outcome_measures: List[str] = field(default_factory=list)
    raw_data: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def study_duration_months(self) -> Optional[float]:
        """Calculate study duration in months from start to primary completion."""
        if self.start_date and self.primary_completion_date:
            delta = self.primary_completion_date - self.start_date
            return delta.days / 30.44  # Average days per month
        return None
    
    @property
    def accrual_period_months(self) -> Optional[float]:
        """
        Estimate accrual period (assume 70% of time before primary completion).
        
        This is a rough estimate - actual accrual period should be specified
        in the protocol.
        """
        duration = self.study_duration_months
        if duration:
            return duration * 0.7
        return None
    
    def to_study_params(self, 
                        HR: float = 0.7,
                        alpha: float = 0.05,
                        power: float = 0.8,
                        r: float = 1.0,
                        ctrl_median: Optional[float] = None,
                        shape: float = 1.0) -> dict:
        """
        Convert trial info to Study parameters.
        
        Parameters
        ----------
        HR : float
            Hazard ratio (must be provided, no default from trial)
        alpha : float
            Significance level
        power : float
            Target power
        r : float
            Allocation ratio
        ctrl_median : float, optional
            Control arm median survival (must be provided)
        shape : float
            Weibull shape parameter
            
        Returns
        -------
        dict
            Parameters suitable for Study constructor
        """
        if ctrl_median is None:
            raise ValueError("ctrl_median must be provided - "
                             "this cannot be determined from ClinicalTrials.gov")
        
        params = {
            'N': self.enrollment or 500,
            'study_duration': self.study_duration_months or 36,
            'acc_period': self.accrual_period_months or 18,
            'ctrl_median': ctrl_median,
            'HR': HR,
            'alpha': alpha,
            'power': power,
            'r': r,
            'k': 1.0,  # Assume uniform accrual
            'shape': shape,
            'two_sided': True
        }
        
        return params
    
    def __str__(self) -> str:
        lines = [
            f"NCT ID: {self.nct_id}",
            f"Title: {self.brief_title or self.title}",
            f"Status: {self.status}",
            f"Phase: {self.phase}",
            f"Enrollment: {self.enrollment}",
            f"Start Date: {self.start_date}",
            f"Primary Completion: {self.primary_completion_date}",
            f"Completion Date: {self.completion_date}",
        ]
        
        if self.study_duration_months:
            lines.append(f"Study Duration: {self.study_duration_months:.1f} months")
        
        if self.conditions:
            lines.append(f"Conditions: {', '.join(self.conditions[:3])}")
        
        if self.sponsor:
            lines.append(f"Sponsor: {self.sponsor}")
        
        return '\n'.join(lines)


def _parse_date(date_str: Optional[str]) -> Optional[date]:
    """Parse date string from ClinicalTrials.gov API."""
    if not date_str:
        return None
    
    # Try different formats
    formats = ['%Y-%m-%d', '%B %d, %Y', '%B %Y', '%Y-%m']
    
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt.date()
        except ValueError:
            continue
    
    # Try to extract year and month
    parts = date_str.split()
    if len(parts) >= 2:
        try:
            # "Month Year" format
            month_str = parts[0]
            year_str = parts[-1]
            dt = datetime.strptime(f"{month_str} {year_str}", "%B %Y")
            return dt.date()
        except ValueError:
            pass
    
    return None


def fetch_trial_info(nct_id: str) -> TrialInfo:
    """
    Fetch trial information from ClinicalTrials.gov API.
    
    Parameters
    ----------
    nct_id : str
        NCT identifier (e.g., 'NCT12345678' or just '12345678')
        
    Returns
    -------
    TrialInfo
        Trial information
        
    Raises
    ------
    ValueError
        If trial not found or API error
    """
    # Clean up NCT ID
    nct_id = nct_id.strip().upper()
    if not nct_id.startswith('NCT'):
        nct_id = f'NCT{nct_id}'
    
    # ClinicalTrials.gov API v2
    url = f"https://clinicaltrials.gov/api/v2/studies/{nct_id}"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            raise ValueError(f"Trial {nct_id} not found on ClinicalTrials.gov")
        raise ValueError(f"Error fetching trial data: {e}")
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Network error fetching trial data: {e}")
    
    data = response.json()
    
    # Extract protocol section
    protocol = data.get('protocolSection', {})
    
    # Identification
    id_module = protocol.get('identificationModule', {})
    
    # Status
    status_module = protocol.get('statusModule', {})
    
    # Design
    design_module = protocol.get('designModule', {})
    
    # Conditions
    conditions_module = protocol.get('conditionsModule', {})
    
    # Arms/Interventions
    arms_module = protocol.get('armsInterventionsModule', {})
    
    # Outcomes
    outcomes_module = protocol.get('outcomesModule', {})
    
    # Sponsor
    sponsor_module = protocol.get('sponsorCollaboratorsModule', {})
    
    # Parse dates
    start_date_struct = status_module.get('startDateStruct', {})
    primary_completion_struct = status_module.get('primaryCompletionDateStruct', {})
    completion_struct = status_module.get('completionDateStruct', {})
    
    start_date = _parse_date(start_date_struct.get('date'))
    primary_completion = _parse_date(primary_completion_struct.get('date'))
    completion_date = _parse_date(completion_struct.get('date'))
    
    # Parse enrollment
    enrollment_info = design_module.get('enrollmentInfo', {})
    enrollment = enrollment_info.get('count')
    
    # Parse interventions
    interventions = []
    for interv in arms_module.get('interventions', []):
        name = interv.get('name', '')
        itype = interv.get('type', '')
        if name:
            interventions.append(f"{itype}: {name}" if itype else name)
    
    # Parse outcomes
    primary_outcomes = []
    for outcome in outcomes_module.get('primaryOutcomes', []):
        measure = outcome.get('measure', '')
        if measure:
            primary_outcomes.append(measure)
    
    secondary_outcomes = []
    for outcome in outcomes_module.get('secondaryOutcomes', []):
        measure = outcome.get('measure', '')
        if measure:
            secondary_outcomes.append(measure)
    
    # Lead sponsor
    lead_sponsor = sponsor_module.get('leadSponsor', {})
    sponsor_name = lead_sponsor.get('name', '')
    
    # Phases
    phases = design_module.get('phases', [])
    phase_str = ', '.join(phases) if phases else 'N/A'
    
    return TrialInfo(
        nct_id=nct_id,
        title=id_module.get('officialTitle', ''),
        brief_title=id_module.get('briefTitle', ''),
        status=status_module.get('overallStatus', ''),
        start_date=start_date,
        primary_completion_date=primary_completion,
        completion_date=completion_date,
        enrollment=enrollment,
        phase=phase_str,
        study_type=design_module.get('studyType', ''),
        conditions=conditions_module.get('conditions', []),
        interventions=interventions,
        sponsor=sponsor_name,
        primary_outcome_measures=primary_outcomes,
        secondary_outcome_measures=secondary_outcomes,
        raw_data=data
    )


def create_study_from_nct(nct_id: str,
                          HR: float,
                          ctrl_median: float,
                          alpha: float = 0.05,
                          power: float = 0.8,
                          r: float = 1.0,
                          shape: float = 1.0,
                          k: float = 1.0,
                          override_enrollment: Optional[int] = None,
                          override_duration: Optional[float] = None,
                          override_acc_period: Optional[float] = None) -> 'Study':
    """
    Create a Study object from a ClinicalTrials.gov NCT ID.
    
    Parameters
    ----------
    nct_id : str
        NCT identifier
    HR : float
        Hazard ratio
    ctrl_median : float
        Control arm median survival (in months)
    alpha : float
        Significance level
    power : float
        Target power
    r : float
        Allocation ratio
    shape : float
        Weibull shape parameter
    k : float
        Accrual non-uniformity parameter
    override_enrollment : int, optional
        Override enrollment from trial
    override_duration : float, optional
        Override study duration (months)
    override_acc_period : float, optional
        Override accrual period (months)
        
    Returns
    -------
    Study
    """
    from .study import Study
    
    # Fetch trial info
    trial_info = fetch_trial_info(nct_id)
    
    print(f"Fetched trial info for {nct_id}:")
    print(trial_info)
    print()
    
    # Get values with overrides
    N = override_enrollment or trial_info.enrollment or 500
    
    if override_duration:
        study_duration = override_duration
    elif trial_info.study_duration_months:
        study_duration = trial_info.study_duration_months
    else:
        warnings.warn("Could not determine study duration, using 36 months")
        study_duration = 36
    
    if override_acc_period:
        acc_period = override_acc_period
    elif trial_info.accrual_period_months:
        acc_period = trial_info.accrual_period_months
    else:
        acc_period = study_duration * 0.6
    
    # Make sure acc_period < study_duration
    if acc_period >= study_duration:
        acc_period = study_duration * 0.6
    
    print(f"Creating Study with:")
    print(f"  N = {N}")
    print(f"  Study duration = {study_duration:.1f} months")
    print(f"  Accrual period = {acc_period:.1f} months")
    print(f"  HR = {HR}")
    print(f"  Control median = {ctrl_median} months")
    print()
    
    return Study(
        N=int(N),
        study_duration=study_duration,
        acc_period=acc_period,
        ctrl_median=ctrl_median,
        HR=HR,
        alpha=alpha,
        power=power,
        r=r,
        k=k,
        shape=shape,
        two_sided=True
    )


def search_trials(query: str,
                  status: Optional[str] = None,
                  phase: Optional[str] = None,
                  max_results: int = 10) -> List[TrialInfo]:
    """
    Search for trials on ClinicalTrials.gov.
    
    Parameters
    ----------
    query : str
        Search query (condition, intervention, etc.)
    status : str, optional
        Filter by status ('RECRUITING', 'COMPLETED', etc.)
    phase : str, optional
        Filter by phase ('PHASE3', 'PHASE2', etc.)
    max_results : int
        Maximum number of results to return
        
    Returns
    -------
    List[TrialInfo]
    """
    url = "https://clinicaltrials.gov/api/v2/studies"
    
    params = {
        'query.term': query,
        'pageSize': min(max_results, 100),
        'format': 'json'
    }
    
    if status:
        params['filter.overallStatus'] = status
    
    if phase:
        params['filter.phase'] = phase
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Error searching trials: {e}")
    
    data = response.json()
    studies = data.get('studies', [])
    
    results = []
    for study_data in studies[:max_results]:
        try:
            protocol = study_data.get('protocolSection', {})
            id_module = protocol.get('identificationModule', {})
            nct_id = id_module.get('nctId', '')
            
            if nct_id:
                # Fetch full info for each study
                results.append(fetch_trial_info(nct_id))
        except Exception as e:
            warnings.warn(f"Error processing study: {e}")
            continue
    
    return results

