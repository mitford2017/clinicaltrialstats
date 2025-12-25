"""
ClinicalTrials.gov API integration.

This module provides functions to fetch trial information from ClinicalTrials.gov
and create Study objects directly from NCT identifiers.

Example Usage:
    from eventprediction import fetch_trial_info, create_study_from_nct
    
    # Fetch trial info
    trial = fetch_trial_info('NCT01844505')
    print(trial)
    
    # Create Study from NCT
    study = create_study_from_nct(
        nct_id='NCT01844505',
        HR=0.65,
        ctrl_median=12
    )
"""

import warnings
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional, Dict, Any, List, TYPE_CHECKING

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

if TYPE_CHECKING:
    from .study import Study


# =============================================================================
# Constants
# =============================================================================

API_BASE_URL = "https://clinicaltrials.gov/api/v2"
DEFAULT_TIMEOUT = 30


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TrialInfo:
    """
    Information about a clinical trial from ClinicalTrials.gov.
    
    Attributes
    ----------
    nct_id : str
        NCT identifier (e.g., 'NCT12345678')
    title : str
        Official study title
    brief_title : str
        Brief title
    status : str
        Overall recruitment status
    start_date : date, optional
        Study start date
    primary_completion_date : date, optional
        Primary completion date
    completion_date : date, optional
        Study completion date
    enrollment : int, optional
        Target or actual enrollment
    phase : str
        Study phase
    study_type : str
        Type of study
    conditions : list
        Conditions being studied
    interventions : list
        Interventions
    sponsor : str
        Lead sponsor
    primary_outcome_measures : list
        Primary outcomes
    secondary_outcome_measures : list
        Secondary outcomes
    raw_data : dict
        Raw API response
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
    
    # -------------------------------------------------------------------------
    # Computed Properties
    # -------------------------------------------------------------------------
    
    @property
    def study_duration_months(self) -> Optional[float]:
        """Calculate study duration in months (start to primary completion)."""
        if self.start_date and self.primary_completion_date:
            delta = self.primary_completion_date - self.start_date
            return delta.days / 30.44  # Average days per month
        return None
    
    @property
    def accrual_period_months(self) -> Optional[float]:
        """
        Estimate accrual period in months.
        
        Assumes ~70% of time before primary completion is for accrual.
        This is a rough estimate - actual should come from protocol.
        """
        duration = self.study_duration_months
        return duration * 0.7 if duration else None
    
    # -------------------------------------------------------------------------
    # Methods
    # -------------------------------------------------------------------------
    
    def to_study_params(self, 
                        HR: float,
                        ctrl_median: float,
                        alpha: float = 0.05,
                        power: float = 0.8,
                        r: float = 1.0,
                        shape: float = 1.0) -> dict:
        """
        Convert to Study constructor parameters.
        
        Parameters
        ----------
        HR : float
            Hazard ratio (required - not in API)
        ctrl_median : float
            Control arm median survival in months (required)
        alpha : float
            Significance level
        power : float
            Target power
        r : float
            Allocation ratio
        shape : float
            Weibull shape parameter
            
        Returns
        -------
        dict
            Parameters for Study constructor
        """
        return {
            'N': self.enrollment or 500,
            'study_duration': self.study_duration_months or 36,
            'acc_period': self.accrual_period_months or 18,
            'ctrl_median': ctrl_median,
            'HR': HR,
            'alpha': alpha,
            'power': power,
            'r': r,
            'k': 1.0,
            'shape': shape,
            'two_sided': True
        }
    
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


# =============================================================================
# Internal Helper Functions
# =============================================================================

def _check_requests():
    """Ensure requests library is available."""
    if not HAS_REQUESTS:
        raise ImportError(
            "The 'requests' library is required for ClinicalTrials.gov integration. "
            "Install it with: pip install requests"
        )


def _clean_nct_id(nct_id: str) -> str:
    """Normalize NCT identifier."""
    nct_id = nct_id.strip().upper()
    if not nct_id.startswith('NCT'):
        nct_id = f'NCT{nct_id}'
    return nct_id


def _parse_date(date_str: Optional[str]) -> Optional[date]:
    """Parse date string from API response."""
    if not date_str:
        return None
    
    formats = ['%Y-%m-%d', '%B %d, %Y', '%B %Y', '%Y-%m']
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    
    # Try "Month Year" extraction
    parts = date_str.split()
    if len(parts) >= 2:
        try:
            return datetime.strptime(f"{parts[0]} {parts[-1]}", "%B %Y").date()
        except ValueError:
            pass
    
    return None


def _extract_trial_info(data: dict) -> TrialInfo:
    """Extract TrialInfo from API response."""
    protocol = data.get('protocolSection', {})
    
    # Module extractions
    id_module = protocol.get('identificationModule', {})
    status_module = protocol.get('statusModule', {})
    design_module = protocol.get('designModule', {})
    conditions_module = protocol.get('conditionsModule', {})
    arms_module = protocol.get('armsInterventionsModule', {})
    outcomes_module = protocol.get('outcomesModule', {})
    sponsor_module = protocol.get('sponsorCollaboratorsModule', {})
    
    # Parse dates
    start_date = _parse_date(
        status_module.get('startDateStruct', {}).get('date')
    )
    primary_completion = _parse_date(
        status_module.get('primaryCompletionDateStruct', {}).get('date')
    )
    completion_date = _parse_date(
        status_module.get('completionDateStruct', {}).get('date')
    )
    
    # Enrollment
    enrollment = design_module.get('enrollmentInfo', {}).get('count')
    
    # Interventions
    interventions = []
    for interv in arms_module.get('interventions', []):
        name = interv.get('name', '')
        itype = interv.get('type', '')
        if name:
            interventions.append(f"{itype}: {name}" if itype else name)
    
    # Outcomes
    primary_outcomes = [
        o.get('measure', '') 
        for o in outcomes_module.get('primaryOutcomes', []) 
        if o.get('measure')
    ]
    secondary_outcomes = [
        o.get('measure', '') 
        for o in outcomes_module.get('secondaryOutcomes', []) 
        if o.get('measure')
    ]
    
    # Phases
    phases = design_module.get('phases', [])
    phase_str = ', '.join(phases) if phases else 'N/A'
    
    return TrialInfo(
        nct_id=id_module.get('nctId', ''),
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
        sponsor=sponsor_module.get('leadSponsor', {}).get('name', ''),
        primary_outcome_measures=primary_outcomes,
        secondary_outcome_measures=secondary_outcomes,
        raw_data=data
    )


# =============================================================================
# Public API Functions
# =============================================================================

def fetch_trial_info(nct_id: str) -> TrialInfo:
    """
    Fetch trial information from ClinicalTrials.gov.
    
    Parameters
    ----------
    nct_id : str
        NCT identifier (e.g., 'NCT01844505' or '01844505')
        
    Returns
    -------
    TrialInfo
        Trial information
        
    Raises
    ------
    ValueError
        If trial not found or API error
    ImportError
        If requests library not installed
        
    Example
    -------
    >>> trial = fetch_trial_info('NCT01844505')
    >>> print(trial.brief_title)
    >>> print(trial.enrollment)
    """
    _check_requests()
    
    nct_id = _clean_nct_id(nct_id)
    url = f"{API_BASE_URL}/studies/{nct_id}"
    
    try:
        response = requests.get(url, timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            raise ValueError(f"Trial {nct_id} not found on ClinicalTrials.gov")
        raise ValueError(f"Error fetching trial data: {e}")
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Network error: {e}")
    
    return _extract_trial_info(response.json())


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
                          override_acc_period: Optional[float] = None,
                          verbose: bool = True) -> 'Study':
    """
    Create a Study object from ClinicalTrials.gov NCT ID.
    
    Fetches trial information and uses it to populate Study parameters.
    HR and ctrl_median must be provided as they are not in the API.
    
    Parameters
    ----------
    nct_id : str
        NCT identifier
    HR : float
        Hazard ratio to detect
    ctrl_median : float
        Control arm median survival (months)
    alpha : float
        Significance level
    power : float
        Target power
    r : float
        Allocation ratio (1:r)
    shape : float
        Weibull shape parameter
    k : float
        Accrual non-uniformity
    override_enrollment : int, optional
        Override enrollment from API
    override_duration : float, optional
        Override study duration (months)
    override_acc_period : float, optional
        Override accrual period (months)
    verbose : bool
        Print fetched info
        
    Returns
    -------
    Study
    
    Example
    -------
    >>> study = create_study_from_nct(
    ...     nct_id='NCT01844505',
    ...     HR=0.65,
    ...     ctrl_median=12
    ... )
    >>> results = study.predict(event_pred=[200, 400])
    """
    from .study import Study
    
    trial_info = fetch_trial_info(nct_id)
    
    if verbose:
        print(f"Fetched trial: {nct_id}")
        print(trial_info)
        print()
    
    # Determine values with overrides
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
    
    # Ensure acc_period < study_duration
    if acc_period >= study_duration:
        acc_period = study_duration * 0.6
        if verbose:
            print(f"Adjusted accrual period to {acc_period:.1f} months")
    
    if verbose:
        print(f"Creating Study: N={N}, duration={study_duration:.1f}mo, "
              f"accrual={acc_period:.1f}mo")
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
        Filter: 'RECRUITING', 'COMPLETED', 'ACTIVE_NOT_RECRUITING', etc.
    phase : str, optional
        Filter: 'PHASE3', 'PHASE2', 'PHASE1', etc.
    max_results : int
        Maximum results to return (default 10)
        
    Returns
    -------
    List[TrialInfo]
    
    Example
    -------
    >>> trials = search_trials('melanoma immunotherapy', status='COMPLETED', phase='PHASE3')
    >>> for trial in trials:
    ...     print(trial.brief_title)
    """
    _check_requests()
    
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
        response = requests.get(f"{API_BASE_URL}/studies", 
                                params=params, timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Error searching trials: {e}")
    
    studies = response.json().get('studies', [])
    
    results = []
    for study_data in studies[:max_results]:
        try:
            nct_id = study_data.get('protocolSection', {}).get(
                'identificationModule', {}).get('nctId', '')
            if nct_id:
                results.append(fetch_trial_info(nct_id))
        except Exception as e:
            warnings.warn(f"Error processing study: {e}")
    
    return results
