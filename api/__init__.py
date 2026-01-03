try:
    from .clinicaltrials import ClinicalTrialsAPI, StudyData
except ImportError:
    from clinicaltrials import ClinicalTrialsAPI, StudyData

__all__ = ["ClinicalTrialsAPI", "StudyData"]
