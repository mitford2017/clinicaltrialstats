"""
EventData class for clinical trial data.
"""

from dataclasses import dataclass, field
from typing import Optional, Union, List
import numpy as np
import pandas as pd
from datetime import date, datetime
import warnings

from .utils import fix_dates, standarddaysinyear, average_rec


@dataclass
class EventData:
    """
    Class representing data to use for predictions.
    
    Attributes
    ----------
    subject_data : pd.DataFrame
        DataFrame with columns: subject, rand_date, has_event, withdrawn, 
        censored_at_follow_up, time, site, event_type
    followup : float
        Fixed follow up period in days (inf if no fixed followup)
    """
    subject_data: pd.DataFrame
    followup: float = np.inf
    
    def __post_init__(self):
        self._validate()
    
    def _validate(self):
        """Validate the event data."""
        required_cols = ['has_event', 'rand_date', 'withdrawn', 'subject', 
                         'time', 'censored_at_follow_up']
        
        data = self.subject_data
        
        if len(data) > 0:
            for col in required_cols:
                if col not in data.columns:
                    raise ValueError(f"Missing required column: {col}")
            
            if not data['has_event'].isin([0, 1]).all():
                raise ValueError("has_event must be 0 or 1")
            if not data['withdrawn'].isin([0, 1]).all():
                raise ValueError("withdrawn must be 0 or 1")
            if not data['censored_at_follow_up'].isin([0, 1]).all():
                raise ValueError("censored_at_follow_up must be 0 or 1")
            
            if data['time'].isna().all() or (data['time'] == 0).all():
                raise ValueError("time value incorrect - all are 0 or NA!")
            
            if (data['time'] < 0).any() or data['time'].isna().any():
                bad_subjects = data.loc[data['time'] < 0 | data['time'].isna(), 'subject']
                raise ValueError(f"subjects cannot have non-positive time: {list(bad_subjects)}")
            
            if data['subject'].duplicated().any():
                raise ValueError("subject ID must be unique")
            
            if (data['has_event'] == 0).all():
                warnings.warn("No events have occurred - a model cannot be fit")
            if (data['has_event'] == 1).all():
                raise ValueError("All events have occurred!")
            
            # Check for subjects both withdrawn and having event
            both = (data['has_event'] == 1) & (data['withdrawn'] == 1)
            if both.any():
                bad_subjects = data.loc[both, 'subject']
                raise ValueError(f"subjects cannot be both withdrawn and have an event: {list(bad_subjects)}")
        
        if not isinstance(self.followup, (int, float)) or self.followup <= 0:
            raise ValueError("Invalid followup argument")
    
    @property
    def n_subjects(self) -> int:
        """Number of subjects."""
        return len(self.subject_data)
    
    @property
    def n_events(self) -> int:
        """Number of events."""
        return int(self.subject_data['has_event'].sum())
    
    @property
    def n_withdrawn(self) -> int:
        """Number of withdrawn subjects."""
        return int(self.subject_data['withdrawn'].sum())
    
    def summary(self) -> str:
        """Return a summary string of the data."""
        daysinyear = standarddaysinyear()
        df = self.subject_data
        
        if len(df) == 0:
            return "Empty data frame!"
        
        lines = []
        lines.append(f"Number of subjects: {len(df)}")
        lines.append(f"Number of events: {self.n_events}")
        
        if 'event_type' in df.columns and df['event_type'].notna().any():
            event_types = df.loc[df['has_event'] == 1, 'event_type'].value_counts()
            for etype, count in event_types.items():
                lines.append(f"  Of type {etype}: {count}")
        
        lines.append(f"Number of withdrawn: {self.n_withdrawn}")
        lines.append(f"First subject randomized: {df['rand_date'].min()}")
        lines.append(f"Last subject randomized: {df['rand_date'].max()}")
        
        if self.n_events > 0:
            event_data = df[df['has_event'] == 1]
            last_dates = event_data['rand_date'] + pd.to_timedelta(event_data['time'] - 1, unit='D')
            lines.append(f"First Event: {last_dates.min()}")
            lines.append(f"Last Event: {last_dates.max()}")
        
        av = average_rec(len(df), 
                         df['rand_date'].min().toordinal(),
                         df['rand_date'].max().toordinal())
        lines.append(f"Average recruitment (subjects/day): {av:.2f}")
        
        if np.isfinite(self.followup):
            lines.append(f"Subjects followed for {round(self.followup)} days "
                         f"({self.followup/daysinyear:.2f} years)")
            lines.append(f"Number of subjects censored at end of follow up period: "
                         f"{int(df['censored_at_follow_up'].sum())}")
        
        return '\n'.join(lines)
    
    def __str__(self) -> str:
        return self.summary()
    
    def fit(self, dist: str = 'weibull') -> 'EventModel':
        """
        Fit a survival model to the data.
        
        Parameters
        ----------
        dist : str
            Distribution type ('weibull' or 'loglogistic')
            
        Returns
        -------
        EventModel
        """
        from .event_model import EventModel
        return EventModel.from_event_data(self, dist=dist)
    
    def calculate_days_at_risk(self) -> float:
        """Calculate total days at risk."""
        return float(self.subject_data['time'].sum())
    
    def only_use_rec_times(self) -> 'EventData':
        """Create new EventData with only recruitment times (reset time to 0)."""
        new_data = self.subject_data.copy()
        new_data['withdrawn'] = 0
        new_data['has_event'] = 0
        new_data['censored_at_follow_up'] = 0
        new_data['time'] = 0
        new_data['event_type'] = pd.NA
        
        # Return a new object without validation since time=0
        result = object.__new__(EventData)
        result.subject_data = new_data
        result.followup = self.followup
        return result


def EventData_from_dataframe(data: pd.DataFrame,
                              subject: str,
                              rand_date: str,
                              has_event: str,
                              withdrawn: str,
                              time: Union[str, dict],
                              site: Optional[str] = None,
                              event_type: Optional[str] = None,
                              remove_0_time: bool = False,
                              followup: float = np.inf) -> EventData:
    """
    Constructor for EventData from a DataFrame.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data
    subject : str
        Column name for subject identifiers
    rand_date : str
        Column name for randomization dates
    has_event : str
        Column name for event indicator (1=event, 0=no event)
    withdrawn : str
        Column name for withdrawal indicator (1=withdrawn, 0=not)
    time : str or dict
        Column name for time on study, or dict with date columns to derive time
    site : str, optional
        Column name for site
    event_type : str, optional
        Column name for event type
    remove_0_time : bool
        If True, remove subjects with time=0 or NA
    followup : float
        Follow up period in days (inf if no fixed followup)
        
    Returns
    -------
    EventData
    """
    # Validate columns exist
    required = [subject, rand_date, has_event, withdrawn]
    if isinstance(time, str):
        required.append(time)
    
    for col in required:
        if col not in data.columns:
            raise ValueError(f"Column name {col} not found in data frame")
    
    if followup <= 0:
        raise ValueError("Invalid followup argument")
    
    # Fix dates
    data = data.copy()
    data[rand_date] = fix_dates(data[rand_date])
    
    # Get time column
    if isinstance(time, str):
        time_values = data[time].astype(float)
    else:
        # Derive time from other columns
        time_values = _add_time_column(data, rand_date, has_event, withdrawn, subject, time)
    
    # Get site
    if site is not None:
        site_values = data[site]
    else:
        site_values = pd.Series([pd.NA] * len(data))
    
    # Get event type
    if event_type is not None:
        event_type_values = data[event_type].astype('category')
        event_type_values = event_type_values.cat.remove_unused_categories()
    else:
        event_type_values = pd.Series(
            np.where(data[has_event] == 1, 'Has Event', pd.NA),
            dtype='category'
        )
    
    # Create subject data
    subject_data = pd.DataFrame({
        'subject': data[subject],
        'rand_date': data[rand_date],
        'time': time_values,
        'has_event': data[has_event].astype(int),
        'withdrawn': data[withdrawn].astype(int),
        'site': site_values,
        'event_type': event_type_values
    })
    
    # Remove subjects with invalid rand_date or missing subject ID
    valid_mask = (subject_data['subject'].astype(str) != '') & subject_data['rand_date'].notna()
    if not valid_mask.all():
        invalid = subject_data.loc[~valid_mask, 'subject'].tolist()
        warnings.warn(f"Subjects {invalid} removed due to invalid rand_date or missing subject ID.")
        subject_data = subject_data[valid_mask]
    
    # Handle time=0 or NA
    zero_time_mask = (subject_data['time'] == 0) | subject_data['time'].isna()
    if zero_time_mask.any():
        zero_subjects = subject_data.loc[zero_time_mask, 'subject'].tolist()
        if remove_0_time:
            warnings.warn(f"Subjects {zero_subjects} have time=NA or 0 and have been removed.")
            subject_data = subject_data[~zero_time_mask]
        else:
            warnings.warn(f"Subjects {zero_subjects} have time=NA or 0. Their time has been set to 0.")
            subject_data.loc[zero_time_mask, 'time'] = 0
    
    # Include follow up
    subject_data = _include_followup(subject_data, followup)
    
    # Validation checks with warnings
    # Event with no event type
    mask = (subject_data['has_event'] == 1) & subject_data['event_type'].isna()
    if mask.any():
        warnings.warn(f"Subjects {subject_data.loc[mask, 'subject'].tolist()} have event but no event type. Set to 'Has Event'")
        subject_data.loc[mask, 'event_type'] = 'Has Event'
    
    # Event type but no event
    mask = (subject_data['has_event'] != 1) & subject_data['event_type'].notna()
    if mask.any():
        warnings.warn(f"Subjects {subject_data.loc[mask, 'subject'].tolist()} have event type but no event")
        subject_data.loc[mask, 'event_type'] = pd.NA
    
    # Both event and withdrawn
    mask = (subject_data['has_event'] == 1) & (subject_data['withdrawn'] == 1)
    if mask.any():
        warnings.warn(f"Subjects {subject_data.loc[mask, 'subject'].tolist()} have event and are withdrawn. Assuming event.")
        subject_data.loc[mask, 'withdrawn'] = 0
    
    return EventData(subject_data=subject_data.reset_index(drop=True), followup=followup)


def _include_followup(subject_data: pd.DataFrame, followup: float) -> pd.DataFrame:
    """Set censored_at_follow_up column and adjust times."""
    subject_data = subject_data.copy()
    
    subject_data['censored_at_follow_up'] = (subject_data['time'] > followup).astype(int)
    subject_data['time'] = np.minimum(subject_data['time'], followup)
    
    # Reset subjects censored at followup
    mask = (subject_data['withdrawn'] == 1) & (subject_data['censored_at_follow_up'] == 1)
    if mask.any():
        warnings.warn(f"Subjects {subject_data.loc[mask, 'subject'].tolist()} withdrew after followup, censored at followup")
        subject_data.loc[mask, 'withdrawn'] = 0
        subject_data.loc[mask, 'has_event'] = 0
        subject_data.loc[mask, 'event_type'] = pd.NA
    
    mask = (subject_data['has_event'] == 1) & (subject_data['censored_at_follow_up'] == 1)
    if mask.any():
        warnings.warn(f"Subjects {subject_data.loc[mask, 'subject'].tolist()} had event after followup, censored at followup")
        subject_data.loc[mask, 'has_event'] = 0
        subject_data.loc[mask, 'event_type'] = pd.NA
    
    return subject_data


def _add_time_column(data: pd.DataFrame, rand_date: str, has_event: str,
                     withdrawn: str, subject: str, time_dict: dict) -> pd.Series:
    """Derive time column from date columns."""
    # Convert date columns
    for col_key, col_name in time_dict.items():
        if col_name in data.columns:
            data[col_key] = fix_dates(data[col_name])
    
    times = pd.Series([np.nan] * len(data), dtype=float)
    
    # For each subject, calculate time based on their status
    for i in range(len(data)):
        row = data.iloc[i]
        
        # Get the most appropriate date
        event_date = None
        for key in ['dth_date', 'event_date', 'prog_date', 'last_date']:
            if key in time_dict and time_dict[key] in data.columns:
                val = row.get(key)
                if pd.notna(val):
                    event_date = val
                    break
        
        if event_date is not None and pd.notna(row[rand_date]):
            times.iloc[i] = (event_date - row[rand_date]).days + 1
    
    return times


def EmptyEventData(followup: float = np.inf) -> EventData:
    """Create an empty EventData object."""
    if followup <= 0:
        raise ValueError("Invalid followup argument")
    
    empty_df = pd.DataFrame({
        'subject': pd.Series([], dtype=str),
        'rand_date': pd.Series([], dtype='datetime64[ns]'),
        'time': pd.Series([], dtype=float),
        'has_event': pd.Series([], dtype=int),
        'withdrawn': pd.Series([], dtype=int),
        'site': pd.Series([], dtype=object),
        'event_type': pd.Series([], dtype='category'),
        'censored_at_follow_up': pd.Series([], dtype=int)
    })
    
    result = object.__new__(EventData)
    result.subject_data = empty_df
    result.followup = followup
    return result


def CutData(event_data: EventData, cut_date: Union[str, date, datetime]) -> EventData:
    """
    Cut the event data at a given date.
    
    Creates new EventData showing how data would have looked on the given date.
    
    Parameters
    ----------
    event_data : EventData
        Original data
    cut_date : str or date
        Date to cut at
        
    Returns
    -------
    EventData
    """
    cut_date = fix_dates(cut_date)
    if isinstance(cut_date, pd.Series):
        cut_date = cut_date.iloc[0]
    
    subject_data = event_data.subject_data.copy()
    
    # Only include subjects randomized by this time
    subject_data = subject_data[subject_data['rand_date'] <= cut_date]
    
    if len(subject_data) == 0:
        raise ValueError("Cut date before first subject randomization")
    
    # Calculate censored times
    censored_times = (cut_date - subject_data['rand_date']).dt.days + 1
    
    # Find subjects whose time needs to be cut
    idx = censored_times < subject_data['time']
    
    if not idx.any() and len(subject_data) == len(event_data.subject_data):
        warnings.warn("Cut date is too late (no ongoing subjects) and has no effect")
    
    # Update times and status
    subject_data['time'] = np.minimum(censored_times, subject_data['time'])
    subject_data.loc[idx, 'has_event'] = 0
    subject_data.loc[idx, 'event_type'] = pd.NA
    subject_data.loc[idx, 'withdrawn'] = 0
    
    return EventData_from_dataframe(
        data=subject_data,
        subject='subject',
        rand_date='rand_date',
        has_event='has_event',
        withdrawn='withdrawn',
        time='time',
        site='site' if 'site' in subject_data.columns else None,
        event_type='event_type' if 'event_type' in subject_data.columns else None,
        followup=event_data.followup
    )

