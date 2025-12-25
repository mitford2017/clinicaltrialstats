"""
Tests for EventData class.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import date

from eventprediction import EventData, EmptyEventData, CutData
from eventprediction.event_data import EventData_from_dataframe


class TestEventData:
    """Tests for EventData class."""
    
    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for testing."""
        return pd.DataFrame({
            'subject_id': ['S001', 'S002', 'S003', 'S004', 'S005'],
            'rand_date': ['2024-01-15', '2024-02-20', '2024-03-10', '2024-04-05', '2024-05-01'],
            'has_event': [1, 0, 1, 0, 0],
            'withdrawn': [0, 0, 0, 1, 0],
            'time': [150, 200, 180, 100, 120]
        })
    
    def test_create_event_data(self, sample_df):
        """Test creating EventData from DataFrame."""
        event_data = EventData_from_dataframe(
            data=sample_df,
            subject='subject_id',
            rand_date='rand_date',
            has_event='has_event',
            withdrawn='withdrawn',
            time='time'
        )
        
        assert event_data.n_subjects == 5
        assert event_data.n_events == 2
        assert event_data.n_withdrawn == 1
    
    def test_empty_event_data(self):
        """Test creating empty EventData."""
        event_data = EmptyEventData()
        
        assert event_data.n_subjects == 0
        assert event_data.n_events == 0
    
    def test_empty_event_data_with_followup(self):
        """Test empty EventData with followup."""
        event_data = EmptyEventData(followup=365)
        
        assert event_data.followup == 365
    
    def test_empty_event_data_invalid_followup(self):
        """Test that invalid followup raises error."""
        with pytest.raises(ValueError, match="Invalid followup"):
            EmptyEventData(followup=-1)
    
    def test_calculate_days_at_risk(self, sample_df):
        """Test days at risk calculation."""
        event_data = EventData_from_dataframe(
            data=sample_df,
            subject='subject_id',
            rand_date='rand_date',
            has_event='has_event',
            withdrawn='withdrawn',
            time='time'
        )
        
        days_at_risk = event_data.calculate_days_at_risk()
        expected = 150 + 200 + 180 + 100 + 120
        
        assert days_at_risk == expected
    
    def test_only_use_rec_times(self, sample_df):
        """Test resetting to only recruitment times."""
        event_data = EventData_from_dataframe(
            data=sample_df,
            subject='subject_id',
            rand_date='rand_date',
            has_event='has_event',
            withdrawn='withdrawn',
            time='time'
        )
        
        rec_only = event_data.only_use_rec_times()
        
        assert rec_only.n_events == 0
        assert rec_only.n_withdrawn == 0
        assert (rec_only.subject_data['time'] == 0).all()
    
    def test_fit_weibull(self, sample_df):
        """Test fitting Weibull model."""
        event_data = EventData_from_dataframe(
            data=sample_df,
            subject='subject_id',
            rand_date='rand_date',
            has_event='has_event',
            withdrawn='withdrawn',
            time='time'
        )
        
        model = event_data.fit(dist='weibull')
        
        assert model is not None
        assert model.rate > 0
        assert model.shape > 0
    
    def test_summary(self, sample_df):
        """Test summary method."""
        event_data = EventData_from_dataframe(
            data=sample_df,
            subject='subject_id',
            rand_date='rand_date',
            has_event='has_event',
            withdrawn='withdrawn',
            time='time'
        )
        
        summary = event_data.summary()
        
        assert 'Number of subjects: 5' in summary
        assert 'Number of events: 2' in summary


class TestCutData:
    """Tests for CutData function."""
    
    @pytest.fixture
    def event_data(self):
        """Create EventData for testing."""
        df = pd.DataFrame({
            'subject_id': ['S001', 'S002', 'S003'],
            'rand_date': ['2024-01-01', '2024-02-01', '2024-03-01'],
            'has_event': [1, 0, 0],
            'withdrawn': [0, 0, 0],
            'time': [60, 90, 60]
        })
        return EventData_from_dataframe(
            data=df,
            subject='subject_id',
            rand_date='rand_date',
            has_event='has_event',
            withdrawn='withdrawn',
            time='time'
        )
    
    def test_cut_data(self, event_data):
        """Test cutting data at a date."""
        cut_data = CutData(event_data, '2024-03-15')
        
        # Should include first two subjects fully, third partially
        assert cut_data.n_subjects == 3
    
    def test_cut_data_before_first_subject(self, event_data):
        """Test that cutting before first subject raises error."""
        with pytest.raises(ValueError, match="Cut date before first subject"):
            CutData(event_data, '2023-12-01')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

