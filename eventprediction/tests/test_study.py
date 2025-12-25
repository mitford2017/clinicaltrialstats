"""
Tests for Study class.
"""

import pytest
import numpy as np
from eventprediction import Study, SingleArmStudy, CRGIStudy, SingleArmCRGIStudy


class TestStudy:
    """Tests for Study class."""
    
    def test_create_two_arm_study(self):
        """Test creating a two-arm study."""
        study = Study(
            alpha=0.05,
            power=0.9,
            HR=0.7,
            r=1,
            N=500,
            study_duration=36,
            ctrl_median=12,
            k=1,
            acc_period=18,
            two_sided=True
        )
        
        assert study.N == 500
        assert study.study_duration == 36
        assert study.HR == 0.7
        assert not study.is_single_arm()
    
    def test_single_arm_study(self):
        """Test creating a single-arm study."""
        study = SingleArmStudy(
            N=200,
            study_duration=24,
            ctrl_median=10,
            k=1,
            acc_period=12
        )
        
        assert study.N == 200
        assert study.is_single_arm()
        assert np.isnan(study.HR)
    
    def test_crgi_study(self):
        """Test creating a CRGI study."""
        study = CRGIStudy(
            alpha=0.05,
            power=0.8,
            HR=0.75,
            r=1,
            N=400,
            study_duration=48,
            ctrl_time=12,
            ctrl_proportion=0.3,
            k=1,
            acc_period=24,
            two_sided=True,
            followup=36
        )
        
        assert study.N == 400
        assert study.study_type == 'CRGI'
        assert study.followup == 36
    
    def test_study_validation_invalid_shape(self):
        """Test that invalid shape raises error."""
        with pytest.raises(ValueError, match="Invalid shape"):
            Study(
                alpha=0.05,
                power=0.9,
                HR=0.7,
                r=1,
                N=500,
                study_duration=36,
                ctrl_median=12,
                k=1,
                acc_period=18,
                two_sided=True,
                shape=-1  # Invalid
            )
    
    def test_study_validation_acc_period_too_long(self):
        """Test that acc_period >= study_duration raises error."""
        with pytest.raises(ValueError, match="acc_period must be < study_duration"):
            Study(
                alpha=0.05,
                power=0.9,
                HR=0.7,
                r=1,
                N=500,
                study_duration=18,
                ctrl_median=12,
                k=1,
                acc_period=20,  # > study_duration
                two_sided=True
            )
    
    def test_study_predict(self):
        """Test study predict method."""
        study = Study(
            alpha=0.05,
            power=0.9,
            HR=0.7,
            r=1,
            N=500,
            study_duration=36,
            ctrl_median=12,
            k=1,
            acc_period=18,
            two_sided=True
        )
        
        results = study.predict(time_pred=[24, 30])
        
        assert results is not None
        assert len(results.grid['time']) > 0
        assert results.predict_data is not None


class TestSingleArmCRGIStudy:
    """Tests for SingleArmCRGIStudy."""
    
    def test_create_single_arm_crgi(self):
        """Test creating single arm CRGI study."""
        study = SingleArmCRGIStudy(
            N=150,
            study_duration=30,
            ctrl_time=12,
            ctrl_proportion=0.4,
            k=1,
            acc_period=15,
            followup=24
        )
        
        assert study.is_single_arm()
        assert study.study_type == 'CRGI'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

