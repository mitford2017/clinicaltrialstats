"""
Tests for accrual generation.
"""

import pytest
import numpy as np
import pandas as pd

from eventprediction import PoissonAccrual, PowerLawAccrual


class TestPoissonAccrual:
    """Tests for Poisson accrual generator."""
    
    def test_create_poisson_accrual(self):
        """Test creating Poisson accrual generator."""
        accrual = PoissonAccrual(start_date='2024-01-01', rate=0.5)
        
        assert accrual.model == "Poisson process"
        assert "rate=0.50" in accrual.text
    
    def test_generate_subjects(self):
        """Test generating subject recruitment times."""
        np.random.seed(42)
        accrual = PoissonAccrual(start_date='2024-01-01', rate=0.5)
        
        dates = accrual.generate(100)
        
        assert len(dates) == 100
        # Dates should be in order (cumulative sum)
        assert all(dates[i] <= dates[i+1] for i in range(len(dates)-1))
    
    def test_invalid_rate(self):
        """Test that non-positive rate raises error."""
        with pytest.raises(ValueError, match="rate must be positive"):
            PoissonAccrual(start_date='2024-01-01', rate=0)


class TestPowerLawAccrual:
    """Tests for power law accrual generator."""
    
    def test_create_power_law_accrual(self):
        """Test creating power law accrual generator."""
        accrual = PowerLawAccrual(
            start_date='2024-01-01',
            end_date='2024-06-30',
            k=1.5
        )
        
        assert accrual.model == "Power law allocation"
        assert "k=1.50" in accrual.text
    
    def test_deterministic_accrual(self):
        """Test deterministic accrual."""
        accrual = PowerLawAccrual(
            start_date='2024-01-01',
            end_date='2024-06-30',
            k=1.0,
            deterministic=True
        )
        
        dates1 = accrual.generate(50)
        dates2 = accrual.generate(50)
        
        # Deterministic should give same results
        np.testing.assert_array_equal(dates1, dates2)
    
    def test_stochastic_accrual(self):
        """Test stochastic accrual generates different results."""
        np.random.seed(42)
        accrual = PowerLawAccrual(
            start_date='2024-01-01',
            end_date='2024-06-30',
            k=1.0,
            deterministic=False
        )
        
        dates1 = accrual.generate(50)
        
        np.random.seed(123)
        dates2 = accrual.generate(50)
        
        # Should be different (with very high probability)
        assert not np.array_equal(dates1, dates2)
    
    def test_uniform_accrual_k1(self):
        """Test that k=1 gives approximately uniform accrual."""
        np.random.seed(42)
        accrual = PowerLawAccrual(
            start_date='2024-01-01',
            end_date='2024-12-31',
            k=1.0,
            deterministic=True
        )
        
        dates = pd.to_datetime(accrual.generate(100))
        
        # With uniform accrual, half should be in first half of period
        mid_date = pd.to_datetime('2024-07-01')
        first_half = (dates < mid_date).sum()
        
        assert 45 <= first_half <= 55
    
    def test_invalid_dates(self):
        """Test that end_date before start_date raises error."""
        with pytest.raises(ValueError, match="Invalid arguments"):
            PowerLawAccrual(
                start_date='2024-06-30',
                end_date='2024-01-01',
                k=1.0
            )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

