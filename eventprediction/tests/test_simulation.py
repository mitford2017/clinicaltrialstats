"""
Tests for simulation functionality.
"""

import pytest
import numpy as np
import pandas as pd

from eventprediction import simulate, PoissonAccrual
from eventprediction.event_data import EventData_from_dataframe
from eventprediction.event_model import FromDataSimParam


class TestSimulation:
    """Tests for simulation function."""
    
    @pytest.fixture
    def sample_event_data(self):
        """Create sample EventData for testing."""
        df = pd.DataFrame({
            'subject_id': [f'S{i:03d}' for i in range(1, 51)],
            'rand_date': pd.date_range('2024-01-01', periods=50, freq='3D'),
            'has_event': [1] * 10 + [0] * 40,
            'withdrawn': [0] * 48 + [1] * 2,
            'time': [50 + i*5 for i in range(50)]
        })
        return EventData_from_dataframe(
            data=df,
            subject='subject_id',
            rand_date='rand_date',
            has_event='has_event',
            withdrawn='withdrawn',
            time='time'
        )
    
    @pytest.fixture
    def sim_params(self):
        """Create simulation parameters."""
        return FromDataSimParam(
            type_='weibull',
            rate=0.01,
            shape=1.2,
            sigma=np.zeros((2, 2))
        )
    
    def test_basic_simulation(self, sample_event_data, sim_params):
        """Test basic simulation."""
        results = simulate(
            data=sample_event_data,
            sim_params=sim_params,
            n_sim=100,
            seed=42
        )
        
        assert results is not None
        assert results.limit == 0.05
        assert len(results.event_quantiles.median) > 0
    
    def test_simulation_with_accrual(self, sample_event_data, sim_params):
        """Test simulation with additional accrual."""
        accrual = PoissonAccrual(start_date='2024-06-01', rate=0.3)
        
        results = simulate(
            data=sample_event_data,
            sim_params=sim_params,
            accrual_generator=accrual,
            n_accrual=20,
            n_sim=100,
            seed=42
        )
        
        assert results is not None
        assert results.n_accrual == 20
    
    def test_simulation_with_dropout(self, sample_event_data, sim_params):
        """Test simulation with dropout."""
        results = simulate(
            data=sample_event_data,
            sim_params=sim_params,
            n_sim=100,
            seed=42,
            dropout={'proportion': 0.1, 'time': 365}
        )
        
        assert results is not None
        assert results.dropout_rate > 0
    
    def test_simulation_reproducibility(self, sample_event_data, sim_params):
        """Test that seed provides reproducibility."""
        results1 = simulate(
            data=sample_event_data,
            sim_params=sim_params,
            n_sim=50,
            seed=42
        )
        
        results2 = simulate(
            data=sample_event_data,
            sim_params=sim_params,
            n_sim=50,
            seed=42
        )
        
        np.testing.assert_array_equal(
            results1.event_quantiles.median,
            results2.event_quantiles.median
        )
    
    def test_simulation_from_model(self, sample_event_data):
        """Test simulation from fitted model."""
        model = sample_event_data.fit(dist='weibull')
        
        results = simulate(
            model=model,
            n_sim=100,
            seed=42
        )
        
        assert results is not None
    
    def test_invalid_limit(self, sample_event_data, sim_params):
        """Test that invalid limit raises error."""
        with pytest.raises(ValueError, match="Invalid limit"):
            simulate(
                data=sample_event_data,
                sim_params=sim_params,
                n_sim=100,
                limit=0.6  # Invalid
            )
    
    def test_accrual_without_generator(self, sample_event_data, sim_params):
        """Test that n_accrual > 0 without generator raises error."""
        with pytest.raises(ValueError, match="accrual_generator required"):
            simulate(
                data=sample_event_data,
                sim_params=sim_params,
                n_sim=100,
                n_accrual=10  # No generator
            )


class TestFromDataSimParam:
    """Tests for FromDataSimParam class."""
    
    def test_create_weibull_params(self):
        """Test creating Weibull parameters."""
        params = FromDataSimParam(
            type_='weibull',
            rate=0.01,
            shape=1.5
        )
        
        assert params.type_ == 'weibull'
        assert params.rate == 0.01
        assert params.shape == 1.5
    
    def test_generate_parameters(self):
        """Test parameter generation."""
        params = FromDataSimParam(
            type_='weibull',
            rate=0.01,
            shape=1.5,
            sigma=np.eye(2) * 0.01
        )
        
        np.random.seed(42)
        sim_params = params.generate_parameters(100)
        
        assert sim_params.shape == (100, 3)
        assert np.all(sim_params[:, 0] == np.arange(1, 101))  # IDs
        assert np.all(sim_params[:, 1] > 0)  # rates
        assert np.all(sim_params[:, 2] > 0)  # shapes
    
    def test_conditional_sample_weibull(self):
        """Test conditional Weibull sampling."""
        params = FromDataSimParam(
            type_='weibull',
            rate=0.01,
            shape=1.5
        )
        
        np.random.seed(42)
        t_cond = np.array([10, 20, 30, 40, 50])
        sim_params = np.array([1, 0.01, 1.5])
        HR = np.ones(5)
        
        samples = params.conditional_sample(t_cond, sim_params, HR)
        
        # All samples should be >= conditional times
        assert np.all(samples >= t_cond)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

