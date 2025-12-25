"""
EventModel class for fitted survival models.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

from .utils import standarddaysinyear


@dataclass
class FromDataSimParam:
    """
    Parameters for simulating conditional survival times.
    
    Attributes
    ----------
    type_ : str
        Distribution type ('weibull' or 'loglogistic')
    rate : float
        Rate parameter
    shape : float
        Shape parameter
    sigma : np.ndarray
        Covariance matrix for uncertainty
    """
    type_: str
    rate: float
    shape: float
    sigma: np.ndarray = field(default_factory=lambda: np.zeros((2, 2)))
    
    def generate_parameters(self, n_sim: int) -> np.ndarray:
        """
        Generate parameters for simulations.
        
        Parameters
        ----------
        n_sim : int
            Number of simulations
            
        Returns
        -------
        np.ndarray
            Array with columns [Id, rate, shape]
        """
        w_scale = np.log(1 / self.rate)
        w_shape = 1 / self.shape
        
        # Sample from multivariate normal
        mean = np.array([w_scale, np.log(w_shape)])
        samples = np.random.multivariate_normal(mean, self.sigma, size=n_sim)
        
        # Convert to standard parameters
        rates = np.exp(-samples[:, 0])
        shapes = np.exp(-samples[:, 1])
        
        return np.column_stack([np.arange(1, n_sim + 1), rates, shapes])
    
    def conditional_sample(self, t_conditional: np.ndarray, 
                           params: np.ndarray, 
                           HR: np.ndarray) -> np.ndarray:
        """
        Sample from conditional distribution.
        
        Parameters
        ----------
        t_conditional : np.ndarray
            Current survival times
        params : np.ndarray
            Parameters [Id, rate, shape]
        HR : np.ndarray
            Hazard ratios for each subject
            
        Returns
        -------
        np.ndarray
            Sampled survival times
        """
        rate = params[1]
        shape = params[2]
        
        if self.type_ == 'weibull':
            return self._rcweibull(t_conditional, rate, shape, HR)
        else:
            return self._rcloglogistic(t_conditional, rate, shape, HR)
    
    @staticmethod
    def _rcweibull(t_conditional: np.ndarray, rate: float, shape: float,
                   HR: np.ndarray) -> np.ndarray:
        """Sample from conditional Weibull distribution."""
        rate_adj = rate * (HR ** (1 / shape))
        t_cond_scaled = t_conditional * rate_adj
        
        exp_samples = np.random.exponential(1, len(t_conditional))
        return ((t_cond_scaled ** shape + exp_samples) ** (1 / shape)) / rate_adj
    
    @staticmethod
    def _rcloglogistic(t_conditional: np.ndarray, rate: float, shape: float,
                       HR: np.ndarray) -> np.ndarray:
        """Sample from conditional log-logistic distribution."""
        if not np.allclose(HR, 1):
            raise ValueError("Cannot use HR argument with loglogistic model")
        
        ret_val = 1 + (t_conditional * rate) ** shape
        ret_val = ret_val / np.random.uniform(0, 1, len(t_conditional)) - 1
        return (ret_val ** (1 / shape)) / rate


@dataclass
class EventModel:
    """
    A fitted survival model.
    
    Attributes
    ----------
    rate : float
        Estimated rate parameter
    shape : float
        Estimated shape parameter
    sigma : np.ndarray
        Covariance matrix
    dist : str
        Distribution type
    event_data : EventData
        Original data
    sim_params : FromDataSimParam
        Simulation parameters
    """
    rate: float
    shape: float
    sigma: np.ndarray
    dist: str
    event_data: 'EventData'
    sim_params: FromDataSimParam = field(init=False)
    
    def __post_init__(self):
        self.sim_params = FromDataSimParam(
            type_=self.dist,
            rate=self.rate,
            shape=self.shape,
            sigma=self.sigma
        )
    
    @classmethod
    def from_event_data(cls, event_data: 'EventData', dist: str = 'weibull') -> 'EventModel':
        """
        Fit a survival model to EventData.
        
        Parameters
        ----------
        event_data : EventData
            Data to fit
        dist : str
            Distribution type ('weibull' or 'loglogistic')
            
        Returns
        -------
        EventModel
        """
        if dist not in ['weibull', 'loglogistic']:
            raise ValueError("dist must be 'weibull' or 'loglogistic'")
        
        if len(event_data.subject_data) == 0:
            raise ValueError("Empty data frame!")
        
        if event_data.n_events == 0:
            raise ValueError("Cannot fit a model to a dataset with no events")
        
        # Get data for fitting (ignore time=0)
        data = event_data.subject_data.copy()
        data.loc[data['time'] == 0, 'time'] = np.nan
        
        times = data['time'].values
        events = data['has_event'].values
        
        # Fit the model using maximum likelihood
        result = cls._fit_survreg(times, events, dist)
        
        return cls(
            rate=result['rate'],
            shape=result['shape'],
            sigma=result['sigma'],
            dist=dist,
            event_data=event_data
        )
    
    @staticmethod
    def _fit_survreg(times: np.ndarray, events: np.ndarray, dist: str) -> dict:
        """
        Fit a parametric survival model using maximum likelihood.
        
        Parameters
        ----------
        times : np.ndarray
            Survival/censoring times
        events : np.ndarray
            Event indicators (1=event, 0=censored)
        dist : str
            Distribution type
            
        Returns
        -------
        dict
            {'rate': rate, 'shape': shape, 'sigma': covariance_matrix}
        """
        # Remove missing values
        valid = ~np.isnan(times) & (times > 0)
        t = times[valid]
        d = events[valid]
        
        if dist == 'weibull':
            return EventModel._fit_weibull(t, d)
        else:
            return EventModel._fit_loglogistic(t, d)
    
    @staticmethod
    def _fit_weibull(t: np.ndarray, d: np.ndarray) -> dict:
        """Fit Weibull distribution."""
        n = len(t)
        
        def neg_log_lik(params):
            mu = params[0]  # log(scale)
            sigma = np.exp(params[1])  # log(shape)
            
            # Weibull parameterization: scale = exp(mu), shape = 1/sigma
            scale = np.exp(mu)
            shape = 1 / sigma
            
            # Log-likelihood
            ll = 0
            for i in range(n):
                z = (np.log(t[i]) - mu) / sigma
                if d[i] == 1:  # Event
                    ll += -np.log(sigma) - np.log(t[i]) + z - np.exp(z)
                else:  # Censored
                    ll += -np.exp(z)
            
            return -ll
        
        # Initial values
        log_t = np.log(t[t > 0])
        init_mu = np.mean(log_t)
        init_sigma = np.log(np.std(log_t))
        
        result = minimize(neg_log_lik, [init_mu, init_sigma], method='BFGS')
        
        mu = result.x[0]
        log_sigma = result.x[1]
        sigma = np.exp(log_sigma)
        
        # Convert to rate/shape
        scale = np.exp(mu)
        shape = 1 / sigma
        rate = 1 / scale
        
        # Estimate covariance from Hessian inverse
        try:
            from scipy.optimize import approx_fprime
            eps = 1e-5
            hess = np.zeros((2, 2))
            for i in range(2):
                def grad_i(x):
                    return approx_fprime(x, neg_log_lik, eps)[i]
                hess[i, :] = approx_fprime(result.x, grad_i, eps)
            cov = np.linalg.inv(hess)
        except Exception:
            cov = np.zeros((2, 2))
        
        return {'rate': rate, 'shape': shape, 'sigma': cov}
    
    @staticmethod
    def _fit_loglogistic(t: np.ndarray, d: np.ndarray) -> dict:
        """Fit log-logistic distribution."""
        n = len(t)
        
        def neg_log_lik(params):
            mu = params[0]
            sigma = np.exp(params[1])
            
            ll = 0
            for i in range(n):
                z = (np.log(t[i]) - mu) / sigma
                if d[i] == 1:  # Event
                    ll += -np.log(sigma) - np.log(t[i]) + z - 2 * np.log(1 + np.exp(z))
                else:  # Censored
                    ll += -np.log(1 + np.exp(z))
            
            return -ll
        
        # Initial values
        log_t = np.log(t[t > 0])
        init_mu = np.mean(log_t)
        init_sigma = np.log(np.std(log_t))
        
        result = minimize(neg_log_lik, [init_mu, init_sigma], method='BFGS')
        
        mu = result.x[0]
        sigma = np.exp(result.x[1])
        
        # Convert to rate/shape
        scale = np.exp(mu)
        shape = 1 / sigma
        rate = 1 / scale
        
        # Estimate covariance
        try:
            from scipy.optimize import approx_fprime
            eps = 1e-5
            hess = np.zeros((2, 2))
            for i in range(2):
                def grad_i(x):
                    return approx_fprime(x, neg_log_lik, eps)[i]
                hess[i, :] = approx_fprime(result.x, grad_i, eps)
            cov = np.linalg.inv(hess)
        except Exception:
            cov = np.zeros((2, 2))
        
        return {'rate': rate, 'shape': shape, 'sigma': cov}
    
    def predict_quantiles(self, p: np.ndarray) -> np.ndarray:
        """
        Predict survival times at given quantiles.
        
        Parameters
        ----------
        p : np.ndarray
            Quantiles (probabilities)
            
        Returns
        -------
        np.ndarray
            Survival times
        """
        if self.dist == 'weibull':
            # Weibull quantile function: t = scale * (-log(1-p))^(1/shape)
            scale = 1 / self.rate
            return scale * (-np.log(1 - p)) ** (1 / self.shape)
        else:
            # Log-logistic quantile function
            scale = 1 / self.rate
            return scale * (p / (1 - p)) ** (1 / self.shape)
    
    def __str__(self) -> str:
        lines = [f"Fitted {self.dist} survival model"]
        lines.append(f"Rate: {self.rate:.6f}")
        lines.append(f"Shape: {self.shape:.6f}")
        lines.append(f"Median survival: {self.predict_quantiles(np.array([0.5]))[0]:.2f}")
        return '\n'.join(lines)

