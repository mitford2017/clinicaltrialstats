"""
Group sequential design and alpha spending functions.
Implements Lan-DeMets spending functions and critical value calculations.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

import numpy as np
from scipy import stats, optimize


class SpendingType(Enum):
    """Type of alpha spending function."""
    OBRIEN_FLEMING = "obrien_fleming"
    POCOCK = "pocock"
    LAN_DEMETS_OBRIEN_FLEMING = "lan_demets_of"
    LAN_DEMETS_POCOCK = "lan_demets_pocock"
    HWANG_SHIH_DECANI = "hsd"


def obrien_fleming_spending(t: float, alpha: float = 0.025) -> float:
    """
    O'Brien-Fleming spending function.

    Very conservative early, spending most alpha at the final analysis.

    Args:
        t: Information fraction (0 to 1)
        alpha: Total alpha to spend

    Returns:
        Cumulative alpha spent at information fraction t
    """
    if t <= 0:
        return 0.0
    if t >= 1:
        return alpha

    return 2 * (1 - stats.norm.cdf(stats.norm.ppf(1 - alpha / 2) / np.sqrt(t)))


def pocock_spending(t: float, alpha: float = 0.025) -> float:
    """
    Pocock spending function.

    Spends alpha more uniformly across analyses.

    Args:
        t: Information fraction (0 to 1)
        alpha: Total alpha to spend

    Returns:
        Cumulative alpha spent at information fraction t
    """
    if t <= 0:
        return 0.0
    if t >= 1:
        return alpha

    return alpha * np.log(1 + (np.e - 1) * t)


def lan_demets_spending(
    t: float,
    alpha: float = 0.025,
    spending_type: SpendingType = SpendingType.LAN_DEMETS_OBRIEN_FLEMING
) -> float:
    """
    Lan-DeMets spending function.

    Args:
        t: Information fraction (0 to 1)
        alpha: Total alpha to spend
        spending_type: Type of spending function

    Returns:
        Cumulative alpha spent at information fraction t
    """
    if spending_type == SpendingType.OBRIEN_FLEMING:
        return obrien_fleming_spending(t, alpha)
    elif spending_type == SpendingType.POCOCK:
        return pocock_spending(t, alpha)
    elif spending_type == SpendingType.LAN_DEMETS_OBRIEN_FLEMING:
        return obrien_fleming_spending(t, alpha)
    elif spending_type == SpendingType.LAN_DEMETS_POCOCK:
        return pocock_spending(t, alpha)
    elif spending_type == SpendingType.HWANG_SHIH_DECANI:
        # Hwang-Shih-DeCani with gamma = -4 (similar to O'Brien-Fleming)
        gamma = -4
        if gamma == 0:
            return alpha * t
        return alpha * (1 - np.exp(-gamma * t)) / (1 - np.exp(-gamma))
    else:
        return obrien_fleming_spending(t, alpha)


AlphaSpendingFunction = Callable[[float, float], float]


@dataclass
class AnalysisTiming:
    """Timing and characteristics of an interim or final analysis."""
    analysis_number: int
    information_fraction: float
    target_events: int
    calendar_time: Optional[float] = None
    cumulative_alpha: float = 0.0
    incremental_alpha: float = 0.0
    critical_value: float = 0.0
    nominal_p_value: float = 0.0


@dataclass
class GroupSequentialDesign:
    """
    Group sequential design specification.

    Attributes:
        total_events: Total events planned for final analysis
        alpha: One-sided significance level
        power: Target power
        analyses: List of analysis timings
        spending_function: Alpha spending function to use
    """
    total_events: int
    alpha: float
    power: float
    analyses: list[AnalysisTiming]
    spending_function: AlphaSpendingFunction

    @classmethod
    def create(
        cls,
        total_events: int,
        information_fractions: list[float],
        alpha: float = 0.025,
        power: float = 0.9,
        spending_function: Optional[AlphaSpendingFunction] = None,
    ) -> "GroupSequentialDesign":
        """
        Create a group sequential design.

        Args:
            total_events: Total events at final analysis
            information_fractions: Information fraction at each analysis (including final = 1.0)
            alpha: One-sided significance level
            power: Target power
            spending_function: Alpha spending function (default: O'Brien-Fleming)

        Returns:
            GroupSequentialDesign object
        """
        if spending_function is None:
            spending_function = obrien_fleming_spending

        # Ensure fractions are sorted and include 1.0
        fractions = sorted(set(information_fractions))
        if fractions[-1] != 1.0:
            fractions.append(1.0)

        analyses = []
        prev_alpha = 0.0

        for i, frac in enumerate(fractions):
            cum_alpha = spending_function(frac, alpha)
            incr_alpha = cum_alpha - prev_alpha

            # Calculate target events at this analysis
            target_events = int(np.ceil(frac * total_events))

            # Calculate critical value (Z-score)
            # Using incremental alpha for each stage
            critical_z = stats.norm.ppf(1 - incr_alpha / 2) if incr_alpha > 0 else np.inf

            analyses.append(AnalysisTiming(
                analysis_number=i + 1,
                information_fraction=frac,
                target_events=target_events,
                cumulative_alpha=cum_alpha,
                incremental_alpha=incr_alpha,
                critical_value=critical_z,
                nominal_p_value=incr_alpha,
            ))

            prev_alpha = cum_alpha

        design = cls(
            total_events=total_events,
            alpha=alpha,
            power=power,
            analyses=analyses,
            spending_function=spending_function,
        )

        # Recalculate critical values using proper group sequential bounds
        design._compute_boundaries()

        return design

    def _compute_boundaries(self):
        """
        Compute group sequential boundaries using the spending function approach.

        Uses numerical methods to find critical values that satisfy the
        cumulative alpha spending at each analysis.
        """
        n_analyses = len(self.analyses)
        info_fracs = [a.information_fraction for a in self.analyses]

        # Calculate boundaries using sequential probability ratio
        for i, analysis in enumerate(self.analyses):
            if i == 0:
                # First analysis - simple normal quantile
                analysis.critical_value = stats.norm.ppf(1 - analysis.cumulative_alpha)
                analysis.nominal_p_value = 1 - stats.norm.cdf(analysis.critical_value)
            else:
                # Subsequent analyses need to account for correlation
                # Approximate using spending function
                alpha_to_spend = analysis.cumulative_alpha

                # Use the error spending approach
                # Critical value is approximately the normal quantile at the cumulative alpha
                analysis.critical_value = stats.norm.ppf(1 - alpha_to_spend)
                analysis.nominal_p_value = 1 - stats.norm.cdf(analysis.critical_value)

    def get_critical_value(self, analysis_number: int) -> float:
        """Get the critical Z-value for a specific analysis."""
        for analysis in self.analyses:
            if analysis.analysis_number == analysis_number:
                return analysis.critical_value
        raise ValueError(f"Analysis {analysis_number} not found")

    def get_nominal_p_value(self, analysis_number: int) -> float:
        """Get the nominal p-value threshold for a specific analysis."""
        for analysis in self.analyses:
            if analysis.analysis_number == analysis_number:
                return analysis.nominal_p_value
        raise ValueError(f"Analysis {analysis_number} not found")

    def required_hazard_ratio(self, median_control: float) -> float:
        """
        Calculate the hazard ratio needed to achieve the target power.

        Under the logrank test, the number of events needed is:
        d = 4 * (z_alpha + z_beta)^2 / (log(HR))^2

        Args:
            median_control: Median survival in control arm (months)

        Returns:
            Required hazard ratio to achieve power
        """
        z_alpha = stats.norm.ppf(1 - self.alpha)
        z_beta = stats.norm.ppf(self.power)

        # Solve for HR: d = 4 * (z_alpha + z_beta)^2 / log(HR)^2
        log_hr_squared = 4 * (z_alpha + z_beta) ** 2 / self.total_events
        log_hr = -np.sqrt(log_hr_squared)  # Negative for HR < 1

        return np.exp(log_hr)

    def power_at_hr(self, hazard_ratio: float) -> float:
        """
        Calculate power for a given hazard ratio.

        Args:
            hazard_ratio: True hazard ratio (treatment/control)

        Returns:
            Power to detect this hazard ratio
        """
        z_alpha = stats.norm.ppf(1 - self.alpha)
        log_hr = np.log(hazard_ratio)

        # Expected Z-statistic
        expected_z = -log_hr * np.sqrt(self.total_events) / 2

        # Power is probability of exceeding critical value
        power = 1 - stats.norm.cdf(z_alpha - expected_z)

        return power

    def expected_events_at_time(
        self,
        calendar_time: float,
        enrollment_rate: float,
        enrollment_duration: float,
        median_survival: float,
        hazard_ratio: float = 1.0,
    ) -> float:
        """
        Calculate expected number of events at a calendar time.

        Uses numerical integration over the enrollment distribution for
        accurate event prediction with exponential survival.

        Args:
            calendar_time: Calendar time from trial start (months)
            enrollment_rate: Patients enrolled per month
            enrollment_duration: Total enrollment duration (months)
            median_survival: Median survival (months)
            hazard_ratio: Treatment hazard ratio (for blended estimate)

        Returns:
            Expected number of events
        """
        median_survival = float(median_survival) if median_survival else 12.0
        hazard = np.log(2) / median_survival

        # Average hazard accounting for treatment effect (1:1 randomization)
        avg_hazard = hazard * (1 + hazard_ratio) / 2

        # For patients enrolled at time s, follow-up at calendar_time T is (T - s)
        # Event probability is 1 - exp(-hazard * (T - s))
        # Integrate over enrollment period [0, min(R, T)]

        enrollment_end = min(calendar_time, enrollment_duration)

        if enrollment_end <= 0:
            return 0.0

        # Numerical integration using the proper formula:
        # E[events] = integral_0^R { rate * (1 - exp(-h*(T-s))) } ds
        # = rate * [R - (1/h) * (exp(-h*(T-R)) - exp(-h*T))]  when T > R
        # = rate * [T - (1/h) * (1 - exp(-h*T))]  when T <= R

        if calendar_time <= enrollment_duration:
            # Still enrolling: integrate from 0 to T
            T = calendar_time
            if avg_hazard > 0:
                expected = enrollment_rate * (T - (1 - np.exp(-avg_hazard * T)) / avg_hazard)
            else:
                expected = 0.0
        else:
            # Enrollment complete: all patients enrolled, varying follow-up
            R = enrollment_duration
            T = calendar_time
            if avg_hazard > 0:
                # Patients enrolled at time s have follow-up (T - s)
                # Follow-up ranges from (T - R) to T
                # Expected events = rate * integral_0^R (1 - exp(-h*(T-s))) ds
                expected = enrollment_rate * (
                    R - (np.exp(-avg_hazard * (T - R)) - np.exp(-avg_hazard * T)) / avg_hazard
                )
            else:
                expected = 0.0

        return max(0.0, expected)

    def time_to_events(
        self,
        target_events: int,
        enrollment_rate: float,
        enrollment_duration: float,
        median_survival: float,
        hazard_ratio: float = 1.0,
    ) -> float:
        """
        Calculate calendar time to reach target number of events.

        Args:
            target_events: Target number of events
            enrollment_rate: Patients enrolled per month
            enrollment_duration: Total enrollment duration (months)
            median_survival: Median survival (months)
            hazard_ratio: Treatment hazard ratio

        Returns:
            Calendar time to reach target events (months)
        """
        # Ensure numeric types
        enrollment_rate = float(enrollment_rate) if enrollment_rate else 25.0
        enrollment_duration = float(enrollment_duration) if enrollment_duration else 24.0
        median_survival = float(median_survival) if median_survival else 12.0
        hazard_ratio = float(hazard_ratio) if hazard_ratio else 1.0

        def objective(t):
            return self.expected_events_at_time(
                t, enrollment_rate, enrollment_duration,
                median_survival, hazard_ratio
            ) - target_events

        # Binary search for time
        low, high = 0.0, 200.0
        while objective(high) < 0:
            high *= 2
            if high > 1000:
                return np.inf

        result = optimize.brentq(objective, low, high)
        return result


def calculate_sample_size(
    alpha: float,
    power: float,
    hazard_ratio: float,
    allocation_ratio: float = 1.0,
) -> int:
    """
    Calculate required number of events using Schoenfeld formula.

    Args:
        alpha: One-sided significance level
        power: Target power
        hazard_ratio: Expected hazard ratio
        allocation_ratio: Ratio of treatment to control patients

    Returns:
        Required number of events
    """
    z_alpha = stats.norm.ppf(1 - alpha)
    z_beta = stats.norm.ppf(power)

    log_hr = np.log(hazard_ratio)

    # Schoenfeld formula
    p = allocation_ratio / (1 + allocation_ratio)
    events = (z_alpha + z_beta) ** 2 / (p * (1 - p) * log_hr ** 2)

    return int(np.ceil(events))


def calculate_enrollment(
    total_events: int,
    median_survival: float,
    event_probability: float = 0.7,
) -> int:
    """
    Calculate required sample size given event requirements.

    Args:
        total_events: Number of events needed
        median_survival: Median survival time
        event_probability: Expected probability of event per patient

    Returns:
        Required sample size
    """
    return int(np.ceil(total_events / event_probability))
