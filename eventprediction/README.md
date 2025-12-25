# eventprediction

Event Prediction in Clinical Trials with Time-to-Event Outcomes

## Overview

This Python package implements methods to predict either the required number of events to achieve a target or the expected time at which you will reach the required number of events. You can use this package in the design phase of clinical trials or in the reporting phase.

This is a Python port of the R package `eventPrediction` by Daniel Dalevi, Nik Burkoff, and contributors.

## Features

- **Predict from Parameters**: Calculate expected events and times from study design parameters
- **Predict from Data**: Use accumulated trial data to predict event timing
- **Accrual Modeling**: Support for Poisson and power-law accrual patterns
- **Survival Models**: Weibull and log-logistic survival distributions
- **Flexible Study Designs**: Single-arm, two-arm, CRGI-type studies
- **Lag Effects**: Model delayed treatment effects
- **Dropout Modeling**: Include subject dropout in simulations
- **Survival Curve Plotting**: Kaplan-Meier curves with confidence intervals
- **ClinicalTrials.gov Integration**: Fetch trial info using NCT numbers

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Predict from Study Parameters

```python
from eventprediction import Study

# Create a two-arm study
study = Study(
    alpha=0.05,
    power=0.9,
    HR=0.7,
    r=1,                    # 1:1 randomization
    N=500,                  # 500 subjects
    study_duration=36,      # 36 months
    ctrl_median=12,         # Control arm median 12 months
    k=1,                    # Uniform accrual
    acc_period=18,          # 18 month accrual period
    two_sided=True
)

# Make predictions
results = study.predict(
    time_pred=[24, 30, 36],    # Predict events at these times
    event_pred=[100, 150]       # Predict times for these event counts
)

print(results)

# Plot results
results.plot()
```

### Predict from Data

```python
import pandas as pd
from eventprediction import EventData_from_dataframe, PoissonAccrual, simulate

# Load your trial data
df = pd.DataFrame({
    'subject_id': ['S001', 'S002', 'S003'],
    'rand_date': ['2024-01-15', '2024-02-20', '2024-03-10'],
    'has_event': [1, 0, 0],
    'withdrawn': [0, 0, 0],
    'time': [150, 200, 180]
})

# Create EventData object
event_data = EventData_from_dataframe(
    data=df,
    subject='subject_id',
    rand_date='rand_date',
    has_event='has_event',
    withdrawn='withdrawn',
    time='time'
)

# Fit survival model
model = event_data.fit(dist='weibull')
print(model)

# Simulate with additional recruitment
accrual = PoissonAccrual(start_date='2024-06-01', rate=0.5)

results = simulate(
    model=model,
    accrual_generator=accrual,
    n_accrual=100,
    n_sim=1000,
    seed=42
)

# Make predictions
results = results.predict(event_pred=[50, 100])
print(results.summary())

# Plot
results.plot()
```

### Single-Arm Study

```python
from eventprediction import SingleArmStudy

study = SingleArmStudy(
    N=200,
    study_duration=24,
    ctrl_median=10,
    k=1,
    acc_period=12,
    shape=1.5  # Weibull shape
)

results = study.predict(time_pred=[12, 18, 24])
print(results)
```

### Survival Curve Plotting

```python
from eventprediction import EventData_from_dataframe, plot_survival_curve

# Load your data
event_data = EventData_from_dataframe(data=df, ...)

# Plot Kaplan-Meier curve
fig = plot_survival_curve(
    event_data, 
    units='Months',
    show_ci=True,
    show_risk_table=True,
    title='Overall Survival'
)
plt.show()

# With fitted model overlay
model = event_data.fit(dist='weibull')
fig = plot_survival_curve(event_data, event_model=model)
```

### ClinicalTrials.gov Integration

```python
from eventprediction import fetch_trial_info, create_study_from_nct

# Fetch trial information by NCT ID
trial_info = fetch_trial_info('NCT01844505')
print(trial_info)
# Output:
# NCT ID: NCT01844505
# Title: Phase 3 Study of Nivolumab...
# Status: COMPLETED
# Enrollment: 945
# Start Date: 2013-06-11
# Primary Completion: 2016-08-01

# Create a Study object directly from NCT ID
study = create_study_from_nct(
    nct_id='NCT01844505',
    HR=0.65,              # Must provide - not in API
    ctrl_median=12,       # Must provide - not in API
    alpha=0.05,
    power=0.9
)

# Make predictions
results = study.predict(event_pred=[200, 400])
print(f"Critical events required: {results.critical_events_req:.0f}")
```

### CRGI-Type Study

```python
from eventprediction import CRGIStudy

study = CRGIStudy(
    alpha=0.05,
    power=0.8,
    HR=0.75,
    r=1,
    N=400,
    study_duration=48,
    ctrl_time=12,           # Time point for proportion
    ctrl_proportion=0.3,    # 30% have event by 12 months
    k=1.5,                  # Non-uniform accrual
    acc_period=24,
    two_sided=True,
    followup=36             # 36 month fixed follow-up
)

results = study.predict(event_pred=[100, 150, 200])
```

## Main Classes

### Study Classes
- `Study`: Two-arm oncology study
- `SingleArmStudy`: Single-arm oncology study  
- `CRGIStudy`: Two-arm CRGI-type study with fixed follow-up
- `SingleArmCRGIStudy`: Single-arm CRGI-type study

### Data Classes
- `EventData`: Container for clinical trial data
- `EventModel`: Fitted survival model
- `FromDataSimParam`: Parameters for data-based simulations

### Accrual Generators
- `PoissonAccrual`: Poisson process recruitment
- `PowerLawAccrual`: Power-law recruitment pattern

### Results Classes
- `AnalysisResults`: Results from parameter-based predictions
- `FromDataResults`: Results from data-based simulations

### Supporting Classes
- `LagEffect`: Delayed treatment effect modeling
- `DisplayOptions`: Output formatting options

### Plotting Functions
- `plot_survival_curve()`: Kaplan-Meier survival curves
- `plot_weibull_diagnostic()`: Weibull diagnostic plot
- `plot_events_vs_time()`: Events over time bar chart

### ClinicalTrials.gov
- `fetch_trial_info()`: Fetch trial data by NCT ID
- `create_study_from_nct()`: Create Study from NCT ID
- `search_trials()`: Search for trials
- `TrialInfo`: Trial information container

## Key Parameters

### Study Definition
- `N`: Total number of subjects
- `study_duration`: Study length in months
- `acc_period`: Accrual period in months
- `k`: Accrual non-uniformity (1=uniform, >1=accelerating, <1=decelerating)
- `shape`: Weibull shape parameter (1=exponential)
- `HR`: Hazard ratio
- `alpha`: Significance level
- `power`: Target power
- `r`: Allocation ratio (control:experimental = 1:r)

### Simulation Parameters
- `n_sim`: Number of simulations (default 10,000)
- `limit`: Confidence interval width (default 0.05 for 5%-95%)
- `seed`: Random seed for reproducibility
- `dropout`: Dropout parameters dictionary

## Contributing

This package is a Python port of the R package `eventPrediction`. Contributions are welcome.

## License

GPL (>=2)

## References

Based on the R package `eventPrediction`:
- Dalevi, Daniel; Burkoff, Nikolas; Hollis, Sally; Mann, Helen; Metcalfe, Paul; Ruau, David
- https://github.com/scientific-computing-solutions/eventPrediction

