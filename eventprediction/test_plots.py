"""Quick test for plotting functions."""

import numpy as np

# Test imports
from eventprediction import (
    plot_survival_curve,
    plot_event_curve,
    plot_cumulative_events,
    plot_event_rate,
    plot_prediction_curve,
    kaplan_meier_estimate,
    fetch_trial_info,
    create_study_from_nct,
    Study
)
print('All imports successful')

# Test Study + prediction curve
study = Study(
    N=500,
    study_duration=36,
    acc_period=18,
    ctrl_median=12,
    HR=0.7,
    alpha=0.05,
    power=0.8
)
results = study.predict(event_pred=[200, 300])
n_times = len(results.grid['time'])
print(f'Study created, prediction has {n_times} time points')

# Test plot_prediction_curve
fig = plot_prediction_curve(results)
fig.savefig('test_prediction_curve.png', dpi=80)
print('plot_prediction_curve: OK')

# Test kaplan_meier_estimate
times = np.array([10, 20, 30, 40, 50, 60, 70, 80])
events = np.array([1, 0, 1, 1, 0, 1, 0, 1])
km = kaplan_meier_estimate(times, events)
print(f'kaplan_meier_estimate: OK (survival at t=0: {km.survival[0]:.2f})')

print()
print('All tests passed!')

