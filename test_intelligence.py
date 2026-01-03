#!/usr/bin/env python3
"""Test script for the Clinical Trial Intelligence Engine."""

import os
import sys
from datetime import date

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from intelligence.engine import TrialIntelligenceEngine, quick_intelligence
from intelligence.scenario_analyzer import ScenarioAnalyzer


def test_scenario_analyzer():
    """Test the scenario analyzer without Gemini."""
    print("=" * 60)
    print("Testing ScenarioAnalyzer (no API needed)")
    print("=" * 60)

    analyzer = ScenarioAnalyzer()

    # Test parameters (approximate values for a trial like IDeate-Prostate01)
    sample_size = 1000
    enroll_duration = 24  # months
    ctrl_median = 14.0    # months (typical for mCRPC)
    start_date = date(2025, 5, 1)
    final_events = 350
    design_hr = 0.75

    # Define scenarios
    scenarios = {
        "Bull (Best-in-Class)": 0.62,
        "Base Case": 0.75,
        "Bear (Modest)": 0.85,
    }

    print(f"\nTrial Parameters:")
    print(f"  Sample Size: {sample_size}")
    print(f"  Enrollment Duration: {enroll_duration} months")
    print(f"  Control Median: {ctrl_median} months")
    print(f"  Start Date: {start_date}")
    print(f"  Final Events Target: {final_events}")
    print(f"  Design HR: {design_hr}")

    # Run multi-scenario analysis
    print("\nRunning multi-scenario analysis...")
    result = analyzer.multi_scenario_analysis(
        sample_size=sample_size,
        enroll_duration=enroll_duration,
        ctrl_median=ctrl_median,
        start_date=start_date,
        final_events=final_events,
        design_hr=design_hr,
        scenarios=scenarios,
        n_sim=500,
    )

    print("\n" + "=" * 60)
    print("SCENARIO ANALYSIS RESULTS")
    print("=" * 60)
    print(result.summary_table.to_string(index=False))

    print("\n--- Scenario Details ---")
    for scenario in result.scenarios:
        print(f"\n{scenario.scenario_name}:")
        print(f"  HR: {scenario.hazard_ratio:.2f}")
        print(f"  Median Date: {scenario.median_date}")
        print(f"  Date Range: {scenario.q05_date} to {scenario.q95_date}")
        print(f"  P(Success): {scenario.probability_of_success:.1%}")
        print(f"  Critical HR: {scenario.critical_hr:.3f}")

    # Test reverse solve
    print("\n" + "=" * 60)
    print("REVERSE SOLVE TEST")
    print("=" * 60)

    target_date = date(2028, 6, 1)
    print(f"\nIf readout on {target_date}, what to expect?")

    reverse = analyzer.reverse_solve(
        target_date=target_date,
        start_date=start_date,
        sample_size=sample_size,
        enroll_duration=enroll_duration,
        ctrl_median=ctrl_median,
        final_events=final_events,
    )

    print(f"\n  Expected Events: {reverse.expected_events}")
    print(f"  Events Range (90% CI): {reverse.events_range}")
    print(f"  Implied HR: {reverse.implied_hr:.2f}")
    print(f"  Critical HR: {reverse.critical_hr:.3f}")
    print(f"  Maturity: {reverse.maturity:.1%}")
    print(f"  P(Crossing): {reverse.probability_of_crossing:.1%}")
    print(f"\n  Interpretation: {reverse.interpretation}")

    return result


def test_full_intelligence(nct_id: str = "NCT06925737"):
    """Test full intelligence engine (requires Gemini API key)."""
    print("\n" + "=" * 60)
    print(f"Testing Full Intelligence Engine for {nct_id}")
    print("=" * 60)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("\nNote: GEMINI_API_KEY not set. Running without competitive research.")
        print("Set GEMINI_API_KEY environment variable for full functionality.")

    try:
        engine = TrialIntelligenceEngine(
            gemini_api_key=api_key,
            n_simulations=500,
        )

        print(f"\nAnalyzing {nct_id}...")
        report = engine.analyze(nct_id)

        print("\n" + "=" * 60)
        print("INTELLIGENCE REPORT")
        print("=" * 60)

        print(f"\nTRIAL: {report.trial_name}")
        print(f"Sponsor: {report.sponsor}")
        print(f"Indication: {report.indication}")
        print(f"Drug: {report.drug_name}")
        print(f"Comparator: {report.comparator}")
        print(f"Primary Endpoint: {report.primary_endpoint}")
        print(f"Enrollment: {report.enrollment}")
        print(f"Status: {report.status}")

        print(f"\nDESIGN PARAMETERS:")
        print(f"  Design HR: {report.design_hr:.2f}")
        print(f"  Power: {report.power:.0%}")
        print(f"  Alpha: {report.alpha}")
        print(f"  Interim Events: {report.interim_events}")
        print(f"  Final Events: {report.final_events}")

        print(f"\nTIMELINE PREDICTIONS:")
        print(f"  Start Date: {report.start_date}")
        print(f"  Interim Analysis: {report.interim_date}")
        print(f"    Range: {report.interim_date_range[0]} to {report.interim_date_range[1]}")
        print(f"  Final Analysis: {report.final_date}")
        print(f"    Range: {report.final_date_range[0]} to {report.final_date_range[1]}")

        print(f"\nCRITICAL THRESHOLDS:")
        print(f"  Interim Critical HR: {report.interim_critical_hr:.3f}")
        print(f"  Final Critical HR: {report.final_critical_hr:.3f}")

        print(f"\nPROBABILITY OF SUCCESS: {report.probability_of_success:.1%}")

        print(f"\nSCENARIO ANALYSIS:")
        print(report.scenario_analysis.summary_table.to_string(index=False))

        print(f"\nKEY INSIGHTS:")
        for i, insight in enumerate(report.key_insights, 1):
            print(f"  {i}. {insight}")

        if report.landscape:
            print(f"\nCOMPETITIVE LANDSCAPE:")
            print(f"  Standard of Care: {report.landscape.standard_of_care}")
            print(f"  SOC Median Survival: {report.soc_median:.1f} months")
            if report.best_in_class_hr:
                print(f"  Best-in-Class HR: {report.best_in_class_hr:.2f}")
            print(f"  Competitors: {len(report.landscape.competitors)}")
            for comp in report.landscape.competitors[:3]:
                hr_str = f"HR={comp.hazard_ratio:.2f}" if comp.hazard_ratio else "HR=N/A"
                print(f"    - {comp.trial_name}: {hr_str}")

        return report

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_reverse_solve_for_trial(nct_id: str = "NCT06925737"):
    """Test reverse solve for a specific trial."""
    print("\n" + "=" * 60)
    print(f"Testing Reverse Solve for {nct_id}")
    print("=" * 60)

    try:
        engine = TrialIntelligenceEngine(n_simulations=100)

        # Test a few different dates
        test_dates = [
            date(2027, 6, 1),
            date(2028, 1, 1),
            date(2028, 6, 1),
            date(2029, 1, 1),
        ]

        for target in test_dates:
            print(f"\nIf readout on {target}:")
            result = engine.reverse_solve(nct_id, target)
            print(f"  Events: {result.expected_events} ({result.events_range[0]}-{result.events_range[1]})")
            print(f"  Maturity: {result.maturity:.1%}")
            print(f"  Critical HR: {result.critical_hr:.3f}")
            print(f"  P(Success): {result.probability_of_crossing:.1%}")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Test scenario analyzer (no API needed)
    test_scenario_analyzer()

    # Test full intelligence with trial
    print("\n")
    test_full_intelligence("NCT06925737")

    # Test reverse solve
    print("\n")
    test_reverse_solve_for_trial("NCT06925737")
