#!/usr/bin/env python3
"""
Clinical Trial Intelligence CLI.

Easy-to-use command-line interface for the intelligence engine.

Usage:
    python cli.py NCT06925737                      # Analyze a trial
    python cli.py NCT06925737 --date 2028-06-01    # Reverse solve for a date
    python cli.py NCT06925737 --scenario 0.65      # Custom HR scenario
"""

import argparse
import sys
import os
from datetime import date, datetime

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from intelligence.engine import TrialIntelligenceEngine
from intelligence.scenario_analyzer import ScenarioAnalyzer


def format_date(d):
    """Format date for display."""
    if isinstance(d, datetime):
        return d.strftime("%Y-%m-%d")
    return d.strftime("%Y-%m-%d") if d else "N/A"


def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


def analyze_trial(nct_id: str, custom_hr: float = None, n_sim: int = 500):
    """Run full intelligence analysis on a trial."""
    print(f"\nAnalyzing {nct_id}...")

    engine = TrialIntelligenceEngine(n_simulations=n_sim)

    # Build custom scenarios if HR provided
    custom_scenarios = None
    if custom_hr:
        custom_scenarios = {
            "Custom": custom_hr,
            "Bull": custom_hr * 0.85,
            "Bear": min(custom_hr * 1.2, 0.95),
        }

    report = engine.analyze(nct_id, custom_scenarios=custom_scenarios)

    # Print report
    print_header(f"INTELLIGENCE REPORT: {nct_id}")

    print(f"\n{report.trial_name}")
    print(f"Sponsor: {report.sponsor}")
    print(f"Indication: {report.indication}")
    print(f"Drug: {report.drug_name} vs {report.comparator}")
    print(f"Primary Endpoint: {report.primary_endpoint}")
    print(f"Enrollment: {report.enrollment:,} patients")
    print(f"Status: {report.status}")

    print_header("DESIGN PARAMETERS")
    print(f"  Design HR: {report.design_hr:.2f}")
    print(f"  Power: {report.power:.0%}")
    print(f"  Alpha: {report.alpha} (one-sided)")
    print(f"  Interim Events: {report.interim_events:,}")
    print(f"  Final Events: {report.final_events:,}")

    print_header("TIMELINE PREDICTIONS")
    print(f"  Trial Start: {format_date(report.start_date)}")
    print(f"\n  Interim Analysis ({report.interim_events:,} events):")
    print(f"    Model Prediction: {format_date(report.interim_date)}")
    print(f"    Range: {format_date(report.interim_date_range[0])} to {format_date(report.interim_date_range[1])}")
    print(f"    Critical HR: {report.interim_critical_hr:.3f} (need HR ≤ this for early stop)")
    print(f"\n  Final Analysis ({report.final_events:,} events):")
    print(f"    Model Prediction: {format_date(report.final_date)}")
    print(f"    Range: {format_date(report.final_date_range[0])} to {format_date(report.final_date_range[1])}")
    print(f"    Critical HR: {report.final_critical_hr:.3f}")

    # Show registry date comparison
    if report.primary_completion_date:
        print(f"\n  Registry Primary Completion: {format_date(report.primary_completion_date)}")
        model_date = report.final_date
        registry_date = report.primary_completion_date
        if model_date and registry_date:
            diff_days = (registry_date - model_date).days
            if diff_days > 0:
                print(f"    ⚠ Model is {diff_days // 30} months EARLY vs registry")
            elif diff_days < 0:
                print(f"    Model is {-diff_days // 30} months LATE vs registry")

    print_header("PROBABILITY OF SUCCESS")
    print(f"  Base Case (HR={report.design_hr:.2f}): {report.probability_of_success:.1%}")

    print_header("SCENARIO ANALYSIS")
    print(report.scenario_analysis.summary_table.to_string(index=False))

    if report.key_insights:
        print_header("KEY INSIGHTS")
        for i, insight in enumerate(report.key_insights, 1):
            print(f"  {i}. {insight}")

    if report.landscape:
        print_header("COMPETITIVE LANDSCAPE")
        print(f"  Standard of Care: {report.landscape.standard_of_care}")
        print(f"  SOC Median Survival: {report.soc_median:.1f} months")
        if report.best_in_class_hr:
            print(f"  Best-in-Class HR: {report.best_in_class_hr:.2f}")

        if report.landscape.competitors:
            print(f"\n  Key Competitors:")
            for comp in report.landscape.competitors[:5]:
                hr_str = f"HR={comp.hazard_ratio:.2f}" if comp.hazard_ratio else "HR=N/A"
                print(f"    - {comp.trial_name}: {hr_str}")

    return report


def reverse_solve(nct_id: str, target_date: date, n_sim: int = 500):
    """Solve for what to expect at a given date."""
    print(f"\nReverse solving {nct_id} for {target_date}...")

    engine = TrialIntelligenceEngine(n_simulations=n_sim)
    result = engine.reverse_solve(nct_id, target_date)

    print_header(f"REVERSE SOLVE: {nct_id}")
    print(f"Target Date: {target_date}")

    print(f"\n  Expected Events: {result.expected_events:,}")
    print(f"  Events Range (90% CI): {result.events_range[0]:,} - {result.events_range[1]:,}")
    print(f"  Maturity: {result.maturity:.1%}")
    print(f"\n  Implied HR: {result.implied_hr:.2f}")
    print(f"  Critical HR: {result.critical_hr:.3f}")
    print(f"  P(Success): {result.probability_of_crossing:.1%}")

    print(f"\n  Interpretation: {result.interpretation}")

    return result


def what_if(nct_id: str, competitor: str, competitor_hr: float, n_sim: int = 500):
    """What if this trial performs like a competitor?"""
    print(f"\nWhat if {nct_id} performs like {competitor} (HR={competitor_hr:.2f})?")

    engine = TrialIntelligenceEngine(n_simulations=n_sim)
    result = engine.what_if(nct_id, competitor, competitor_hr)

    print_header(f"WHAT-IF ANALYSIS: Like {competitor}")

    print(f"\n  HR: {result.hazard_ratio:.2f}")
    print(f"  Median Date: {format_date(result.median_date)}")
    print(f"  Date Range: {format_date(result.q05_date)} to {format_date(result.q95_date)}")
    print(f"  P(Success): {result.probability_of_success:.1%}")
    print(f"  Critical HR: {result.critical_hr:.3f}")
    print(f"\n  {result.description}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Clinical Trial Intelligence Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py NCT06925737                        # Full analysis
  python cli.py NCT06925737 --date 2028-06-01      # What to expect on date
  python cli.py NCT06925737 --scenario 0.65        # Custom HR scenario
  python cli.py NCT06925737 --like VISION --hr 0.62  # Compare to competitor
        """
    )

    parser.add_argument("nct_id", help="ClinicalTrials.gov NCT ID")
    parser.add_argument("--date", type=str, help="Target date for reverse solve (YYYY-MM-DD)")
    parser.add_argument("--scenario", type=float, help="Custom hazard ratio scenario")
    parser.add_argument("--like", type=str, help="Competitor trial name for what-if")
    parser.add_argument("--hr", type=float, help="Competitor HR (use with --like)")
    parser.add_argument("--sims", type=int, default=500, help="Number of simulations (default: 500)")

    args = parser.parse_args()

    try:
        if args.date:
            # Reverse solve mode
            target = datetime.strptime(args.date, "%Y-%m-%d").date()
            reverse_solve(args.nct_id, target, args.sims)
        elif args.like and args.hr:
            # What-if mode
            what_if(args.nct_id, args.like, args.hr, args.sims)
        else:
            # Full analysis
            analyze_trial(args.nct_id, args.scenario, args.sims)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
