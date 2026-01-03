#!/usr/bin/env python3
"""
Clinical Trial Predictor - Main Application Entry Point

This module provides both CLI and Streamlit dashboard interfaces
for predicting Phase 3 clinical trial outcomes.

Usage:
    # Run Streamlit dashboard
    streamlit run app.py

    # Or use CLI
    python app.py NCT02302807 --gemini-key YOUR_KEY
"""

import argparse
import os
import sys
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_cli(args):
    """Run command-line analysis."""
    # Ensure proper path for imports
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from predictor import TrialPredictor

    print(f"\nAnalyzing trial: {args.nct_id}")
    print("=" * 50)

    predictor = TrialPredictor(
        gemini_api_key=args.gemini_key,
        n_simulations=args.simulations,
    )

    try:
        prediction = predictor.predict(
            nct_id=args.nct_id,
            custom_hr=args.hr,
            custom_control_median=args.median,
        )

        print_results(prediction)

    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


def print_results(prediction):
    """Print prediction results to console."""
    study = prediction.study
    benchmark = prediction.benchmark_analysis

    print(f"\nTrial: {study.brief_title}")
    print(f"NCT ID: {study.nct_id}")
    print(f"Status: {study.status}")
    print(f"Enrollment: {study.design.enrollment} patients")
    print(f"Conditions: {', '.join(study.conditions)}")

    print("\n" + "=" * 50)
    print("PREDICTION SUMMARY")
    print("=" * 50)

    print(f"\nProbability of Success: {prediction.overall_probability_of_success:.1%}")
    print(f"Expected Hazard Ratio: {prediction.expected_hr_at_final:.3f}")
    print(f"Predicted Final Analysis: {prediction.expected_final_date.strftime('%Y-%m-%d')}")
    if study.dates.primary_completion_date:
        print(f"Registry Primary Completion: {study.dates.primary_completion_date.strftime('%Y-%m-%d')}")

    print("\n" + "-" * 50)
    print("GROUP SEQUENTIAL DESIGN")
    print("-" * 50)

    design = prediction.design
    print(f"Total Events Required: {design.total_events}")
    print(f"Alpha (one-sided): {design.alpha}")
    print(f"Power: {design.power:.0%}")

    print("\nAnalysis Boundaries:")
    for analysis in design.analyses:
        print(f"  {analysis.analysis_number}. Info={analysis.information_fraction:.0%}, "
              f"Events={analysis.target_events}, Z-crit={analysis.critical_value:.3f}")

    print("\n" + "-" * 50)
    print("TIMELINE")
    print("-" * 50)

    for event in prediction.timeline:
        print(f"  {event.predicted_date.strftime('%Y-%m-%d')}: "
              f"{event.event_type.replace('_', ' ').title()} "
              f"(Events: {event.events_at_time}, P(Cross): {event.probability_of_crossing:.1%})")

    print("\n" + "-" * 50)
    print("SIMULATION RESULTS")
    print("-" * 50)

    sim = prediction.simulation_summary
    print(f"Simulations: {sim.get('n_simulations', 0)}")
    print(f"Success Rate: {sim.get('success_rate', 0):.1%}")
    print(f"Mean HR: {sim.get('mean_hr', 1):.3f}")
    print(f"Median HR: {sim.get('median_hr', 1):.3f}")

    print("\n" + "-" * 50)
    print("BENCHMARK ANALYSIS")
    print("-" * 50)

    print(f"Recommended HR: {benchmark.recommended_hr:.2f}")
    print(f"Control Median: {benchmark.control_median:.1f} months")
    print(f"Justification: {benchmark.hr_justification}")

    if benchmark.risk_factors:
        print("\nRisk Factors:")
        for risk in benchmark.risk_factors:
            print(f"  - {risk}")

    if benchmark.assumptions:
        print("\nAssumptions:")
        for assumption in benchmark.assumptions:
            print(f"  - {assumption}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Clinical Trial Outcome Predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run Streamlit dashboard
    streamlit run app.py

    # Analyze a specific trial
    python app.py NCT02302807

    # With custom parameters
    python app.py NCT02302807 --hr 0.7 --median 12.0 --simulations 2000

    # With Gemini API for benchmark analysis
    python app.py NCT02302807 --gemini-key YOUR_API_KEY
        """
    )

    parser.add_argument(
        "nct_id",
        nargs="?",
        help="ClinicalTrials.gov NCT ID (e.g., NCT02302807)"
    )

    parser.add_argument(
        "--gemini-key",
        default=os.environ.get("GEMINI_API_KEY"),
        help="Gemini API key for AI-powered benchmark analysis"
    )

    parser.add_argument(
        "--hr",
        type=float,
        help="Custom hazard ratio to use for predictions"
    )

    parser.add_argument(
        "--median",
        type=float,
        help="Custom control arm median survival (months)"
    )

    parser.add_argument(
        "--simulations",
        type=int,
        default=1000,
        help="Number of Monte Carlo simulations (default: 1000)"
    )

    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Launch Streamlit dashboard instead of CLI"
    )

    args = parser.parse_args()

    if args.dashboard or args.nct_id is None:
        # Check if we're being run by streamlit
        if "streamlit" in sys.modules:
            from ui.dashboard import run_dashboard
            run_dashboard()
        else:
            print("Starting Streamlit dashboard...")
            print("Run: streamlit run app.py")
            import subprocess
            subprocess.run([sys.executable, "-m", "streamlit", "run", __file__])
    else:
        run_cli(args)


if __name__ == "__main__":
    # Check if running under Streamlit (has ScriptRunContext)
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        ctx = get_script_run_ctx()
        if ctx is not None:
            from ui.dashboard import run_dashboard
            run_dashboard()
        else:
            main()
    except (ImportError, ModuleNotFoundError):
        main()
