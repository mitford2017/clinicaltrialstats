"""
Streamlit dashboard for clinical trial predictions.
"""

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Import after setting up path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.clinicaltrials import ClinicalTrialsAPI
from analysis.benchmark import BenchmarkAnalyzer
from analysis.group_sequential import (
    GroupSequentialDesign,
    obrien_fleming_spending,
    pocock_spending,
)
from predictor import TrialPredictor, TrialOutcomePrediction


def run_dashboard():
    """Main entry point for the Streamlit dashboard."""
    st.set_page_config(
        page_title="Clinical Trial Predictor",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Clinical Trial Outcome Predictor")
    st.markdown("""
    Predict Phase 3 clinical trial outcomes using simulation and AI-powered benchmark analysis.
    Enter a ClinicalTrials.gov NCT number to get started.
    """)

    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")

        nct_id = st.text_input(
            "NCT Number",
            value="NCT02302807",
            help="Enter the ClinicalTrials.gov identifier (e.g., NCT02302807)"
        )

        gemini_key = st.text_input(
            "Gemini API Key (optional)",
            type="password",
            value=os.environ.get("GEMINI_API_KEY", ""),
            help="Required for AI-powered benchmark analysis"
        )

        st.subheader("Custom Parameters")

        use_custom = st.checkbox("Use custom parameters", value=False)

        custom_hr = None
        custom_median = None
        custom_alpha = 0.025
        custom_power = 0.9

        if use_custom:
            custom_hr = st.slider(
                "Hazard Ratio",
                min_value=0.3,
                max_value=1.0,
                value=0.7,
                step=0.05,
                help="Expected hazard ratio (treatment/control)"
            )

            custom_median = st.number_input(
                "Control Median (months)",
                min_value=1.0,
                max_value=60.0,
                value=12.0,
                step=0.5,
                help="Expected median survival in control arm"
            )

            custom_alpha = st.selectbox(
                "Alpha (one-sided)",
                options=[0.025, 0.01, 0.05],
                index=0,
            )

            custom_power = st.slider(
                "Power",
                min_value=0.8,
                max_value=0.95,
                value=0.9,
                step=0.05,
            )

        st.subheader("Simulation Settings")

        n_simulations = st.slider(
            "Number of Simulations",
            min_value=100,
            max_value=5000,
            value=1000,
            step=100,
        )

        spending_function = st.selectbox(
            "Alpha Spending Function",
            options=["O'Brien-Fleming", "Pocock"],
            index=0,
        )

        analyze_button = st.button("Analyze Trial", type="primary", use_container_width=True)

    # Main content area
    if analyze_button and nct_id:
        with st.spinner("Fetching trial data and running analysis..."):
            try:
                prediction = run_analysis(
                    nct_id=nct_id,
                    gemini_key=gemini_key if gemini_key else None,
                    custom_hr=custom_hr if use_custom else None,
                    custom_median=custom_median if use_custom else None,
                    n_simulations=n_simulations,
                )

                if prediction:
                    display_results(prediction)

            except Exception as e:
                st.error(f"Error analyzing trial: {str(e)}")
                st.exception(e)
    else:
        # Show example/placeholder
        st.info("Enter an NCT number and click 'Analyze Trial' to get predictions.")

        with st.expander("Example Trials to Try"):
            st.markdown("""
            - **NCT02302807**: IMvigor211 - Atezolizumab in urothelial carcinoma
            - **NCT02684006**: KEYNOTE-189 - Pembrolizumab + chemo in NSCLC
            - **NCT02578680**: PACIFIC - Durvalumab after chemoradiation in NSCLC
            - **NCT01844505**: CheckMate 067 - Nivolumab + Ipilimumab in melanoma
            """)


@st.cache_data(ttl=3600)
def run_analysis(
    nct_id: str,
    gemini_key: str | None,
    custom_hr: float | None,
    custom_median: float | None,
    n_simulations: int,
) -> TrialOutcomePrediction | None:
    """Run the trial analysis with caching."""
    predictor = TrialPredictor(
        gemini_api_key=gemini_key,
        n_simulations=n_simulations,
    )

    return predictor.predict(
        nct_id=nct_id,
        custom_hr=custom_hr,
        custom_control_median=custom_median,
    )


def display_results(prediction: TrialOutcomePrediction):
    """Display prediction results."""
    study = prediction.study
    benchmark = prediction.benchmark_analysis

    # Header with key metrics
    st.header(f"{study.brief_title}")
    st.caption(f"NCT ID: {study.nct_id} | Sponsor: {study.sponsor} | Status: {study.status}")

    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Probability of Success",
            f"{prediction.overall_probability_of_success:.1%}",
        )

    with col2:
        st.metric(
            "Expected HR",
            f"{prediction.expected_hr_at_final:.2f}",
            help="Hazard ratio used for predictions"
        )

    with col3:
        st.metric(
            "Target Events",
            f"{prediction.design.total_events}",
        )

    with col4:
        if prediction.expected_final_date:
            st.metric(
                "Expected Final Analysis",
                prediction.expected_final_date.strftime("%b %Y"),
            )

    st.divider()

    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Timeline & Predictions",
        "Hazard Ratio Analysis",
        "Group Sequential Design",
        "Simulation Results",
        "Trial Details",
    ])

    with tab1:
        display_timeline(prediction)

    with tab2:
        display_hr_analysis(prediction)

    with tab3:
        display_gsd(prediction)

    with tab4:
        display_simulations(prediction)

    with tab5:
        display_trial_details(prediction)


def display_timeline(prediction: TrialOutcomePrediction):
    """Display trial timeline predictions."""
    st.subheader("Predicted Trial Timeline")

    # Create timeline dataframe
    timeline_data = []
    for event in prediction.timeline:
        timeline_data.append({
            "Event": event.event_type.replace("_", " ").title(),
            "Date": event.predicted_date.strftime("%Y-%m-%d"),
            "Months": f"{event.calendar_months:.1f}",
            "Events": event.events_at_time,
            "Info Fraction": f"{event.information_fraction:.0%}" if event.information_fraction > 0 else "-",
            "P(Crossing)": f"{event.probability_of_crossing:.1%}" if event.probability_of_crossing > 0 else "-",
        })

    df = pd.DataFrame(timeline_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Timeline visualization
    fig = go.Figure()

    # Add events as markers
    for i, event in enumerate(prediction.timeline):
        color = {
            "enrollment_complete": "blue",
            "interim_analysis": "orange",
            "final_analysis": "green",
        }.get(event.event_type, "gray")

        fig.add_trace(go.Scatter(
            x=[event.predicted_date],
            y=[0],
            mode="markers+text",
            marker=dict(size=20, color=color),
            text=[event.event_type.replace("_", " ").title()],
            textposition="top center",
            name=event.event_type.replace("_", " ").title(),
            hovertemplate=(
                f"<b>{event.event_type.replace('_', ' ').title()}</b><br>" +
                f"Date: {event.predicted_date.strftime('%Y-%m-%d')}<br>" +
                f"Events: {event.events_at_time}<br>" +
                f"P(Crossing): {event.probability_of_crossing:.1%}<extra></extra>"
            ),
        ))

    fig.update_layout(
        title="Trial Timeline",
        xaxis_title="Date",
        yaxis_visible=False,
        showlegend=True,
        height=300,
    )

    st.plotly_chart(fig, use_container_width=True)


def display_hr_analysis(prediction: TrialOutcomePrediction):
    """Display hazard ratio trajectory analysis."""
    st.subheader("Hazard Ratio Trajectory")

    if not prediction.hr_trajectory:
        st.warning("No hazard ratio trajectory data available.")
        return

    # Create trajectory plot
    dates = [hr.calendar_date for hr in prediction.hr_trajectory]
    hrs = [hr.expected_hr for hr in prediction.hr_trajectory]
    hr_lower = [hr.hr_95_ci[0] for hr in prediction.hr_trajectory]
    hr_upper = [hr.hr_95_ci[1] for hr in prediction.hr_trajectory]
    probs = [hr.probability_of_success for hr in prediction.hr_trajectory]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=("Expected Hazard Ratio with 95% CI", "Probability of Success"),
        vertical_spacing=0.15,
    )

    # HR plot
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=hr_upper,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=hr_lower,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(0, 100, 200, 0.2)",
            name="95% CI",
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=hrs,
            mode="lines+markers",
            line=dict(color="blue", width=2),
            name="Expected HR",
        ),
        row=1, col=1,
    )

    # Reference line at HR=1
    fig.add_hline(y=1.0, line_dash="dash", line_color="red", row=1, col=1)

    # Probability plot
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=probs,
            mode="lines+markers",
            line=dict(color="green", width=2),
            name="P(Success)",
            fill="tozeroy",
            fillcolor="rgba(0, 200, 0, 0.2)",
        ),
        row=2, col=1,
    )

    fig.update_layout(height=600, showlegend=True)
    fig.update_yaxes(title_text="Hazard Ratio", row=1, col=1)
    fig.update_yaxes(title_text="Probability", range=[0, 1], row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # Data table
    with st.expander("View detailed data"):
        hr_data = []
        for hr in prediction.hr_trajectory:
            hr_data.append({
                "Date": hr.calendar_date.strftime("%Y-%m-%d"),
                "Months": f"{hr.calendar_months:.1f}",
                "Events": hr.events,
                "HR": f"{hr.expected_hr:.3f}",
                "95% CI": f"({hr.hr_95_ci[0]:.3f}, {hr.hr_95_ci[1]:.3f})",
                "Z-stat": f"{hr.z_statistic:.2f}",
                "Critical Value": f"{hr.critical_value:.2f}",
                "P(Success)": f"{hr.probability_of_success:.1%}",
            })
        st.dataframe(pd.DataFrame(hr_data), use_container_width=True, hide_index=True)


def display_gsd(prediction: TrialOutcomePrediction):
    """Display group sequential design details."""
    st.subheader("Group Sequential Design")

    design = prediction.design

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        **Design Parameters:**
        - Total Events: {design.total_events}
        - Alpha (one-sided): {design.alpha}
        - Power: {design.power:.0%}
        - Number of Analyses: {len(design.analyses)}
        """)

    with col2:
        # Required HR for this design
        required_hr = design.required_hazard_ratio(prediction.benchmark_analysis.control_median)
        st.markdown(f"""
        **Required Performance:**
        - HR needed for {design.power:.0%} power: {required_hr:.3f}
        - Expected HR: {prediction.expected_hr_at_final:.3f}
        """)

    # Analysis boundaries table
    st.markdown("### Analysis Boundaries")

    boundaries = []
    for analysis in design.analyses:
        boundaries.append({
            "Analysis": analysis.analysis_number,
            "Info Fraction": f"{analysis.information_fraction:.0%}",
            "Events": analysis.target_events,
            "Cumulative Alpha": f"{analysis.cumulative_alpha:.4f}",
            "Critical Value (Z)": f"{analysis.critical_value:.3f}",
            "Nominal p-value": f"{analysis.nominal_p_value:.4f}",
        })

    st.dataframe(pd.DataFrame(boundaries), use_container_width=True, hide_index=True)

    # Alpha spending visualization
    st.markdown("### Alpha Spending Function")

    info_fracs = np.linspace(0, 1, 100)
    of_spending = [obrien_fleming_spending(t, design.alpha) for t in info_fracs]
    pocock_sp = [pocock_spending(t, design.alpha) for t in info_fracs]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=info_fracs,
        y=of_spending,
        mode="lines",
        name="O'Brien-Fleming",
        line=dict(color="blue"),
    ))

    fig.add_trace(go.Scatter(
        x=info_fracs,
        y=pocock_sp,
        mode="lines",
        name="Pocock",
        line=dict(color="orange"),
    ))

    # Add actual analysis points
    for analysis in design.analyses:
        fig.add_trace(go.Scatter(
            x=[analysis.information_fraction],
            y=[analysis.cumulative_alpha],
            mode="markers",
            marker=dict(size=12, color="red"),
            name=f"Analysis {analysis.analysis_number}",
        ))

    fig.update_layout(
        xaxis_title="Information Fraction",
        yaxis_title="Cumulative Alpha Spent",
        height=400,
    )

    st.plotly_chart(fig, use_container_width=True)


def display_simulations(prediction: TrialOutcomePrediction):
    """Display simulation results."""
    st.subheader("Simulation Results")

    sim = prediction.simulation_summary

    if not sim:
        st.warning("No simulation results available.")
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Simulations Run", sim.get("n_simulations", 0))
        st.metric("Success Rate", f"{sim.get('success_rate', 0):.1%}")

    with col2:
        st.metric("Mean HR", f"{sim.get('mean_hr', 1):.3f}")
        st.metric("Median HR", f"{sim.get('median_hr', 1):.3f}")

    with col3:
        st.metric("HR Std Dev", f"{sim.get('hr_std', 0):.3f}")
        st.metric("Mean Time to Final", f"{sim.get('mean_final_time', 0):.1f} months")

    # HR distribution percentiles
    st.markdown("### Hazard Ratio Distribution")

    percentiles = sim.get("hr_percentiles", {})
    if percentiles:
        pct_data = pd.DataFrame([
            {"Percentile": k, "HR": f"{v:.3f}"}
            for k, v in percentiles.items()
        ])
        st.dataframe(pct_data, use_container_width=True, hide_index=True)

    # Interpretation
    st.markdown("### Interpretation")

    success_rate = sim.get("success_rate", 0)
    if success_rate >= 0.8:
        st.success(f"""
        **High probability of success ({success_rate:.0%})**

        The trial is well-powered and likely to demonstrate a statistically significant
        treatment effect if the assumed hazard ratio is achieved.
        """)
    elif success_rate >= 0.5:
        st.warning(f"""
        **Moderate probability of success ({success_rate:.0%})**

        The trial has a reasonable chance of success, but there is substantial risk
        of a negative result. Consider the confidence in the assumed treatment effect.
        """)
    else:
        st.error(f"""
        **Low probability of success ({success_rate:.0%})**

        Based on the current assumptions, the trial is more likely to fail than succeed.
        Review the hazard ratio assumptions or consider increasing sample size.
        """)


def display_trial_details(prediction: TrialOutcomePrediction):
    """Display detailed trial information."""
    study = prediction.study
    benchmark = prediction.benchmark_analysis

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Study Information")
        st.markdown(f"""
        - **Title:** {study.title}
        - **Phase:** {', '.join(study.design.phases)}
        - **Status:** {study.status}
        - **Enrollment:** {study.design.enrollment} patients
        - **Conditions:** {', '.join(study.conditions)}
        """)

        if study.dates.start_date:
            st.markdown(f"- **Start Date:** {study.dates.start_date.strftime('%Y-%m-%d')}")
        if study.dates.primary_completion_date:
            st.markdown(f"- **Primary Completion:** {study.dates.primary_completion_date.strftime('%Y-%m-%d')}")

    with col2:
        st.markdown("### Benchmark Analysis")
        st.markdown(f"""
        - **Recommended HR:** {benchmark.recommended_hr:.2f}
        - **Control Median:** {benchmark.control_median:.1f} months
        - **Expected Events:** {benchmark.expected_events}
        - **Power:** {benchmark.recommended_power:.0%}
        - **Alpha:** {benchmark.recommended_alpha}
        - **P(Success):** {benchmark.probability_of_success:.0%}
        """)

    # Study arms
    st.markdown("### Study Arms")
    for arm in study.arms:
        with st.expander(f"{arm.label} ({arm.arm_type})"):
            st.markdown(f"**Interventions:** {', '.join(arm.interventions)}")
            st.markdown(f"**Description:** {arm.description}")

    # Primary outcomes
    st.markdown("### Primary Outcomes")
    for outcome in study.primary_outcomes:
        with st.expander(outcome.title):
            st.markdown(f"**Description:** {outcome.description}")
            st.markdown(f"**Time Frame:** {outcome.time_frame}")

    # Risk factors and assumptions
    if benchmark.risk_factors:
        st.markdown("### Risk Factors")
        for risk in benchmark.risk_factors:
            st.markdown(f"- {risk}")

    if benchmark.assumptions:
        st.markdown("### Key Assumptions")
        for assumption in benchmark.assumptions:
            st.markdown(f"- {assumption}")

    # Brief summary
    with st.expander("Brief Summary"):
        st.markdown(study.brief_summary)


if __name__ == "__main__":
    run_dashboard()
