"""
Event Prediction UI - Simulation-Based

A Streamlit interface for clinical trial event prediction using Merck's simtrial methodology.

Run with:
    cd merck/simtrial_py
    python3 -m streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import date, timedelta
from analysis import solve_hr_for_date
from trial_information_gatherer import gather_trial_info, get_ai_scenario_hr, get_ai_design_hr
import os
import json

# Try to import Google GenAI
try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

# Configure page
st.set_page_config(
    page_title="Trial Simulation (Merck Method)",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #00857C;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #5C5C5C;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #00857C 0%, #005C55 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
    }
    .stButton button {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def run_simulation(
    sample_size: int,
    enroll_duration: float,
    ctrl_median: float,
    hazard_ratio: float,
    design_hr: float,
    start_date_str: str,
    interim_alpha: float,
    final_alpha: float,
    power: float,
    n_sim: int,
    interim_events: int = None,
    final_events: int = None
):
    """Run simulation and cache results."""
    # Force reload of analysis module to ensure updates are picked up
    import sys
    if 'analysis' in sys.modules:
        import importlib
        import analysis
        importlib.reload(analysis)
    
    from analysis import predict_analysis_dates
    
    start_date = date.fromisoformat(start_date_str)
    
    # Simple linear enrollment
    enroll_rate = pd.DataFrame({
        'duration': [enroll_duration],
        'rate': [sample_size / enroll_duration]
    })
    
    results = predict_analysis_dates(
        sample_size=sample_size,
        enroll_rate=enroll_rate,
        ctrl_median=ctrl_median,
        hazard_ratio=hazard_ratio,
        start_date=start_date,
        design_hr=design_hr,
        interim_events=interim_events,
        final_events=final_events,
        interim_alpha=interim_alpha,
        final_alpha=final_alpha,
        power=power,
        n_sim=n_sim,
        seed=42
    )
    
    return results


def plot_results(results: dict, pcd_date: date = None, pcd_hr: float = None):
    """Plot simulation results."""
    # Data extraction
    interim = results['interim']
    final = results['final']
    curve = results['event_curve']
    
    # Setup plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot expected events
    ax.plot(curve['date'], curve['events'], label='Expected Events', color='#00857C', linewidth=2.5)
    
    # Plot enrollment
    ax.plot(curve['date'], curve['enrollment'], label='Enrollment', color='#7F8C8D', linestyle='--', linewidth=1.5)
    
    # Interim Analysis Marker
    ax.axvline(x=interim['median_date'], color='#9B59B6', linestyle=':', linewidth=2)
    ax.scatter([interim['median_date']], [interim['events']], color='#9B59B6', zorder=5)
    
    # Annotate Interim
    i_text = (
        f"Interim Analysis\n"
        f"{interim['median_date'].strftime('%b %Y')}\n"
        f"{int(interim['events'])} events\n"
        f"Enrolled: {interim['enrolled_pct']:.0%}"
    )
    ax.annotate(
        i_text, 
        xy=(interim['median_date'], interim['events']),
        xytext=(-10, 20), textcoords='offset points',
        ha='right', va='bottom',
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#9B59B6", alpha=0.9),
        arrowprops=dict(arrowstyle="->", color="#9B59B6")
    )
    
    # Final Analysis Marker
    ax.axvline(x=final['median_date'], color='#E74C3C', linestyle=':', linewidth=2)
    ax.scatter([final['median_date']], [final['events']], color='#E74C3C', zorder=5)
    
    # Annotate Final
    f_text = (
        f"Final Analysis\n"
        f"{final['median_date'].strftime('%b %Y')}\n"
        f"{int(final['events'])} events\n"
        f"Enrolled: {final['enrolled_pct']:.0%}"
    )
    ax.annotate(
        f_text, 
        xy=(final['median_date'], final['events']),
        xytext=(-10, 20), textcoords='offset points',
        ha='right', va='bottom',
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#E74C3C", alpha=0.9),
        arrowprops=dict(arrowstyle="->", color="#E74C3C")
    )

    # Reference PCD
    if pcd_date:
        ax.axvline(x=pcd_date, color='black', linestyle='-', alpha=0.3, linewidth=1)
        ax.text(pcd_date, 0, "PCD (Est.)", rotation=90, verticalalignment='bottom', alpha=0.5)

    # Formatting
    ax.set_title("Projected Events & Enrollment Over Time", fontsize=14, pad=15)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.2)
    
    # Date formatting
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    fig.autofmt_xdate()
    
    st.pyplot(fig)
    
    # Metrics Table
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📊 Interim Analysis")
        st.write(f"**Date:** {interim['median_date'].strftime('%B %Y')}")
        st.write(f"**Events:** {int(interim['events'])}")
        st.write(f"**Enrollment:** {int(interim['enrolled_count'])} ({interim['enrolled_pct']:.1%})")
        st.write(f"**Critical HR:** {interim['critical_hr']:.3f}")
        st.caption(f"90% CI: {interim['q05_date'].strftime('%b %Y')} - {interim['q95_date'].strftime('%b %Y')}")

    with col2:
        st.markdown("### 🏁 Final Analysis")
        st.write(f"**Date:** {final['median_date'].strftime('%B %Y')}")
        st.write(f"**Events:** {int(final['events'])}")
        st.write(f"**Enrollment:** {int(final['enrolled_count'])} ({final['enrolled_pct']:.1%})")
        st.write(f"**Critical HR:** {final['critical_hr']:.3f}")
        st.caption(f"90% CI: {final['q05_date'].strftime('%b %Y')} - {final['q95_date'].strftime('%b %Y')}")


def render_sidebar():
    """Render sidebar with simulation settings."""
    st.sidebar.markdown("## 🔬 Simulation Settings")
    
    n_sim = st.sidebar.slider(
        "Number of Simulations",
        min_value=100, max_value=5000, value=1000, step=100,
        help="More simulations = more accurate but slower"
    )
    
    st.sidebar.markdown("---")
    
    if st.sidebar.button("🔄 Reset / New Trial"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    st.sidebar.markdown("### About")
    st.sidebar.markdown(
        "Simulation-based event prediction using Merck's `simtrial` methodology. "
        "Uses Monte Carlo simulation of individual patient data."
    )
    
    return n_sim


def landing_page():
    """Initial input page."""
    st.markdown('<p class="main-header">🧪 Clinical Trial Event Prediction</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Enter trial details to gather design parameters and run simulations.</p>', unsafe_allow_html=True)
    
    with st.container():
        col1, col2 = st.columns([2, 1])
        
        with col1:
            nct_id = st.text_input("ClinicalTrials.gov ID", placeholder="e.g. NCT04223856")
            api_key = os.environ.get("GEMINI_API_KEY")
            
            if st.button("🔍 Gather Information", type="primary"):
                if not nct_id:
                    st.error("Please enter an NCT ID.")
                    return
                
                # Check format
                if not nct_id.upper().startswith("NCT"):
                    st.error("Invalid ID format. Must start with 'NCT'.")
                    return
                
                if not api_key:
                    st.warning("⚠️ No API key provided. AI features will be disabled.")
                
                with st.status("Gathering trial information...", expanded=True) as status:
                    st.write("📡 Fetching from ClinicalTrials.gov API...")
                    st.write("🕸️ Scraping enrollment history (may take 30-60s)...")
                    if api_key:
                        st.write("🤖 Calling Gemini AI for HR suggestions...")
                    
                    trial_info = gather_trial_info(nct_id, api_key)
                    
                    st.write("---")
                    for log in trial_info['logs']:
                        st.write(log)
                    
                    # Show what was found
                    if 'scenario_hr' in trial_info:
                        st.write(f"**Scenario HR:** {trial_info['scenario_hr']}")
                        st.write(f"**Rationale:** {trial_info.get('scenario_hr_rationale', 'N/A')}")
                    
                    st.session_state['trial_info'] = trial_info
                    if api_key:
                        st.session_state['gemini_api_key'] = api_key
                    
                    status.update(label="Information gathered!", state="complete", expanded=False)
                
                st.rerun()

        with col2:
            st.info("""
            **What happens next?**
            1. We scrape **Enrollment Duration** from history.
            2. AI analyzes the protocol for **Design HR** & **Sample Size**.
            3. AI estimates **True/Scenario HR** from results/benchmarks.
            4. You review and run the simulation.
            """)


def simulation_ui():
    """Main simulation dashboard."""
    trial_info = st.session_state.get('trial_info', {})
    nct_id = trial_info.get('nct_id', 'Unknown Trial')
    ct_data = trial_info.get('ct_data', None)
    
    st.markdown(f'<p class="main-header">🧪 Simulation: {nct_id}</p>', unsafe_allow_html=True)
    
    # Sidebar
    n_sim = render_sidebar()
    
    # Defaults from gathered info
    defaults = {
        'sample_size': 0,
        'enroll_duration': 0.0,
        'start_date': date.today(),
        'design_hr': 0.73,
        'scenario_hr': 0.70,
        'ctrl_median': 0.0,
        'pcd_date': None
    }
    
    # Apply gathered data
    if 'enrollment' in trial_info and trial_info['enrollment']:
        e = trial_info['enrollment']
        defaults['enroll_duration'] = e.get('enrollment_months', 0.0)
        if e.get('start_date'):
            defaults['start_date'] = e['start_date'].date()
            
    if 'design_hr' in trial_info:
        defaults['design_hr'] = trial_info['design_hr']
        
    if 'scenario_hr' in trial_info:
        defaults['scenario_hr'] = trial_info['scenario_hr']
        
    if 'ai_info' in trial_info and trial_info['ai_info']:
        ai = trial_info['ai_info']
        if ai.get('sample_size'): defaults['sample_size'] = ai['sample_size']
        if ai.get('control_median'): defaults['ctrl_median'] = ai['control_median']
        if ai.get('pcd_date'): 
            try:
                defaults['pcd_date'] = date.fromisoformat(ai['pcd_date'])
            except: pass

    # --- Parameter Inputs ---
    st.markdown("### 📊 Trial Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Design Parameters**")
        
        # Sample Size
        ss_val = defaults['sample_size']
        if ss_val == 0:
            st.warning("Sample size not found. Please enter manually.")
            ss_val = 500
        sample_size = st.number_input("Total Enrollment (N)", min_value=50, max_value=10000, value=ss_val)
        
        # Enrollment Duration
        enroll_val = float(defaults['enroll_duration'])
        if enroll_val == 0.0:
            st.warning("Enrollment duration not found. Please enter manually.")
            enroll_val = 24.0
            
        enroll_help = "Auto-fetched from ClinicalTrials.gov"
        if 'enrollment' in trial_info and trial_info['enrollment'].get('is_estimate'):
            enroll_help = f"AI Estimate: {trial_info['enrollment'].get('estimate_rationale', 'Based on start/PCD')}"
            st.info(f"ℹ️ {enroll_help}")
            
        enroll_duration = st.number_input("Enrollment Duration (months)", min_value=1.0, max_value=120.0, 
                                         value=enroll_val,
                                         help=enroll_help)
        start_date = st.date_input("Study Start Date", value=defaults['start_date'])
        
        with st.expander("Sample Size Calculation (Advanced)", expanded=False):
            st.markdown("Assumptions for calculating target events:")
            design_hr = st.number_input("Design Hazard Ratio", min_value=0.1, max_value=1.0, value=defaults['design_hr'], step=0.01,
                                       help="Conservative HR used to calculate required sample size/events")
            
            if 'design_hr_rationale' in trial_info:
                st.info(f"**AI Rationale:** {trial_info['design_hr_rationale']}")

    with col2:
        st.markdown("**Efficacy Assumptions**")
        st.markdown("#### Scenario Hazard Ratio")
        st.caption("What actually happens in the simulation")
        
        # Get HR and rationale from session state or trial_info
        scenario_hr_val = trial_info.get('scenario_hr', 0.70)
        scenario_rationale = trial_info.get('scenario_hr_rationale', None)
        
        # Override with session state if re-suggested
        if 'suggested_scenario_hr' in st.session_state:
            scenario_hr_val = st.session_state['suggested_scenario_hr']
        if 'scenario_hr_rationale' in st.session_state:
            scenario_rationale = st.session_state['scenario_hr_rationale']
            
        hazard_ratio = st.number_input("True Hazard Ratio", min_value=0.1, max_value=1.5, value=float(scenario_hr_val), step=0.01)
        
        # Always show rationale box
        if scenario_rationale:
            st.info(f"**AI Rationale:** {scenario_rationale}")
        else:
            st.warning("No AI rationale available. Enter API key and click 'Re-Suggest' below.")
            
        # Re-query AI button
        if HAS_GENAI and 'gemini_api_key' in st.session_state:
            if st.button("✨ Re-Suggest Scenario HR"):
                with st.spinner("Asking Gemini 3.0 Pro..."):
                    hr_val, rat = get_ai_scenario_hr(st.session_state['gemini_api_key'], nct_id, ct_data)
                    if hr_val:
                        st.session_state['suggested_scenario_hr'] = hr_val
                        st.session_state['scenario_hr_rationale'] = rat
                        st.rerun()
                    else:
                        st.error(f"AI call failed: {rat}")
        
        # Control Median
        cm_val = defaults['ctrl_median']
        if cm_val == 0.0:
            st.warning("Control median not found. Please enter manually.")
            cm_val = 12.0 # Generic fallback
            
        ctrl_median = st.number_input("Control Median (months)", min_value=1.0, max_value=120.0, value=cm_val)
        pcd_date = st.date_input("Primary Completion Date (Est.)", value=defaults['pcd_date'])

    with col3:
        st.markdown("**Statistical Parameters**")
        st.caption("Standard: 90% Power, 0.025 Alpha (1-sided)")
        power = st.number_input("Power", min_value=0.5, max_value=0.99, value=0.90)
        
        interim_alpha = st.number_input("Interim Alpha", min_value=0.001, max_value=0.05, value=0.005, format="%.3f")
        final_alpha = st.number_input("Final Alpha", min_value=0.001, max_value=0.05, value=0.020, format="%.3f")
        
        st.markdown("**Target Events**")
        # Calculate expected events based on Schoenfeld using DESIGN HR
        from scipy.stats import norm
        z_alpha = norm.ppf(1 - (interim_alpha + final_alpha))
        z_beta = norm.ppf(power)
        calc_events = int(((1 + 1) * (z_alpha + z_beta) / (np.sqrt(1) * np.log(design_hr))) ** 2)
        
        use_custom_events = st.checkbox("Use protocol-specified events", value=True,
                                        help=f"Uncheck to use calculated events (~{calc_events} based on Design HR {design_hr})")
        
        if use_custom_events:
            if 'ai_info' in trial_info and trial_info['ai_info'].get('interim_events'):
                def_interim = trial_info['ai_info']['interim_events']
                def_final = trial_info['ai_info']['final_events'] or 500
            else:
                def_interim = 356
                def_final = 526
                
            interim_events = st.number_input("Interim Events", min_value=10, max_value=2000, value=def_interim)
            final_events = st.number_input("Final Events", min_value=10, max_value=2000, value=def_final)
        else:
            st.info(f"Calculated Target: ~{calc_events} events")
            interim_events = None
            final_events = None

    st.markdown("---")
    
    # Run simulation button
    if st.button("🚀 Run Simulation", type="primary", use_container_width=True):
        with st.spinner(f"Running {n_sim} simulations..."):
            try:
                results = run_simulation(
                    sample_size=sample_size,
                    enroll_duration=enroll_duration,
                    ctrl_median=ctrl_median,
                    hazard_ratio=hazard_ratio,
                    design_hr=design_hr,
                    start_date_str=start_date.isoformat(),
                    interim_alpha=interim_alpha,
                    final_alpha=final_alpha,
                    power=power,
                    n_sim=n_sim,
                    interim_events=interim_events,
                    final_events=final_events
                )
                
                st.session_state['results'] = results
                st.success(f"✅ Completed {n_sim} simulations!")
                
            except Exception as e:
                st.error(f"❌ Error: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    # Display results
    if 'results' in st.session_state:
        results = st.session_state['results']
        plot_results(results, pcd_date=pcd_date, pcd_hr=None)


def main():
    if 'trial_info' not in st.session_state:
        landing_page()
    else:
        simulation_ui()

if __name__ == "__main__":
    main()
