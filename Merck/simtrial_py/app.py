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


def render_sidebar():
    """Render sidebar with simulation settings."""
    st.sidebar.markdown("## 🔬 Simulation Settings")
    
    n_sim = st.sidebar.slider(
        "Number of Simulations",
        min_value=100, max_value=5000, value=1000, step=100,
        help="More simulations = more accurate but slower"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.markdown(
        "Simulation-based event prediction using Merck's `simtrial` methodology. "
        "Uses Monte Carlo simulation of individual patient data."
    )
    
    return n_sim


def render_parameter_inputs():
    """Render parameter input form."""
    st.markdown("### 📊 Trial Parameters")
    
    # NCT ID input for auto-fetching enrollment duration
    st.markdown("**Auto-fetch Enrollment Duration**")
    nct_col1, nct_col2 = st.columns([3, 1])
    with nct_col1:
        nct_id = st.text_input("NCT Number", value="NCT04223856", 
                               help="Enter NCT number to auto-fetch actual enrollment duration from ClinicalTrials.gov")
    with nct_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        fetch_clicked = st.button("🔍 Fetch Duration", help="Fetches actual enrollment period from trial history")
    
    # Handle fetch
    if fetch_clicked and nct_id:
        with st.spinner(f"Fetching enrollment data for {nct_id}..."):
            try:
                from recruitment_duration import get_enrollment_duration
                result = get_enrollment_duration(nct_id, headless=True)
                if result:
                    st.session_state['fetched_enrollment'] = result
                    st.success(f"✅ Found: {result['enrollment_months']} months "
                              f"({result['start_date'].strftime('%b %Y')} - {result['end_date'].strftime('%b %Y')})")
                else:
                    st.warning("⚠️ Could not determine enrollment duration. Using manual input.")
            except ImportError:
                st.error("❌ Selenium not installed. Run: pip install selenium")
            except Exception as e:
                st.error(f"❌ Error: {e}")
    
    # Use fetched value if available
    default_enroll = 42.0
    default_start = date(2020, 3, 30)
    if 'fetched_enrollment' in st.session_state:
        fetched = st.session_state['fetched_enrollment']
        default_enroll = fetched['enrollment_months']
        default_start = fetched['start_date'].date() if hasattr(fetched['start_date'], 'date') else fetched['start_date']
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Design Parameters**")
        sample_size = st.number_input("Total Enrollment (N)", min_value=50, max_value=10000, value=886)
        enroll_duration = st.number_input("Enrollment Duration (months)", min_value=1.0, max_value=120.0, 
                                         value=default_enroll,
                                         help="Auto-fetched from ClinicalTrials.gov or enter manually")
        start_date = st.date_input("Study Start Date", value=default_start)
        
        with st.expander("Sample Size Calculation (Advanced)", expanded=False):
            st.markdown("Assumptions for calculating target events:")
            design_hr = st.number_input("Design Hazard Ratio", min_value=0.1, max_value=1.0, value=0.73, step=0.01,
                                       help="Conservative HR used to calculate required sample size/events (default 0.73)")
            
    
    with col2:
        st.markdown("**Efficacy Assumptions**")
        st.markdown("#### Scenario Hazard Ratio")
        st.caption("What actually happens in the simulation")
        
        hazard_ratio = st.number_input("True Hazard Ratio", min_value=0.1, max_value=1.5, value=0.47, step=0.01,
                            help="The HR used for simulating data (e.g. 0.47)")
        
        ctrl_median = st.number_input("Control Median (months)", min_value=1.0, max_value=120.0, value=16.1,
                                      help="Median OS in control arm. EV-302 observed 16.1 months")
        
        pcd_date = st.date_input("Primary Completion Date (Est.)", value=start_date + timedelta(days=365*3),
                               help="The date when the final patient is examined or receives an intervention")

    with col3:
        st.markdown("**Statistical Parameters**")
        st.caption("Standard: 90% Power, 0.025 Alpha (1-sided)")
        power = st.number_input("Power", min_value=0.5, max_value=0.99, value=0.90)
        
        st.markdown("**Alpha Spending (One-Sided)**")
        # Set default alphas to 0.025 total
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
            st.caption("EV-302: 356 OS interim, 526 PFS final")
            interim_events = st.number_input("Interim Events", min_value=10, max_value=2000, value=356,
                                             help="EV-302 interim: 356 OS events")
            final_events = st.number_input("Final Events", min_value=10, max_value=2000, value=526,
                                           help="EV-302 final: 526 PFS events")
        else:
            st.info(f"Calculated Target: ~{calc_events} events")
            interim_events = None
            final_events = None
    
    return {
        'sample_size': int(sample_size),
        'enroll_duration': enroll_duration,
        'ctrl_median': ctrl_median,
        'hazard_ratio': hazard_ratio,
        'design_hr': design_hr,
        'start_date': start_date,
        'pcd_date': pcd_date,
        'power': power,
        'interim_alpha': interim_alpha,
        'final_alpha': final_alpha,
        'interim_events': interim_events if use_custom_events else None,
        'final_events': final_events if use_custom_events else None
    }


def plot_results(results, pcd_date=None, pcd_hr=None):
    """Plot simulation results."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    curve = results['event_curve']
    interim = results['interim']
    final = results['final']
    
    # Plot curves
    ax.plot(curve['date'], curve['events'], color='#00857C', linewidth=2.5, label='Expected Events')
    ax.plot(curve['date'], curve['enrollment'], color='#666666', linewidth=2, linestyle='--', label='Enrollment')
    
    # Interim analysis marker
    ax.axvline(x=interim['median_date'], color='#9B59B6', linestyle='--', linewidth=2, alpha=0.8)
    ax.plot(interim['median_date'], interim['events'], 's', color='#9B59B6', markersize=12, zorder=5)
    
    # Interim CI shading
    ax.axvspan(interim['q05_date'], interim['q95_date'], alpha=0.15, color='#9B59B6')
    
    # Interim annotation
    label_text = (f"Interim Analysis\n"
                  f"{interim['median_date'].strftime('%b %Y')}\n"
                  f"{interim['events']} events\n"
                  f"HR ≤ {interim['critical_hr']:.3f}")
    ax.annotate(label_text, xy=(interim['median_date'], interim['events']),
                xytext=(20, 40), textcoords='offset points',
                fontsize=10, color='#9B59B6', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#9B59B6', alpha=0.95),
                arrowprops=dict(arrowstyle='->', color='#9B59B6', alpha=0.7))
    
    # Final analysis marker
    ax.axvline(x=final['median_date'], color='#E74C3C', linestyle='--', linewidth=2, alpha=0.8)
    ax.plot(final['median_date'], final['events'], 'D', color='#E74C3C', markersize=12, zorder=5)
    
    # Final CI shading
    ax.axvspan(final['q05_date'], final['q95_date'], alpha=0.15, color='#E74C3C')
    
    # Final annotation
    label_text = (f"Final Analysis\n"
                  f"{final['median_date'].strftime('%b %Y')}\n"
                  f"{final['events']} events\n"
                  f"HR ≤ {final['critical_hr']:.3f}")
    ax.annotate(label_text, xy=(final['median_date'], final['events']),
                xytext=(20, -60), textcoords='offset points',
                fontsize=10, color='#E74C3C', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#E74C3C', alpha=0.95),
                arrowprops=dict(arrowstyle='->', color='#E74C3C', alpha=0.7))
    
    # PCD Marker
    if pcd_date:
        ax.axvline(x=pcd_date, color='#F39C12', linestyle=':', linewidth=2, alpha=0.8)
        pcd_text = f"PCD: {pcd_date.strftime('%b %Y')}"
        if pcd_hr is not None:
            pcd_text += f"\nImplied HR: {pcd_hr:.2f}"
        
        # Find y-position for PCD text (middle of plot)
        y_pos = final['events'] * 0.5
        ax.text(pcd_date, y_pos, pcd_text, color='#F39C12', fontweight='bold', 
                rotation=90, va='center', ha='right', fontsize=10)

    # Styling
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Number of Subjects/Events', fontsize=12)
    ax.set_title('Event Prediction (Simulation-Based)', fontsize=14, fontweight='bold')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    fig.autofmt_xdate()
    
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.set_xlim(curve['date'].iloc[0], None)
    ax.set_ylim(0, None)
    
    plt.tight_layout()
    return fig


def main():
    """Main application."""
    st.markdown('<p class="main-header">🧪 Trial Simulation (Merck Method)</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Simulation-based event prediction using piecewise exponential models</p>', unsafe_allow_html=True)
    
    # Sidebar
    n_sim = render_sidebar()
    
    # Parameters
    params = render_parameter_inputs()
    
    st.markdown("---")
    
    # Run simulation button
    if st.button("🚀 Run Simulation", type="primary", use_container_width=True):
        with st.spinner(f"Running {n_sim} simulations..."):
            try:
                results = run_simulation(
                    sample_size=params['sample_size'],
                    enroll_duration=params['enroll_duration'],
                    ctrl_median=params['ctrl_median'],
                    hazard_ratio=params['hazard_ratio'],
                    design_hr=params['design_hr'],
                    start_date_str=params['start_date'].isoformat(),
                    interim_alpha=params['interim_alpha'],
                    final_alpha=params['final_alpha'],
                    power=params['power'],
                    n_sim=n_sim,
                    interim_events=params.get('interim_events'),
                    final_events=params.get('final_events')
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
        
        st.markdown("### 📈 Results")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Interim Events", results['interim']['events'])
        with col2:
            st.metric("Interim Date (Median)", results['interim']['median_date'].strftime('%b %Y'))
        with col3:
            st.metric("Final Events", results['final']['events'])
        with col4:
            st.metric("Final Date (Median)", results['final']['median_date'].strftime('%b %Y'))
        
        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Plot", "📋 Analysis Details", "🧮 Reverse Engineer HR", "📐 Simulation Data"])
        
        with tab1:
            # We don't display a single implied HR on the main plot anymore since it's event-dependent
            fig = plot_results(results, pcd_date=params['pcd_date'], pcd_hr=None)
            st.pyplot(fig)
            plt.close(fig)
        
        with tab2:
            st.markdown("#### Interim Analysis")
            interim = results['interim']
            st.write(f"- **Target Events:** {interim['events']}")
            st.write(f"- **Median Date:** {interim['median_date'].strftime('%B %d, %Y')}")
            st.write(f"- **Enrollment Status:** {interim['enrolled_pct']:.1%} enrolled ({interim['enrolled_count']}/{params['sample_size']})")
            st.write(f"- **90% CI:** {interim['q05_date'].strftime('%b %Y')} - {interim['q95_date'].strftime('%b %Y')}")
            st.write(f"- **Critical HR:** ≤ {interim['critical_hr']:.4f}")
            st.write(f"- **Alpha Spend:** {interim['alpha']:.1%}")
            
            st.markdown("#### Final Analysis")
            final = results['final']
            st.write(f"- **Target Events:** {final['events']}")
            st.write(f"- **Median Date:** {final['median_date'].strftime('%B %d, %Y')}")
            st.write(f"- **Enrollment Status:** {final['enrolled_pct']:.1%} enrolled ({final['enrolled_count']}/{params['sample_size']})")
            if final['enrolled_pct'] < 1.0:
                st.warning("⚠️ Final analysis projected before enrollment completion!")
            st.write(f"- **90% CI:** {final['q05_date'].strftime('%b %Y')} - {final['q95_date'].strftime('%b %Y')}")
            st.write(f"- **Critical HR:** ≤ {final['critical_hr']:.4f}")
            st.write(f"- **Alpha Spend:** {final['alpha']:.1%}")
            
            st.markdown("#### Primary Completion Date (PCD)")
            st.write(f"- **Date:** {params['pcd_date'].strftime('%B %d, %Y')}")
            st.caption("Check the 'Reverse Engineer HR' tab to see implied HRs for this date.")

        with tab3:
            st.markdown("#### 🎯 Reverse Engineer HR from Date")
            st.caption("Calculate the Hazard Ratio required to reach a specific **Target Event Count** by a specific **Date**.")
            
            col_rev1, col_rev2, col_rev3 = st.columns(3)
            with col_rev1:
                target_date_input = st.date_input("Target Readout Date", value=params['pcd_date'])
            
            with col_rev2:
                # Default to Final Events, but allow choosing Interim or Custom
                event_options = {
                    f"Final Analysis ({results['final']['events']})": results['final']['events'],
                    f"Interim Analysis ({results['interim']['events']})": results['interim']['events'],
                    "Custom": 0
                }
                selected_event_option = st.selectbox("Target Event Count", options=list(event_options.keys()))
                
                if selected_event_option == "Custom":
                    target_events_input = st.number_input("Custom Event Count", min_value=10, max_value=params['sample_size'], value=300)
                else:
                    target_events_input = event_options[selected_event_option]
            
            if target_date_input and target_events_input:
                enroll_rate = pd.DataFrame({
                    'duration': [params['enroll_duration']],
                    'rate': [params['sample_size'] / params['enroll_duration']]
                })
                
                implied_hr_val = solve_hr_for_date(
                    target_date=target_date_input,
                    target_events=target_events_input,
                    start_date=params['start_date'],
                    sample_size=params['sample_size'],
                    enroll_rate=enroll_rate,
                    ctrl_median=params['ctrl_median']
                )
                
                with col_rev3:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if implied_hr_val:
                        st.metric("Implied Hazard Ratio", f"{implied_hr_val:.3f}")
                        
                        # Interpretation
                        if implied_hr_val > 1.2:
                            st.error("Requires HR > 1.2 (Likely Failed/Harmful)")
                        elif implied_hr_val > 0.8:
                            st.warning("Requires HR ~1.0 (No Effect)")
                        elif implied_hr_val < 0.6:
                            st.success("Requires Strong Efficacy (HR < 0.6)")
                        else:
                            st.info("Requires Moderate Efficacy")
                    else:
                        st.warning("Cannot calculate HR (Date too early or impossible)")
            
            st.markdown("""
            **Note:** 
            - If the Target Date is very early, the Implied HR may be very high (indicating events must happen very fast).
            - For EV-302, the OS Interim readout occurred with **~356-444 events** around **Aug 2023**, yielding HR ~0.47.
            """)

        with tab4:
            st.markdown("#### Simulation Summary")
            sim_data = results['simulation_results']['event_times']
            
            # Show distribution of times
            st.write(f"**{results['parameters']['n_sim']} simulations completed**")
            
            st.dataframe(sim_data.describe(), use_container_width=True)
            
            # Download
            csv = sim_data.to_csv(index=False)
            st.download_button(
                label="📥 Download Simulation Data (CSV)",
                data=csv,
                file_name="simulation_results.csv",
                mime="text/csv"
            )


if __name__ == "__main__":
    main()
