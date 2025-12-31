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
    true_hr: float,
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
        true_hr=true_hr,
        design_hr=design_hr,
        start_date=start_date,
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
    
    with col2:
        st.markdown("**Efficacy Assumptions**")
        st.markdown("#### True (Observed) Values")
        st.caption("Use these to simulate *what actually happened*")
        
        true_hr = st.number_input("True Hazard Ratio (Observed)", min_value=0.1, max_value=1.5, value=0.47, step=0.01,
                            help="The actual HR observed in the trial (e.g. 0.47 for EV-302 OS)")
        ctrl_median = st.number_input("Control Median (months)", min_value=1.0, max_value=120.0, value=16.1,
                                      help="Median OS in control arm. EV-302 observed 16.1 months")
                                      
        st.markdown("#### Design Assumptions")
        st.caption("Use these to calculate *target events*")
        design_hr = st.number_input("Design Hazard Ratio", min_value=0.1, max_value=1.0, value=0.70, step=0.01,
                            help="HR assumed when designing the trial. Usually conservative (~0.70)")

    with col3:
        st.markdown("**Statistical Parameters**")
        power = st.number_input("Power", min_value=0.5, max_value=0.99, value=0.80)
        
        st.markdown("**Alpha Spending**")
        interim_alpha = st.number_input("Interim Alpha", min_value=0.001, max_value=0.1, value=0.01)
        final_alpha = st.number_input("Final Alpha", min_value=0.001, max_value=0.1, value=0.04)
        
        st.markdown("**Target Events (Protocol)**")
        use_custom_events = st.checkbox("Use protocol-specified events", value=True,
                                        help="RECOMMENDED: Use actual protocol targets, not calculated")
        if use_custom_events:
            st.caption("EV-302: 356 OS interim, 526 PFS final")
            interim_events = st.number_input("Interim Events", min_value=10, max_value=2000, value=356,
                                             help="EV-302 interim: 356 OS events")
            final_events = st.number_input("Final Events", min_value=10, max_value=2000, value=526,
                                           help="EV-302 final: 526 PFS events")
        else:
            interim_events = None
            final_events = None
    
        return {
        'sample_size': int(sample_size),
        'enroll_duration': enroll_duration,
        'ctrl_median': ctrl_median,
        'true_hr': true_hr,
        'design_hr': design_hr,
        'start_date': start_date,
        'power': power,
        'interim_alpha': interim_alpha,
        'final_alpha': final_alpha,
        'interim_events': interim_events if use_custom_events else None,
        'final_events': final_events if use_custom_events else None
    }


def plot_results(results):
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
                    true_hr=params['true_hr'],
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
        tab1, tab2, tab3 = st.tabs(["📊 Plot", "📋 Analysis Details", "📐 Simulation Data"])
        
        with tab1:
            fig = plot_results(results)
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
        
        with tab3:
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

