"""
Event Prediction UI

A Streamlit interface for clinical trial event prediction.

Run with:
    streamlit run eventprediction/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date

# Configure page
st.set_page_config(
    page_title="Event Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# Custom CSS
# =============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #5C5C5C;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E3A5F;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Imports (lazy to avoid startup errors)
# =============================================================================

@st.cache_resource
def load_package():
    """Load eventprediction package."""
    from eventprediction import (
        Study, 
        fetch_trial_info, 
        create_study_from_nct,
        plot_prediction_curve,
        TrialInfo
    )
    return {
        'Study': Study,
        'fetch_trial_info': fetch_trial_info,
        'create_study_from_nct': create_study_from_nct,
        'plot_prediction_curve': plot_prediction_curve,
    }


# =============================================================================
# Sidebar
# =============================================================================

def render_sidebar():
    """Render sidebar with input mode selection."""
    st.sidebar.markdown("## ⚙️ Input Mode")
    
    mode = st.sidebar.radio(
        "Choose input method:",
        ["📋 Manual Parameters", "🔍 From NCT Number"],
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.markdown(
        "Predict events in time-to-event clinical trials. "
        "Based on the R package `eventPrediction`."
    )
    
    return mode


# =============================================================================
# NCT Lookup Section
# =============================================================================

def render_nct_section(pkg):
    """Render NCT number lookup section."""
    st.markdown("### 🔍 Fetch Trial from ClinicalTrials.gov")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        nct_id = st.text_input(
            "NCT Number",
            value="NCT03924895",
            placeholder="e.g., NCT03924895",
            help="Enter an NCT identifier to fetch trial details"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        fetch_btn = st.button("🔎 Fetch Trial Info", type="primary", use_container_width=True)
    
    trial_info = None
    
    if fetch_btn and nct_id:
        with st.spinner("Fetching trial data..."):
            try:
                trial_info = pkg['fetch_trial_info'](nct_id)
                st.session_state['trial_info'] = trial_info
                st.success(f"✅ Found: {trial_info.brief_title[:80]}...")
            except Exception as e:
                st.error(f"❌ Error: {e}")
    
    # Show cached trial info
    if 'trial_info' in st.session_state:
        trial_info = st.session_state['trial_info']
        
        with st.expander("📄 Trial Details", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Enrollment", trial_info.enrollment or "N/A")
                st.metric("Phase", trial_info.phase or "N/A")
            
            with col2:
                st.metric("Start Date", str(trial_info.start_date) if trial_info.start_date else "N/A")
                st.metric("Status", trial_info.status or "N/A")
            
            with col3:
                st.metric("Primary Completion", str(trial_info.primary_completion_date) if trial_info.primary_completion_date else "N/A")
                duration = trial_info.study_duration_months
                st.metric("Duration (months)", f"{duration:.1f}" if duration else "N/A")
            
            if trial_info.conditions:
                st.markdown(f"**Conditions:** {', '.join(trial_info.conditions[:5])}")
            
            if trial_info.sponsor:
                st.markdown(f"**Sponsor:** {trial_info.sponsor}")
    
    return trial_info


# =============================================================================
# Parameter Input Section
# =============================================================================

def render_parameter_inputs(trial_info=None):
    """Render parameter input form."""
    st.markdown("### 📊 Study Parameters")
    
    # Use trial info defaults if available
    default_n = trial_info.enrollment if trial_info and trial_info.enrollment else 500
    default_duration = trial_info.study_duration_months if trial_info and trial_info.study_duration_months else 36
    default_accrual = trial_info.accrual_period_months if trial_info and trial_info.accrual_period_months else 18
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Design Parameters**")
        N = st.number_input("Total Enrollment (N)", min_value=10, max_value=10000, value=int(default_n))
        study_duration = st.number_input("Study Duration (months)", min_value=1.0, max_value=120.0, value=float(default_duration))
        acc_period = st.number_input("Accrual Period (months)", min_value=1.0, max_value=120.0, value=float(default_accrual))
    
    with col2:
        st.markdown("**Efficacy Assumptions**")
        HR = st.number_input("Hazard Ratio", min_value=0.1, max_value=1.5, value=0.75, step=0.05,
                            help="HR < 1 means treatment is better")
        ctrl_median = st.number_input("Control Median (months)", min_value=1.0, max_value=120.0, value=12.0,
                                      help="Median survival in control arm")
        shape = st.number_input("Weibull Shape", min_value=0.1, max_value=5.0, value=1.0,
                               help="1.0 = exponential distribution")
    
    with col3:
        st.markdown("**Statistical Parameters**")
        alpha = st.number_input("Alpha (significance)", min_value=0.001, max_value=0.2, value=0.05)
        power = st.number_input("Power", min_value=0.5, max_value=0.99, value=0.80)
        two_sided = st.checkbox("Two-sided test", value=True)
        r = st.number_input("Allocation Ratio", min_value=0.1, max_value=10.0, value=1.0,
                           help="1:r allocation (1.0 = 1:1)")
    
    return {
        'N': int(N),
        'study_duration': study_duration,
        'acc_period': acc_period,
        'HR': HR,
        'ctrl_median': ctrl_median,
        'shape': shape,
        'alpha': alpha,
        'power': power,
        'two_sided': two_sided,
        'r': r
    }


# =============================================================================
# Prediction Section
# =============================================================================

def render_prediction_section(pkg, params):
    """Render prediction controls and results."""
    st.markdown("### 🎯 Predictions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        event_pred_str = st.text_input(
            "Predict time for event counts",
            value="150, 200, 300",
            help="Comma-separated event counts"
        )
    
    with col2:
        time_pred_str = st.text_input(
            "Predict events at times (months)",
            value="12, 24, 36",
            help="Comma-separated times in months"
        )
    
    # Parse inputs
    event_pred = None
    time_pred = None
    
    if event_pred_str.strip():
        try:
            event_pred = [int(x.strip()) for x in event_pred_str.split(',') if x.strip()]
        except:
            st.warning("Invalid event counts format")
    
    if time_pred_str.strip():
        try:
            time_pred = [float(x.strip()) for x in time_pred_str.split(',') if x.strip()]
        except:
            st.warning("Invalid time format")
    
    run_btn = st.button("🚀 Run Prediction", type="primary", use_container_width=True)
    
    if run_btn:
        with st.spinner("Running prediction..."):
            try:
                # Create study
                study = pkg['Study'](
                    N=params['N'],
                    study_duration=params['study_duration'],
                    acc_period=params['acc_period'],
                    ctrl_median=params['ctrl_median'],
                    HR=params['HR'],
                    alpha=params['alpha'],
                    power=params['power'],
                    r=params['r'],
                    shape=params['shape'],
                    two_sided=params['two_sided']
                )
                
                # Run prediction
                results = study.predict(
                    event_pred=event_pred,
                    time_pred=time_pred
                )
                
                st.session_state['results'] = results
                st.session_state['study'] = study
                st.success("✅ Prediction complete!")
                
            except Exception as e:
                st.error(f"❌ Error: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    # Display results
    if 'results' in st.session_state:
        render_results(pkg, st.session_state['results'])


def render_results(pkg, results):
    """Render prediction results."""
    st.markdown("---")
    st.markdown("### 📈 Results")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Required Events", f"{results.critical_events_req:.0f}" if not np.isnan(results.critical_events_req) else "N/A")
    
    with col2:
        if results.critical_data and 'time' in results.critical_data and len(results.critical_data['time']) > 0:
            st.metric("Time to Required Events", f"{results.critical_data['time'][0]:.1f} months")
        else:
            st.metric("Time to Required Events", "N/A")
    
    with col3:
        st.metric("Average HR", f"{results.av_hr:.3f}" if not np.isnan(results.av_hr) else "N/A")
    
    with col4:
        st.metric("Critical HR", f"{results.critical_hr:.3f}" if not np.isnan(results.critical_hr) else "N/A")
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["📊 Plot", "📋 Predictions", "📐 Grid Data"])
    
    with tab1:
        fig = pkg['plot_prediction_curve'](results, figsize=(10, 6))
        st.pyplot(fig)
        plt.close(fig)
    
    with tab2:
        if results.predict_data and len(results.predict_data.get('time', [])) > 0:
            st.markdown("**Predictions:**")
            pred_df = pd.DataFrame(results.predict_data)
            st.dataframe(pred_df, use_container_width=True)
        
        if results.critical_data and len(results.critical_data.get('time', [])) > 0:
            st.markdown("**Critical Event Time:**")
            crit_df = pd.DataFrame(results.critical_data)
            st.dataframe(crit_df, use_container_width=True)
    
    with tab3:
        grid_df = pd.DataFrame(results.grid)
        st.dataframe(grid_df, use_container_width=True)
        
        # Download button
        csv = grid_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Grid Data (CSV)",
            data=csv,
            file_name="event_prediction_grid.csv",
            mime="text/csv"
        )


# =============================================================================
# Main App
# =============================================================================

def main():
    """Main application."""
    st.markdown('<p class="main-header">📊 Event Prediction</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Predict events in time-to-event clinical trials</p>', unsafe_allow_html=True)
    
    # Load package
    try:
        pkg = load_package()
    except ImportError as e:
        st.error(f"Failed to load eventprediction package: {e}")
        st.info("Make sure you're running from the project directory and dependencies are installed.")
        return
    
    # Sidebar
    mode = render_sidebar()
    
    # Main content based on mode
    trial_info = None
    
    if "NCT" in mode:
        trial_info = render_nct_section(pkg)
        st.markdown("---")
    
    # Parameters
    params = render_parameter_inputs(trial_info)
    
    st.markdown("---")
    
    # Predictions
    render_prediction_section(pkg, params)


if __name__ == "__main__":
    main()

