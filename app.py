"""
Vital-Sync: Post-Operative Remote Monitoring System
Professional Streamlit Dashboard with Hybrid ML-Agentic Architecture
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
import traceback
import os
import time
import re

# Import the modular engine components
from engine.data_loader import generate_live_data
from engine.ml_models import VitalDetector
from engine.agent_logic import ClinicalAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Page Configuration ---
st.set_page_config(
    page_title="Vital-Sync | Clinical Monitoring",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Professional Custom CSS Styling ---
st.markdown("""
    <style>
    .main { background: linear-gradient(135deg, #0e1117 0%, #161b22 100%); }
    
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #00d4ff;
    }
    
    .metric-container {
        background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #30363d;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin-bottom: 10px;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1c2128 0%, #161b22 100%);
        padding: 20px;
        border-radius: 12px;
        border-left: 4px solid #00d4ff;
        margin-bottom: 20px;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #238636 0%, #2ea043 100%);
        color: white;
        border: none;
        width: 100%;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(46, 160, 67, 0.3);
    }

    .clinical-box {
        background: #161b22;
        padding: 20px;
        border-radius: 12px;
        border: 2px solid #30363d;
        margin-top: 15px;
    }
    
    .risk-low { color: #238636; }
    .risk-medium { color: #d29922; }
    .risk-high { color: #f85149; }
    .risk-critical { color: #da3633; font-weight: 700; }
    
    .model-badge {
        background: #1f6feb;
        color: white;
        padding: 5px 10px;
        border-radius: 5px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    .workflow-box {
        background: #1c2128;
        padding: 12px;
        border-radius: 6px;
        margin-bottom: 15px;
        border-left: 3px solid #1f6feb;
    }
    
    .assessment-box {
        background: #0d1117;
        padding: 15px;
        border-radius: 6px;
        line-height: 1.8;
        margin-top: 15px;
        border: 1px solid #30363d;
    }
    
    .escalation-box {
        background: #3d1a1a;
        padding: 12px;
        border-radius: 6px;
        margin-top: 15px;
        border-left: 3px solid #da3633;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Initialize Session State ---
if 'agent_ready' not in st.session_state:
    st.session_state['agent_ready'] = False

if 'last_triage_time' not in st.session_state:
    st.session_state['last_triage_time'] = 0

if 'agent_instance' not in st.session_state:
    st.session_state['agent_instance'] = None

if 'inference_count' not in st.session_state:
    st.session_state['inference_count'] = 0

if 'model_type' not in st.session_state:
    st.session_state['model_type'] = 'llama'

if 'api_call_count' not in st.session_state:
    st.session_state['api_call_count'] = 0

# --- Sidebar Configuration ---
with st.sidebar:
    st.title("⚙️ System Setup")
    
    # Model Selection
    st.subheader("🤖 AI Model Selection")
    model_choice = st.selectbox(
        "Inference Backend",
        ["Llama 3 8B (Local)", "Gemini 1.5 Flash (Cloud)"],
        help="Choose between local or cloud-based inference"
    )
    
    # Parse selection
    use_llama = "Llama" in model_choice
    current_model_type = 'llama' if use_llama else 'gemini'
    
    # Check if model type changed
    if st.session_state.get('model_type') != current_model_type:
        st.session_state['model_type'] = current_model_type
        st.session_state['agent_instance'] = None
        st.session_state['agent_ready'] = False
        logger.info(f"Model type changed to {current_model_type}")
    
    st.divider()
    
    # Model-specific setup
    if use_llama:
        st.info("**Llama 3 8B** - Running locally via Ollama")
        api_key_active = True
        
    else:
        # Gemini setup
        st.info("**Gemini 1.5 Flash** - Cloud API")
        user_api_key = st.text_input(
            "Google API Key", 
            type="password",
            help="Enter your Gemini API key"
        )
        
        col1, col2 = st.columns([2, 1])
        with col1:
            if st.button("🔐 Activate", use_container_width=True):
                if user_api_key:
                    st.session_state['api_key_confirmed'] = user_api_key
                    st.session_state['agent_instance'] = None
                    st.session_state['agent_ready'] = False
                    st.success("✅ Key Activated")
                    st.rerun()
                else:
                    st.error("Enter API key first")
        
        with col2:
            if st.button("🔄", use_container_width=True):
                for key in ['api_key_confirmed', 'agent_instance', 'agent_ready']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.session_state['api_call_count'] = 0
                st.rerun()
        
        api_key_active = 'api_key_confirmed' in st.session_state
    
    # Agent status
    if api_key_active and st.session_state.get('agent_ready'):
        st.success(f"🟢 Agent Active")
    elif api_key_active:
        st.info("🟡 Initializing...")
    else:
        st.warning("🔒 Not Activated")
    
    # Usage counter
    if use_llama:
        st.metric("Inferences", st.session_state['inference_count'])
    else:
        st.metric("API Calls", st.session_state['api_call_count'])
    
    st.divider()
    
    # Patient Context
    st.subheader("📋 Patient Context")
    surgery_type = st.selectbox("Surgery Type", ["Cardiac", "Orthopedic", "General"])
    data_points = st.slider("Monitoring Window (Minutes)", 100, 500, 350)
    
    st.divider()
    
    # ML Sensitivity
    st.subheader("🛡️ Detection Sensitivity")
    z_threshold = st.slider("Acute Spike", 2.0, 4.0, 2.9)
    lstm_threshold = st.slider("Trend Deviation", 5.0, 15.0, 8.0)
    
    st.divider()
    
    # Reset
    if st.button("🔄 Reset System", use_container_width=True):
        st.session_state['agent_instance'] = None
        st.session_state['agent_ready'] = False
        st.session_state['inference_count'] = 0
        st.session_state['api_call_count'] = 0
        st.session_state['last_triage_time'] = 0
        st.rerun()
    
    st.caption("Vital-Sync v2.0")

# --- Cached Components ---
@st.cache_resource
def init_ml_detector(_z_threshold, _lstm_threshold):
    logger.info(f"Initializing ML Detector")
    return VitalDetector(z_threshold=_z_threshold, lstm_residual_threshold=_lstm_threshold)

# --- Initialize ---
detector = init_ml_detector(z_threshold, lstm_threshold)

# Agent initialization
agent = None
if api_key_active:
    if st.session_state['agent_instance'] is None:
        try:
            if use_llama:
                logger.info("Creating Llama agent...")
                with st.sidebar:
                    with st.spinner("Loading model..."):
                        agent = ClinicalAgent(model_type="llama", model_name="llama3")
                st.sidebar.success("✅ Model loaded")
            else:
                logger.info("Creating Gemini agent...")
                os.environ["GOOGLE_API_KEY"] = st.session_state['api_key_confirmed']
                with st.sidebar:
                    with st.spinner("Connecting to API..."):
                        agent = ClinicalAgent(model_type="gemini", model_name="gemini-1.5-flash")
                st.session_state['api_call_count'] += 1
                st.sidebar.success("✅ API connected")
            
            st.session_state['agent_instance'] = agent
            st.session_state['agent_ready'] = True
            
        except Exception as e:
            st.sidebar.error(f"Initialization failed: {str(e)[:100]}")
            logger.error(f"Error: {traceback.format_exc()}")
    else:
        agent = st.session_state['agent_instance']

# --- Header ---
model_display = "Llama 3 8B" if use_llama else "Gemini 1.5 Flash"
st.markdown(f"""
    <div class="main-header">
        <h1 style="margin: 0; color: #00d4ff;">🏥 Vital-Sync Dashboard</h1>
        <p style="margin: 5px 0 0 0; color: #8b949e;">
            Post-Operative Remote Clinical Decision Support System
            <span style="float: right; background: #1f6feb; padding: 3px 8px; border-radius: 4px; font-size: 0.85rem;">
                {model_display}
            </span>
        </p>
    </div>
    """, unsafe_allow_html=True)

# --- Data Engine ---
try:
    data = generate_live_data(n_points=data_points, surgery_type=surgery_type)
    processed_data = detector.run_inference(data)
    latest = processed_data.iloc[-1]
    summary = detector.get_anomaly_summary(processed_data)
except Exception as e:
    st.error(f"Data Error: {e}")
    logger.error(f"Data generation failed: {traceback.format_exc()}")
    st.stop()

# --- Metrics Dashboard ---
st.markdown("### 📊 Real-Time Clinical Vitals")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Heart Rate", f"{latest['hr']:.1f} BPM", delta=f"{latest['z_score']:.2f} σ")
with col2:
    status_map = {"STABLE": "✅ Stable", "HIGH": "🟠 High Risk", "CRITICAL": "🔴 Critical"}
    st.metric("Status", status_map.get(latest['risk_level'], "⚠️ Anomaly"))
with col3:
    risk_class = f"risk-{latest['risk_level'].lower()}"
    st.markdown(f'''<div class="metric-container"><div style="color: #8b949e; font-size: 0.9rem;">Trend Offset</div>
    <div class="{risk_class}" style="font-size: 1.8rem; font-weight: 700;">{latest["residual"]:.2f}</div></div>''', unsafe_allow_html=True)
with col4:
    st.metric("Activity", latest['activity'])

# --- Summary Cards ---
st.markdown("### 🔍 Detection Summary")
sc1, sc2, sc3, sc4 = st.columns(4)
sc1.metric("Acute Spikes", summary.get('z_score_anomalies', 0))
sc2.metric("Pattern Irregularities", summary.get('iso_anomalies', 0))
sc3.metric("Trend Deviations", summary.get('lstm_anomalies', 0))
sc4.metric("Total Events", summary.get('total_anomalies', 0))

# --- Graph ---
st.markdown("### 📈 Multi-Model Analysis")
fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], vertical_spacing=0.1,
                    subplot_titles=('Vital Sign Tracking', 'Recovery Trend Residuals'))

fig.add_trace(go.Scatter(x=processed_data['timestamp'], y=processed_data['hr'], 
                        name="Actual HR", line=dict(color="#00d4ff", width=2)), row=1, col=1)
fig.add_trace(go.Scatter(x=processed_data['timestamp'], y=processed_data['lstm_pred'], 
                        name="Predicted Path", line=dict(color="rgba(255,255,255,0.3)", dash='dot')), row=1, col=1)

z_anom = processed_data[processed_data['z_score_anomaly']]
if not z_anom.empty:
    fig.add_trace(go.Scatter(x=z_anom['timestamp'], y=z_anom['hr'], mode='markers', 
                            name='Acute Spike', marker=dict(color='#ffaa00', size=10)), row=1, col=1)

iso_anom = processed_data[processed_data['iso_anomaly']]
if not iso_anom.empty:
    fig.add_trace(go.Scatter(x=iso_anom['timestamp'], y=iso_anom['hr'], mode='markers', 
                            name='Pattern Irregularity', marker=dict(color='#ff4b4b', size=12, symbol='x')), row=1, col=1)

fig.add_trace(go.Scatter(x=processed_data['timestamp'], y=processed_data['residual'], 
                        name="Trend Offset", fill='tozeroy', line=dict(color="#8b949e")), row=2, col=1)
fig.add_hline(y=lstm_threshold, line_dash="dash", line_color="#f85149", 
             annotation_text="Safety Threshold", row=2, col=1)

fig.update_layout(template="plotly_dark", height=600, margin=dict(l=20, r=20, t=40, b=20), hovermode='x unified')
st.plotly_chart(fig, use_container_width=True)

# --- Agentic Reasoning ---
st.divider()
st.markdown("### 🤖 Clinical Reasoning Engine")
al, ar = st.columns([1, 2])

with al:
    st.info("""
    **LangGraph Multi-Node Workflow**
    
    1️⃣ Triage: Clinical assessment
    
    2️⃣ Route: Risk-based decision
    
    3️⃣ Action: Monitor or escalate
    """)
    
    # Rate limiting
    time_since_last = time.time() - st.session_state['last_triage_time']
    min_interval = 2 if use_llama else 5
    can_invoke = time_since_last >= min_interval
    
    if not can_invoke:
        wait = int(min_interval - time_since_last)
        st.warning(f"⏳ Wait {wait}s")
    
    invoke_btn = st.button(
        "🚀 Run Clinical Triage", 
        type="primary",
        disabled=not can_invoke,
        use_container_width=True
    )
    
    count = st.session_state['inference_count'] if use_llama else st.session_state['api_call_count']
    st.caption(f"Analyses: {count}")

with ar:
    if invoke_btn:
        if not st.session_state.get('agent_ready'):
            st.error("⏳ Model loading...")
        elif agent is None:
            st.error("❌ Agent not initialized")
        else:
            st.session_state['last_triage_time'] = time.time()
            st.session_state['inference_count'] += 1
            if not use_llama:
                st.session_state['api_call_count'] += 1
            
            try:
                with st.status("Executing workflow...", expanded=True) as status:
                    vitals_data = latest.to_dict()
                    vitals_data['surgery_type'] = surgery_type
                    vitals_data['medication'] = "Post-op standard protocol"
                    
                    start_time = time.time()
                    result = agent.invoke({"vitals": vitals_data, "history": []})
                    inference_time = time.time() - start_time
                    
                    status.update(label=f"✅ Complete ({inference_time:.2f}s)", state="complete", expanded=False)
                
                # Extract data
                action_color = '#da3633' if result.get('escalation_required') else '#238636'
                risk_score = result.get('risk_score', 0)
                risk_color = '#da3633' if risk_score > 0.7 else ('#d29922' if risk_score > 0.4 else '#238636')
                
                # Get and clean reasoning
                reasoning = result.get('reasoning', 'No assessment available')
                reasoning_clean = re.sub(r'<[^>]+>', '', reasoning).strip()
                
                # Display workflow box
                st.markdown(f"""
                <div class="workflow-box">
                    <strong>Workflow:</strong> {result.get('triage_summary', 'Triage completed')}<br>
                    <strong>Risk Score:</strong> <span style="color: {risk_color}; font-weight: 700;">{risk_score:.2f}</span>
                </div>
                """, unsafe_allow_html=True)
                
                # Display action
                st.markdown(f"""
                <div style="padding: 10px; margin-bottom: 10px;">
                    <strong>Recommended Action:</strong> 
                    <span style="color: {action_color}; font-weight: bold; font-size: 1.15rem;">
                        {result.get('action_taken', 'Monitoring')}
                    </span>
                </div>
                """, unsafe_allow_html=True)
                
                # Display clinical assessment
                st.markdown("**📋 Clinical Assessment:**")
                st.markdown(f"""
                <div class="assessment-box">
                    {reasoning_clean}
                </div>
                """, unsafe_allow_html=True)
                
                # Metadata
                st.caption(f"⏱️ {result.get('timestamp', 'Unknown')} | {model_display} | {inference_time:.2f}s")
                
                # Escalation alert
                if result.get('escalation_required'):
                    st.markdown("""
                    <div class="escalation-box">
                        <strong>🚨 ESCALATION REQUIRED</strong> - Clinical team notified
                    </div>
                    """, unsafe_allow_html=True)
                
                logger.info(f"Triage completed: {result.get('action_taken')}")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                logger.error(f"Invocation error: {traceback.format_exc()}")
    else:
        st.info("💡 Click 'Run Clinical Triage' to execute the LangGraph workflow and receive clinical recommendations.")

st.divider()
st.caption("Vital-Sync | LangGraph Multi-Node Workflow | Built with Scikit-Learn & LangChain")