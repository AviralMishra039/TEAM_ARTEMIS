"""
Agentic Layer: Full LangGraph State Machine for Clinical Reasoning and Risk Assessment.
Optimized for Llama 3 8B local inference with support for Gemini cloud.
Features multi-node workflow with conditional routing.
"""

from langgraph.graph import StateGraph, END
from langchain_community.llms import Ollama
from typing import Dict, List, TypedDict, Literal
import logging
from datetime import datetime
import os
from functools import wraps
import time
import re

logger = logging.getLogger(__name__)

class ClinicalState(TypedDict):
    vitals: Dict
    history: List[Dict]
    risk_score: float
    reasoning: str
    action_taken: str
    escalation_required: bool
    timestamp: str
    triage_summary: str
    clinical_context: str

def rate_limit(min_interval_seconds=1):
    """Decorator to prevent rapid-fire calls."""
    def decorator(func):
        last_call = {"time": 0}
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_time = time.time()
            elapsed = current_time - last_call["time"]
            
            if elapsed < min_interval_seconds:
                wait_time = min_interval_seconds - elapsed
                logger.warning(f"⏳ Rate limit: waiting {wait_time:.1f}s")
                time.sleep(wait_time)
            
            result = func(*args, **kwargs)
            last_call["time"] = time.time()
            return result
        
        return wrapper
    return decorator

class ClinicalAgent:
    def __init__(self, model_type: str = "llama", model_name: str = "llama3"):
        """
        Initialize the clinical agent with full LangGraph workflow.
        
        Args:
            model_type: "llama" for local Ollama or "gemini" for cloud
            model_name: "llama3" or "gemini-1.5-flash"
        """
        try:
            self.model_type = model_type
            
            if model_type == "llama":
                logger.info(f"🦙 Initializing Llama model: {model_name}")
                self.llm = Ollama(
                    model=model_name,
                    temperature=0.1
                )
                logger.info("✅ Llama model loaded")
                
            elif model_type == "gemini":
                from langchain_google_genai import ChatGoogleGenerativeAI
                api_key = os.getenv("GOOGLE_API_KEY")
                if not api_key:
                    raise ValueError("GOOGLE_API_KEY not found")
                
                logger.info(f"☁️ Initializing Gemini model: {model_name}")
                self.llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    temperature=0.1,
                    google_api_key=api_key,
                    convert_system_message_to_human=True
                )
                logger.info("✅ Gemini model initialized")
            
            else:
                raise ValueError(f"Unknown model_type: {model_type}")
            
            self.workflow = self._build_workflow()
            self._call_count = 0
            
        except Exception as e:
            logger.error(f"❌ Agent initialization failed: {str(e)}")
            raise

    def _build_workflow(self) -> StateGraph:
        """
        Build the full multi-node LangGraph workflow:
        1. Triage Node - Initial clinical assessment
        2. Route Decision - Determine escalation need
        3. Log/Escalate Nodes - Final action
        """
        
        def triage_node(state: ClinicalState) -> ClinicalState:
            """
            Node 1: Initial Triage and Clinical Assessment
            Analyzes vitals and provides detailed reasoning
            """
            vitals = state.get('vitals', {})
            
            # Extract clinical parameters
            hr = vitals.get('hr', 0)
            spike_val = vitals.get('z_score', 0)
            trend_offset = vitals.get('residual', 0)
            iso_anomaly = vitals.get('iso_anomaly', False)
            activity = vitals.get('activity', 'Unknown')
            medication = vitals.get('medication', 'Standard post-op protocol')
            surgery = vitals.get('surgery_type', 'General')
            risk_level = vitals.get('risk_level', 'STABLE')
            
            prompt = f"""You are a Senior Post-Operative Clinical AI Analyst.

PATIENT PROFILE:
- Surgery Type: {surgery}
- Medications: {medication}
- Current Activity: {activity}

VITAL SIGNS:
- Heart Rate: {hr:.1f} BPM
- Activity State: {activity}

ML DETECTION RESULTS:
- Acute Spike (Z-Score): {spike_val:.2f}
  → {'ELEVATED - Potential concern' if abs(spike_val) > 2.5 else 'NORMAL - Within expected range'}
  
- Pattern Irregularity (Isolation Forest): {'DETECTED - Unusual pattern' if iso_anomaly else 'NORMAL - Expected pattern'}
  
- Recovery Trend Offset (LSTM): {trend_offset:.2f}
  → {'CONCERNING - Deviating from recovery path' if abs(trend_offset) > 8 else 'EXPECTED - On recovery trajectory'}

- Risk Classification: {risk_level}

TASK:
Provide a comprehensive clinical assessment (3-4 sentences) addressing:

1. Context Interpretation: Consider how medications (especially beta-blockers) and activity level affect heart rate readings
2. Risk Correlation: Evaluate if the Acute Spike, Pattern Irregularity, and Trend Offset are clinically significant together
3. Differential Diagnosis: Identify potential causes (e.g., normal post-op stress, early infection, medication effects, arrhythmia)
4. Clinical Significance: Determine if these findings warrant immediate intervention or continued monitoring

IMPORTANT: Provide ONLY the clinical assessment text. Do NOT include HTML tags, code formatting, or any markup. Write in clear, professional medical language as if documenting in a patient chart.

CLINICAL ASSESSMENT:"""
            
            try:
                logger.info("🔄 Triage node - invoking LLM for clinical assessment...")
                
                if self.model_type == "llama":
                    response = self.llm.invoke(prompt)
                    reasoning = response
                else:
                    response = self.llm.invoke(prompt)
                    reasoning = response.content
                
                # Clean any residual HTML/markup
                reasoning_clean = re.sub(r'<[^>]+>', '', reasoning).strip()
                
                # Remove common LLM artifacts
                reasoning_clean = reasoning_clean.replace('```', '').replace('**', '')
                
                state['reasoning'] = reasoning_clean
                state['triage_summary'] = f"Triage completed for {surgery} patient"
                state['clinical_context'] = f"HR: {hr:.1f}, Spike: {spike_val:.2f}, Offset: {trend_offset:.2f}"
                state['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                logger.info(f"✅ Triage complete ({len(reasoning_clean)} chars)")
                
            except Exception as e:
                logger.error(f"❌ Triage error: {str(e)}")
                state['reasoning'] = f"Analysis failed: {str(e)[:100]}"
            
            return state

        def route_decision(state: ClinicalState) -> Literal["log", "escalate"]:
            """
            Conditional Edge: Route based on risk level and ML detections
            - HIGH/CRITICAL risk → Escalate
            - STABLE risk → Log for monitoring
            """
            vitals = state.get('vitals', {})
            risk_level = vitals.get('risk_level', 'STABLE')
            
            # Additional checks for escalation
            spike_val = abs(vitals.get('z_score', 0))
            trend_offset = abs(vitals.get('residual', 0))
            
            # Escalate if:
            # 1. Risk level is HIGH or CRITICAL, OR
            # 2. Multiple severe anomalies detected
            should_escalate = (
                risk_level in ['HIGH', 'CRITICAL'] or
                (spike_val > 3.5 and trend_offset > 10)  # Dual severe anomalies
            )
            
            route = "escalate" if should_escalate else "log"
            logger.info(f"🔀 Routing: {route.upper()} (Risk: {risk_level}, Spike: {spike_val:.2f}, Offset: {trend_offset:.2f})")
            
            return route

        def log_node(state: ClinicalState) -> ClinicalState:
            """
            Node 2A: Log for Routine Monitoring
            Patient is stable, continue standard post-op care
            """
            logger.info("📋 Log node - routine monitoring")
            
            state['action_taken'] = "✅ Continue Monitoring"
            state['escalation_required'] = False
            state['risk_score'] = 0.3  # Low risk
            
            return state

        def escalate_node(state: ClinicalState) -> ClinicalState:
            """
            Node 2B: Escalate to Clinical Team
            Concerning findings require immediate attention
            """
            logger.info("🚨 Escalate node - urgent intervention")
            
            vitals = state.get('vitals', {})
            risk_level = vitals.get('risk_level', 'HIGH')
            
            if risk_level == 'CRITICAL':
                state['action_taken'] = "🚨 CRITICAL - Immediate Intervention Required"
                state['risk_score'] = 0.95
            else:
                state['action_taken'] = "⚠️ ESCALATE to Care Team"
                state['risk_score'] = 0.75
            
            state['escalation_required'] = True
            
            return state

        # Build the workflow graph
        workflow = StateGraph(ClinicalState)
        
        # Add nodes
        workflow.add_node("triage", triage_node)
        workflow.add_node("log", log_node)
        workflow.add_node("escalate", escalate_node)

        # Set entry point
        workflow.set_entry_point("triage")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "triage", 
            route_decision, 
            {
                "log": "log",
                "escalate": "escalate"
            }
        )
        
        # Add terminal edges
        workflow.add_edge("log", END)
        workflow.add_edge("escalate", END)

        logger.info("✅ LangGraph workflow compiled: triage → [log|escalate] → END")
        
        return workflow.compile()

    @rate_limit(min_interval_seconds=1)
    def invoke(self, state: Dict) -> ClinicalState:
        """
        Invoke the full clinical workflow.
        Rate limited to 1 call per second.
        """
        self._call_count += 1
        logger.info(f"📞 Agent invoke #{self._call_count} - Starting workflow")
        
        # Initialize clinical state
        clinical_state: ClinicalState = {
            'vitals': state.get('vitals', {}),
            'history': state.get('history', []),
            'risk_score': 0.0,
            'reasoning': '',
            'action_taken': '',
            'escalation_required': False,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'triage_summary': '',
            'clinical_context': ''
        }
        
        try:
            # Execute the workflow
            result = self.workflow.invoke(clinical_state)
            logger.info(f"✅ Workflow completed - Action: {result.get('action_taken')}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Workflow error: {str(e)}")
            return {
                'vitals': clinical_state['vitals'],
                'history': clinical_state['history'],
                'risk_score': 0.5,
                'reasoning': f"Workflow error: {str(e)[:150]}",
                'action_taken': "⚠️ System Error - Manual Review Required",
                'escalation_required': True,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'triage_summary': 'Error in processing',
                'clinical_context': 'N/A'
            }

def get_clinical_agent(model_type: str = "llama", model_name: str = "llama3") -> ClinicalAgent:
    """
    Factory function to create a clinical agent.
    
    Args:
        model_type: "llama" for local or "gemini" for cloud
        model_name: Model identifier
    """
    return ClinicalAgent(model_type=model_type, model_name=model_name)