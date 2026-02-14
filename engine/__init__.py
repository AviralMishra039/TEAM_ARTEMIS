"""
Vital-Sync Engine Package
Post-Operative Remote Monitoring System - ML and Agentic Components
"""

from .data_loader import generate_live_data
from .ml_models import VitalDetector, AnomalyDetection
from .agent_logic import get_clinical_agent, ClinicalAgent

__all__ = [
    'generate_live_data',
    'VitalDetector',
    'AnomalyDetection',
    'get_clinical_agent',
    'ClinicalAgent'
]

__version__ = '1.0.0'
