"""LLM EKG — Mathematical health monitor for LLMs."""
from .engine import LLMAnalyzer, LLMFeatureExtractor, FEATURE_NAMES, N_FEATURES
from .report import EKGReport
from .security import SecurityBaseline, SecurityReport
from .__main__ import LiveMonitor, parse_input

__version__ = "1.1.0"
__author__ = "Carmen Esteban"
