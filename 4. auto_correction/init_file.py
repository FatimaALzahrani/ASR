from .enhanced_auto_correction import EnhancedAutoCorrection
from .model_utils import load_model_and_data, create_simple_model
from .data_simulator import simulate_predictions_with_errors
from .report_generator import create_comprehensive_report

__version__ = "1.0.0"
__author__ = "Auto Correction System"
__description__ = "Enhanced Arabic Speech Recognition Auto Correction System"

__all__ = [
    'EnhancedAutoCorrection',
    'load_model_and_data',
    'create_simple_model',
    'simulate_predictions_with_errors',
    'create_comprehensive_report'
]