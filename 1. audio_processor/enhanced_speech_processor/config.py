# Audio processing settings
TARGET_SAMPLE_RATE = 16000
TARGET_DURATION = 3.0  # seconds
MIN_DURATION = 0.5
MAX_DURATION = 10.0

# Data splitting ratios
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1

# File paths
DATA_ROOT = "data"
CLEAN_AUDIO_DIR = "clean"
PROCESSED_DIR = "processed"
REPORTS_DIR = "reports"

# Speaker information based on filename ranges
SPEAKER_RANGES = {
    range(0, 7): 'أحمد',
    range(7, 14): 'عاصم', 
    range(14, 21): 'هيفاء',
    range(21, 29): 'أسيل',
    range(29, 37): 'وسام'
}

SPEAKER_INFO = {
    'أحمد': {'age': 10, 'gender': 'male', 'iq_level': 38, 'speech_quality': 'poor'},
    'عاصم': {'age': 11, 'gender': 'male', 'iq_level': 55, 'speech_quality': 'excellent'},
    'هيفاء': {'age': 7, 'gender': 'female', 'iq_level': 64, 'speech_quality': 'good'},
    'أسيل': {'age': 16, 'gender': 'female', 'iq_level': 40, 'speech_quality': 'poor'},
    'وسام': {'age': 6, 'gender': 'male', 'iq_level': 'medium', 'speech_quality': 'medium'}
}

# Words to exclude from processing
EXCLUDED_WORDS = ['نوم']

# Audio quality thresholds
QUALITY_THRESHOLDS = {
    'min_rms_energy': 0.005,
    'max_rms_energy': 0.3,
    'min_snr': 5,
    'max_clipping_ratio': 0.05,
    'max_silence_ratio': 0.7
}