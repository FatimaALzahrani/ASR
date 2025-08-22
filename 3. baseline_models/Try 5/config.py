class AudioConfig:
    TARGET_SAMPLE_RATE = 22050
    TARGET_RMS = 0.02
    NOISE_REDUCTION_FACTOR = 0.5
    FRAME_LENGTH = 2048
    HOP_LENGTH = 512
    SILENCE_MARGIN = 0.05
    MAX_AMPLITUDE = 0.95

class FeatureConfig:
    N_MFCC = 13
    N_MELS = 13
    FEATURE_SELECTION_K = 80

class TrainingConfig:
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    N_ESTIMATORS = 100
    MIN_SAMPLES_PER_CLASS = 3
    MAX_WORDS_DEFAULT = 50
    PROGRESS_REPORT_INTERVAL = 200

class SpeakerConfig:
    SPEAKERS = {
        'أحمد': range(0, 7),
        'عاصم': range(7, 14),
        'هيفاء': range(14, 21),
        'أسيل': range(21, 29),
        'وسام': range(29, 37)
    }

class PathConfig:
    DEFAULT_INPUT_DIR = "C:/Users/فاطمة الزهراني/Desktop/ابحاث/الداون/Data/clean"
    DEFAULT_OUTPUT_DIR = "output_files/enhanced_audio_data"
    QUALITY_ANALYSIS_FILE = "detailed_quality_analysis.csv"
    PROCESSING_REPORT_FILE = "processing_report.csv"
    RESULTS_FILE = "enhanced_models_final_comparison.json"