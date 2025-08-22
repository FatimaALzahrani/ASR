FEATURES_PATH = "features"
RESULTS_PATH = "advanced_results"
PROCESSED_DATA_PATH = "processed_data"

AUDIO_CONFIG = {
    'sample_rate': 16000,
    'n_mfcc': 13,
    'n_mels': 10,
    'n_fft': 2048,
    'hop_length': 512,
    'win_length': 2048
}

FEATURE_EXTRACTION_CONFIG = {
    'extract_mfcc': True,
    'extract_delta_mfcc': True,
    'extract_spectral': True,
    'extract_mel': True,
    'extract_f0': True,
    'extract_energy': True,
    'extract_temporal': True
}

MODEL_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'cv_folds': 5,
    'n_jobs': -1
}

ACOUSTIC_MODEL_CONFIG = {
    'ensemble_models': ['RandomForest', 'XGBoost', 'LightGBM', 'MLP'],
    'n_estimators': 100,
    'max_depth': 15,
    'learning_rate': 0.1,
    'confidence_threshold': 0.5
}

LANGUAGE_MODEL_CONFIG = {
    'context_window': 3,
    'smoothing_factor': 0.01,
    'min_word_frequency': 2
}

CORRECTION_CONFIG = {
    'similarity_threshold': 0.6,
    'max_edit_distance': 2,
    'phonetic_weight': 0.4,
    'text_weight': 0.6
}

SPEAKER_INFO = {
    'أحمد': {'age': 8, 'gender': 'ذكر', 'severity': 'متوسط'},
    'عاصم': {'age': 10, 'gender': 'ذكر', 'severity': 'خفيف'},
    'هيفاء': {'age': 9, 'gender': 'أنثى', 'severity': 'متوسط'},
    'أسيل': {'age': 7, 'gender': 'أنثى', 'severity': 'شديد'},
    'وسام': {'age': 11, 'gender': 'ذكر', 'severity': 'متوسط'},
    'غير معروف': {'age': 0, 'gender': 'غير محدد', 'severity': 'غير محدد'}
}

LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'save_logs': True,
    'log_file': 'speech_recognition.log'
}