class Config:
    
    SAMPLE_RATE = 22050
    
    AUDIO_DURATION = 5.0
    
    MIN_SAMPLES_PER_WORD = 3
    
    MFCC_COEFFICIENTS = 13
    
    MEL_BANDS = 20
    
    F0_MIN = 50
    F0_MAX = 300
    
    AUDIO_EXTENSIONS = [".wav", ".mp3", ".flac", ".m4a"]
    
    SPEAKER_MAPPING = {
        range(0, 7): "أحمد",
        range(7, 14): "عاصم", 
        range(14, 21): "هيفاء",
        range(21, 29): "أسيل",
        range(29, 37): "وسام"
    }
    
    SPEAKER_PROFILES = {
        "أحمد": {"age": 10, "iq": 38, "gender": "male", "severity": "moderate"},
        "عاصم": {"age": 11, "iq": 55, "gender": "male", "severity": "excellent"},
        "هيفاء": {"age": 7, "iq": 64, "gender": "female", "severity": "good"},
        "أسيل": {"age": 16, "iq": 40, "gender": "female", "severity": "weak"},
        "وسام": {"age": 6, "iq": 45, "gender": "male", "severity": "moderate"}
    }
    
    DIFFICULT_ARABIC_CHARS = ['خ', 'غ', 'ق', 'ض', 'ظ', 'ث', 'ذ', 'ص', 'ز']
    
    FRICATIVE_ARABIC_CHARS = ['ف', 'ث', 'ذ', 'س', 'ش', 'ص', 'ض', 'خ', 'غ', 'ح', 'ه']
    
    MODEL_CONFIGS = {
        'ExtraTrees': {
            'n_estimators': 500,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'class_weight': 'balanced_subsample'
        },
        'RandomForest': {
            'n_estimators': 300,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'class_weight': 'balanced_subsample'
        },
        'GradientBoosting': {
            'n_estimators': 200,
            'max_depth': 8,
            'learning_rate': 0.1
        },
        'MLP': {
            'hidden_layer_sizes': (256, 128, 64),
            'activation': 'relu',
            'solver': 'adam',
            'max_iter': 1000,
            'early_stopping': True,
            'validation_fraction': 0.1
        },
        'SVM': {
            'kernel': 'rbf',
            'C': 10,
            'gamma': 'scale',
            'class_weight': 'balanced',
            'probability': True
        },
        'XGBoost': {
            'n_estimators': 300,
            'max_depth': 8,
            'learning_rate': 0.1,
            'eval_metric': 'mlogloss',
            'verbosity': 0
        },
        'LightGBM': {
            'n_estimators': 300,
            'max_depth': 8,
            'learning_rate': 0.1,
            'verbose': -1,
            'class_weight': 'balanced'
        }
    }
    
    CROSS_VALIDATION_FOLDS = 5
    
    TEST_SIZE = 0.2
    
    RANDOM_STATE = 42
    
    OUTPUT_DIRECTORY = "asr_system_output"
    
    DEFAULT_DATA_PATHS = [
        "C:/Users/فاطمة الزهراني/Desktop/ابحاث/الداون/Data/clean",
        "./Data/clean",
        "./data",
        "."
    ]