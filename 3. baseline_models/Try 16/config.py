SPEAKER_MAPPING = {
    range(0, 7): "Ahmed",
    range(7, 14): "Asem", 
    range(14, 21): "Haifa",
    range(21, 29): "Aseel",
    range(29, 37): "Wessam"
}

EXCLUDED_WORDS = {"نوم"}

FEATURE_CONFIGS = {
    'mfcc_configs': [
        {'n_mfcc': 13, 'n_fft': 2048, 'hop_length': 512},
        {'n_mfcc': 20, 'n_fft': 1024, 'hop_length': 256},
        {'n_mfcc': 26, 'n_fft': 4096, 'hop_length': 1024},
        {'n_mfcc': 39, 'n_fft': 2048, 'hop_length': 256}
    ],
    'mel_configs': [
        {'n_mels': 40, 'n_fft': 2048, 'hop_length': 512},
        {'n_mels': 80, 'n_fft': 4096, 'hop_length': 1024},
        {'n_mels': 128, 'n_fft': 2048, 'hop_length': 256}
    ],
    'f0_methods': [
        {'fmin': 50, 'fmax': 500},
        {'fmin': 80, 'fmax': 400},
        {'fmin': 100, 'fmax': 300}
    ],
    'onset_methods': [
        {'units': 'frames'},
        {'units': 'time'},
        {'units': 'frames', 'pre_max': 20, 'post_max': 20, 'pre_avg': 100, 'post_avg': 100}
    ],
    'chroma_configs': [
        {'norm': 2},
        {'norm': float('inf')},
        {'norm': 1}
    ]
}

BALANCING_CONFIG = {
    'target_min': 20,
    'target_max': 30
}

MODEL_PARAMS = {
    'random_forest': {
        'n_estimators': 1500,
        'max_depth': 30,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'max_features': 'sqrt',
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1,
        'bootstrap': True,
        'oob_score': True
    },
    'extra_trees': {
        'n_estimators': 1500,
        'max_depth': 30,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'max_features': 'sqrt',
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1,
        'bootstrap': True,
        'oob_score': True
    },
    'xgboost': {
        'n_estimators': 1200,
        'max_depth': 20,
        'learning_rate': 0.03,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'min_child_weight': 2,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': 42,
        'eval_metric': 'mlogloss',
        'verbosity': 0,
        'n_jobs': -1
    },
    'svm': {
        'kernel': 'rbf',
        'C': 200,
        'gamma': 'scale',
        'class_weight': 'balanced',
        'probability': True,
        'random_state': 42
    },
    'lightgbm': {
        'n_estimators': 1200,
        'max_depth': 20,
        'learning_rate': 0.03,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'class_weight': 'balanced',
        'random_state': 42,
        'verbosity': -1,
        'n_jobs': -1
    },
    'mlp': {
        'hidden_layer_sizes': (1024, 512, 256, 128),
        'activation': 'relu',
        'solver': 'adam',
        'alpha': 0.0001,
        'learning_rate': 'adaptive',
        'max_iter': 500,
        'random_state': 42,
        'early_stopping': True,
        'validation_fraction': 0.1
    }
}

SPEAKER_MODEL_PARAMS = {
    'extra_trees': {
        'n_estimators': 1000,
        'max_depth': 25,
        'min_samples_split': 2, 
        'min_samples_leaf': 1,
        'max_features': 'sqrt',
        'class_weight': 'balanced', 
        'random_state': 42,
        'n_jobs': -1,
        'bootstrap': True
    },
    'random_forest': {
        'n_estimators': 1000,
        'max_depth': 25,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'max_features': 'sqrt',
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1,
        'bootstrap': True
    },
    'xgboost': {
        'n_estimators': 800,
        'max_depth': 15,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'random_state': 42,
        'eval_metric': 'mlogloss',
        'verbosity': 0
    },
    'svm': {
        'kernel': 'rbf',
        'C': 100,
        'gamma': 'scale',
        'class_weight': 'balanced', 
        'probability': True,
        'random_state': 42
    }
}
