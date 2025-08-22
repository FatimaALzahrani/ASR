class Config:
    SAMPLE_RATE = 22050
    DURATION = 3.0
    RANDOM_STATE = 42
    MIN_SAMPLES_PER_WORD = 3
    MIN_QUALITY_SCORE = 0.3
    
    SPEAKER_MAPPING = {
        'أحمد': list(range(0, 7)),
        'عاصم': list(range(7, 14)),
        'هيفاء': list(range(14, 21)),
        'أسيل': list(range(21, 29)),
        'وسام': list(range(29, 37))
    }
    
    SUPPORTED_AUDIO_FORMATS = ('.wav', '.mp3', '.m4a', '.flac')
    
    FEATURE_EXTRACTION = {
        'n_mfcc': 13,
        'n_chroma': 12,
        'n_mel': 13,
        'n_fft': 2048,
        'hop_length': 512
    }
    
    QUALITY_WEIGHTS = {
        'snr': 0.4,
        'dynamic_range': 0.2,
        'clipping': 0.2,
        'silence': 0.1,
        'duration': 0.1
    }
    
    MODEL_PARAMS = {
        'random_forest': {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2
        },
        'svm': {
            'C': 1.0,
            'gamma': 'scale'
        },
        'logistic_regression': {
            'C': 1.0,
            'max_iter': 1000
        },
        'gradient_boosting': {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1
        },
        'decision_tree': {
            'max_depth': 8,
            'min_samples_split': 5,
            'min_samples_leaf': 2
        },
        'knn': {
            'n_neighbors': 5,
            'weights': 'distance'
        }
    }
    
    OUTPUT_FILES = {
        'complete_dataset': 'complete_dataset_standalone.csv',
        'train_dataset': 'train_dataset_standalone.csv',
        'test_dataset': 'test_dataset_standalone.csv',
        'statistics': 'dataset_statistics_standalone.json',
        'report': 'comprehensive_report_standalone.json',
        'summary': 'model_performance_summary_standalone.csv'
    }
