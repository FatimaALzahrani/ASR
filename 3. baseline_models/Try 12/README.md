# Realistic ASR System for Down Syndrome (Anti-Overfitting)

A comprehensive Automatic Speech Recognition system specifically designed for Down syndrome speech patterns with built-in overfitting prevention mechanisms.

## Features

- **Conservative Feature Extraction**: Extracts essential audio features while avoiding curse of dimensionality
- **Overfitting Detection**: Built-in mechanisms to detect and prevent model overfitting
- **Realistic Performance Evaluation**: Provides honest, production-ready accuracy estimates
- **Speaker-Aware Processing**: Handles multiple speakers with different severity levels
- **Session-Based Data Management**: Prevents data leakage through proper session grouping

## Installation

### Requirements

```bash
pip install numpy pandas scikit-learn librosa xgboost matplotlib seaborn
```

### Optional Dependencies

- `xgboost` - For XGBoost model support (will gracefully fallback if not available)

## Usage

### Basic Usage

```python
from asr_system import RealisticASRSystem

# Initialize system
asr_system = RealisticASRSystem("path/to/your/audio/data")

# Run complete pipeline
output_path = asr_system.run_realistic_pipeline()

# Make predictions
result = asr_system.predict_with_confidence("audio_file.wav", "speaker_name")
print(f"Predicted word: {result['predicted_word']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### Command Line Usage

```bash
python main.py
```

Follow the prompts to:

1. Enter your data folder path
2. Let the system process and train models
3. Optionally test predictions on new audio files

## Data Structure

Expected data folder structure:

```
data_folder/
├── word1/
│   ├── audio1.wav
│   ├── audio2.wav
│   └── ...
├── word2/
│   ├── audio1.wav
│   └── ...
└── ...
```

## Models

The system includes several conservative models designed to prevent overfitting:

- **Conservative Extra Trees**: Shallow trees with strict constraints
- **Conservative Random Forest**: Limited depth and features
- **Regularized Logistic Regression**: Strong L2 regularization
- **Simple SVM**: Conservative RBF kernel with regularization
- **Conservative XGBoost**: Limited boosting with regularization (optional)

## Output

The system generates:

1. **Trained Models**: Pickled model files for production use
2. **Performance Report**: JSON report with realistic accuracy metrics
3. **Feature Data**: CSV file with extracted features
4. **Health Assessment**: Model reliability evaluation

## Speaker Profiles

The system recognizes 5 speakers with different characteristics:

- **أحمد**: Age 10, IQ 38, Male, Moderate severity
- **عاصم**: Age 11, IQ 55, Male, Excellent performance
- **هيفاء**: Age 7, IQ 64, Female, Good performance
- **أسيل**: Age 16, IQ 40, Female, Weak performance
- **وسام**: Age 6, IQ 45, Male, Moderate severity

## Key Features

### Overfitting Prevention

- **Feature Selection**: Automatically selects most relevant features
- **Conservative Parameters**: Uses shallow models with regularization
- **Cross-Validation**: Rigorous evaluation with multiple folds
- **Reality Check**: Compares CV scores with test performance

### Reliability Assessment

Each model is evaluated for:

- **Overfitting Severity**: None, Mild, Moderate, Severe
- **Generalization Gap**: Difference between training and test performance
- **Production Readiness**: Whether safe for real-world deployment

### Performance Metrics

- **Realistic Accuracy**: Cross-validated performance estimates
- **Confidence Scores**: Prediction reliability indicators
- **Top-3 Predictions**: Multiple prediction candidates
- **Model Health Status**: Overfitting detection results

## Best Practices

1. **Data Quality**: Ensure minimum 5 samples per word
2. **Feature Ratio**: Keep features/samples ratio below 0.1
3. **Model Selection**: Use only "healthy" models for production
4. **Validation**: Always test on new, unseen data before deployment
