# Ultimate High Accuracy ASR System

A comprehensive Automatic Speech Recognition (ASR) system designed for Arabic dysarthric speech recognition with advanced feature extraction and multiple machine learning models.

## Features

### Audio Feature Extraction

- **MFCC Features**: 20 coefficients with delta and delta-delta
- **Mel Spectrogram**: 30 mel bands with statistical analysis
- **Spectral Features**: Centroid, rolloff, bandwidth, flatness, contrast
- **Fundamental Frequency (F0)**: Pitch analysis using YIN algorithm
- **Energy Features**: RMS energy and zero-crossing rate
- **Chroma Features**: 12 chromagram coefficients
- **Temporal Features**: Duration, amplitude, energy statistics
- **LPC Coefficients**: Linear predictive coding analysis

### Machine Learning Models

- Random Forest (Optimized)
- XGBoost (Hypertuned)
- Extra Trees
- Multi-layer Perceptron (MLP)
- Support Vector Machine (SVM)
- LightGBM
- Gradient Boosting
- Voting Ensemble
- Stacking Ensemble

### Data Processing

- Automatic speaker identification from filenames
- Smart data balancing with SMOTE and manual augmentation
- Multiple scaling methods (Standard, Robust)
- Comprehensive error handling and validation

## Installation

1. **Install Python dependencies:**

```bash
pip install numpy pandas librosa scikit-learn xgboost lightgbm scipy
```

2. **Optional (for advanced balancing):**

```bash
pip install imbalanced-learn
```

## Usage

### Basic Usage

1. **Update data path** in `asr_system.py` if needed:

```python
ASRSystem(data_path="your/data/path")
```

2. **Run the system:**

```bash
python main.py
```

### Data Structure Expected

Your data should be organized as:

```
data/
├── word1/
│   ├── 0.wav
│   ├── 1.wav
│   └── ...
├── word2/
│   ├── 7.wav
│   ├── 8.wav
│   └── ...
└── ...
```

### Speaker Mapping

The system automatically identifies speakers based on filename numbers:

- Files 0-6: Ahmed
- Files 7-13: Asem
- Files 14-20: Haifa
- Files 21-28: Aseel
- Files 29-36: Wessam

## Output

The system generates:

1. **Results JSON** (`ultimate_high_accuracy_results.json`):

   - Dataset information
   - Model performance metrics
   - Best model identification
   - Evaluation timestamp

2. **Trained Models** (`ultimate_models.pkl`):
   - All trained models
   - Feature scaler
   - Label encoder
   - Feature column names

## Performance

The system provides:

- Comprehensive model comparison
- Accuracy metrics for all models
- Best model identification
- Feature importance analysis
- Detailed performance logging

## Customization

### Adding New Models

Add new models in `model_trainer.py`:

```python
def train_new_model(self, X_train, X_test, y_train, y_test):
    model = YourModel(parameters)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return model, accuracy
```

### Feature Extraction Customization

Modify `feature_extractor.py` to add new features:

```python
def extract_new_feature(self, y, sr):
    # Your feature extraction code
    new_feature = your_feature_extraction(y, sr)
    return new_feature
```

### Data Path Configuration

Update the default data path in `asr_system.py`:

```python
def __init__(self, data_path="your/custom/path"):
```
