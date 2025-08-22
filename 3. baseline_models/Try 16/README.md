# ASR System for Down Syndrome Speech Recognition

This project implements an Automatic Speech Recognition (ASR) system specifically designed for Down syndrome speech patterns using advanced machine learning techniques.

## Features

### Audio Feature Extraction

- **MFCC Features**: Multiple configurations with statistical analysis
- **Spectral Features**: Mel-spectrogram, spectral centroid, rolloff, bandwidth
- **Prosodic Features**: F0 analysis, energy analysis, pitch characteristics
- **Temporal Features**: Onset detection, tempo, rhythm, silence analysis
- **Harmonic Features**: Chroma analysis, harmonic-percussive separation

### Machine Learning Models

- Random Forest Classifier
- Extra Trees Classifier
- XGBoost Classifier
- Support Vector Machine (SVM)
- LightGBM Classifier
- Multi-layer Perceptron (MLP)
- Voting Ensemble
- Stacking Ensemble

### Data Processing

- Smart data balancing with augmentation techniques
- Speaker-specific model training
- Robust feature scaling
- Missing value handling

## Installation

1. Clone the repository
2. Install required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from asr_system import ASRSystem

# Initialize the system
asr = ASRSystem(data_path="path/to/your/data")

# Run evaluation
results = asr.run_evaluation()
```

### Configuration

Edit `config.py` to modify:

- Speaker mappings
- Feature extraction parameters
- Model hyperparameters
- Data balancing settings

## Data Format

The system expects audio data organized as follows:

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

## Speaker Mapping

- Files 0-6: Ahmed
- Files 7-13: Asem
- Files 14-20: Haifa
- Files 21-28: Aseel
- Files 29-36: Wessam

## Output

The system generates:

- `asr_results.json`: Detailed evaluation results
- `asr_models.pkl`: Trained models and preprocessing objects
- Console output with performance metrics

## Performance Metrics

The system evaluates models using:

- Accuracy scores for general models
- Cross-validation for speaker-specific models
- Ensemble method comparisons
