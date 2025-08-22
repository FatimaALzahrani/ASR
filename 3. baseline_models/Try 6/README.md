# Standalone Arabic Speech Recognition System

A comprehensive Arabic speech recognition system with advanced audio processing, feature extraction, and machine learning capabilities.

## Features

### Audio Processing

- **Spectral Gating Denoising**: Advanced noise reduction using spectral subtraction
- **Audio Enhancement**: Pre-emphasis filtering, normalization, and trimming
- **Quality Assessment**: Comprehensive audio quality scoring based on SNR, dynamic range, and other factors

### Feature Extraction

- **Temporal Features**: RMS energy, amplitude statistics, zero-crossing rate
- **Spectral Features**: Spectral centroid, rolloff, bandwidth
- **MFCC Features**: 13 Mel-frequency cepstral coefficients with statistics
- **Chroma Features**: 12 chromagram features with statistics
- **Mel-spectrogram Features**: 13 mel-scale spectral features

### Machine Learning Models

- Random Forest Classifier
- Support Vector Machine (RBF kernel)
- Logistic Regression
- Gradient Boosting Classifier
- Decision Tree Classifier
- K-Nearest Neighbors

### Data Management

- **Speaker Mapping**: Automatic speaker identification from filenames
- **Quality Filtering**: Removes low-quality audio samples
- **Sample Filtering**: Ensures minimum samples per word class
- **Speaker-Independent Split**: Train/test split by speakers for generalization

## Installation

1. Clone or download the project files
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

```bash
python main.py --data_path "path/to/your/data" --output_dir "output"
```

### Parameters

- `--data_path`: **Required** - Path to the dataset folder
- `--output_dir`: Output directory (default: "output")
- `--min_samples`: Minimum samples per word (default: 3)
- `--min_quality`: Minimum quality score (default: 0.3)
- `--random_seed`: Random seed for reproducibility (default: 42)

### Example

```bash
python main.py --data_path "C:\Users\username\Desktop\Data\clean" --output_dir "results" --min_samples 5 --min_quality 0.4
```

## Dataset Structure

The system expects the following folder structure:

```
data_path/
├── word1/
│   ├── audio_file_001.wav
│   ├── audio_file_002.wav
│   └── ...
├── word2/
│   ├── audio_file_003.wav
│   ├── audio_file_004.wav
│   └── ...
└── ...
```

## Speaker Mapping

The system automatically maps speakers based on file numbering:

- **أحمد**: Files 0-6
- **عاصم**: Files 7-13
- **هيفاء**: Files 14-20
- **أسيل**: Files 21-28
- **وسام**: Files 29-36

## Output Files

### CSV Files

- `complete_dataset_standalone.csv`: Full processed dataset with features
- `train_dataset_standalone.csv`: Training set
- `test_dataset_standalone.csv`: Test set

### Reports

- `dataset_statistics_standalone.json`: Dataset statistics and distributions
- `comprehensive_report_standalone.json`: Complete evaluation results
- `model_performance_summary_standalone.csv`: Model performance summary table

## Classes Overview

### StandaloneASRSystem

Main orchestrator class that coordinates all components and runs the complete pipeline.

### DataManager

Handles data loading, speaker mapping, quality filtering, and train/test splitting.

### AudioProcessor

Provides audio enhancement techniques including noise reduction and signal preprocessing.

### FeatureExtractor

Extracts comprehensive audio features for machine learning.

### QualityAnalyzer

Analyzes audio quality using multiple metrics including SNR estimation.

### ModelTrainer

Trains and evaluates multiple machine learning models with cross-validation.

## Performance Metrics

The system provides:

- **Cross-validation scores** with standard deviation
- **Test set accuracy** for generalization assessment
- **Overfitting analysis** (CV score - Test score)
- **Detailed classification reports** per model

## Quality Control

### Audio Quality Filtering

- Signal-to-noise ratio assessment
- Dynamic range analysis
- Clipping detection
- Silence ratio evaluation
- Duration quality scoring

### Data Quality Assurance

- Minimum samples per word requirement
- Speaker-independent evaluation
- Feature validation and cleaning
- Comprehensive error handling
