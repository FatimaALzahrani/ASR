# Speaker-Specific ASR System

A comprehensive speaker-specific automatic speech recognition system that trains both speaker-specific and global models for Arabic word recognition.

## Features

- **Advanced Audio Processing**: Pre-emphasis, normalization, silence trimming
- **Rich Feature Extraction**: MFCC, Mel-spectrogram, Chroma, Spectral features
- **Speaker-Specific Models**: Individual models for each speaker
- **Global Models**: Universal models trained on all speakers
- **Multiple ML Algorithms**: Random Forest, SVM, Gradient Boosting, etc.
- **Robust Data Handling**: Quality filtering and consistent feature preparation
- **Comprehensive Evaluation**: Detailed accuracy metrics and results saving

## Requirements

```bash
pip install librosa pandas numpy scikit-learn scipy matplotlib seaborn
```

## Usage

### Basic Usage

```bash
python main.py --data_path "path/to/your/audio/data"
```

### Advanced Usage

```bash
python main.py \
    --data_path "path/to/your/audio/data" \
    --output_dir "results" \
    --min_samples 3 \
    --random_seed 42
```

## Parameters

- `--data_path`: Path to the audio data folder (required)
- `--output_dir`: Output directory for results (default: 'output')
- `--min_samples`: Minimum samples per word (default: 3)
- `--random_seed`: Random seed for reproducibility (default: 42)

## Data Structure

Your audio data should be organized as follows:

```
data/
├── word1/
│   ├── 0.wav    # Speaker أحمد (files 0-6)
│   ├── 7.wav    # Speaker عاصم (files 7-13)
│   ├── 14.wav   # Speaker هيفاء (files 14-20)
│   ├── 21.wav   # Speaker أسيل (files 21-28)
│   └── 29.wav   # Speaker وسام (files 29-36)
├── word2/
│   └── ...
└── word3/
    └── ...
```

## Speaker Mapping

The system automatically maps audio files to speakers based on filename numbers:

- أحمد: files 0-6
- عاصم: files 7-13
- هيفاء: files 14-20
- أسيل: files 21-28
- وسام: files 29-36

## Output

The system generates:

1. **Detailed Results**: `final_speaker_specific_results.json`
2. **Summary Table**: `final_speaker_specific_summary.csv`

## Models Trained

### Speaker-Specific Models

- Random Forest
- Extra Trees
- Gradient Boosting
- SVM with RBF kernel
- Logistic Regression

### Global Models

- Global Random Forest
- Global Extra Trees
- Global Gradient Boosting
- Global SVM
- Global Logistic Regression

## Features Extracted

- **Temporal Features**: Duration, RMS energy, amplitude statistics
- **Spectral Features**: Centroid, rolloff, bandwidth, flatness
- **MFCC Features**: 13 coefficients with statistics and deltas
- **Mel-spectrogram**: 10 mel bands with statistics
- **Chroma Features**: 12 chroma bins
- **Spectral Contrast**: 7 contrast bands
- **Quality Metrics**: SNR and quality scores

## Configuration

Modify `config.py` to adjust:

- Audio parameters (sample rate, duration)
- Feature extraction parameters
- Model hyperparameters
- File processing settings

## Class Overview

### AudioProcessor

Handles audio preprocessing and feature extraction from audio files.

### DataLoader

Manages data loading, speaker mapping, and dataset preparation.

### ModelTrainer

Trains both speaker-specific and global machine learning models.

### Evaluator

Evaluates model performance and saves comprehensive results.

### SpeakerASRSystem

Main system class that orchestrates the complete pipeline.
