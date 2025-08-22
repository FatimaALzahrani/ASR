# Whisper Speech Recognition Evaluator

A comprehensive evaluation framework for speech recognition using OpenAI's Whisper model, specifically designed for real-world Arabic speech data analysis.

## Overview

This project provides a modular evaluation system for assessing Whisper model performance on Arabic speech recognition tasks. It includes comprehensive analysis by speaker, word accuracy, and detailed performance metrics.

## Features

- **Whisper Model Integration**: Uses OpenAI's Whisper-small model for Arabic speech recognition
- **Comprehensive Evaluation**: Evaluates on both quick samples and full test datasets
- **Speaker Analysis**: Analyzes performance by individual speakers
- **Word Analysis**: Evaluates accuracy for specific words
- **Detailed Reporting**: Generates comprehensive JSON reports with insights
- **Modular Architecture**: Clean separation of concerns with dedicated classes

## Requirements

```bash
pip install -r requirements.txt
```

### Key Dependencies:

- **PyTorch**: Deep learning framework
- **Transformers**: Hugging Face transformers for Whisper
- **Librosa**: Audio processing
- **Evaluate**: Metrics computation (WER)
- **Pandas**: Data manipulation
- **Tqdm**: Progress bars

## Usage

### Super Quick Start

For first-time users or testing:

```bash
# Create sample data structure
python quick_start.py --sample-data

# Add your audio files to sample_audio/ directory
# Then run:
python quick_start.py
```

This will:

- Check all requirements
- Validate your setup
- Create sample data if needed
- Guide you through any issues

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Configure data paths:**
   Update `config.py` with your data directory:

   ```python
   DATA_DIR = "path/to/your/processed/data"
   ```

3. **Validate setup (recommended):**
   ```bash
   python debug_utils.py
   ```

### Using Makefile (Optional)

For convenience, you can use the provided Makefile:

```bash
# Install dependencies
make install

# Validate system and data
make validate

# Run evaluation
make evaluate

# Run example
make example

# Clean generated files
make clean

# Setup and run everything
make setup-and-run
```

### Command Line Tools

After installation with `pip install .`, you can use:

```bash
# Run evaluation
whisper-evaluate

# Validate setup
whisper-validate
```

### Data Format

Your data directory should contain:

```
processed/
├── train.csv       # Training data with columns: audio_path, text, speaker
├── test.csv        # Test data with same format
└── mappings.json   # Metadata about words and speakers
```

#### CSV Format:

```csv
audio_path,text,speaker,quality
/path/to/audio1.wav,hello,speaker1,high
/path/to/audio2.wav,world,speaker2,medium
```

#### Mappings.json Format:

```json
{
  "num_words": 100,
  "num_speakers": 5,
  "word_to_id": {...},
  "speaker_to_id": {...}
}
```

## Classes Overview

### DataLoader

- Loads training and test datasets
- Manages data sampling for quick evaluation
- Provides metadata access

### AudioProcessor

- Preprocesses audio files for Whisper input
- Handles audio normalization and format conversion
- Manages error handling for corrupted files

### WhisperEvaluator

- Initializes and manages Whisper model
- Performs batch evaluation on datasets
- Computes Word Error Rate (WER) and accuracy metrics

### ResultAnalyzer

- Analyzes performance by speaker
- Evaluates word-level accuracy
- Provides detailed performance breakdowns

### ResultsSaver

- Saves comprehensive results to JSON
- Creates summary reports
- Prints formatted output

## Configuration

All settings are centralized in `config.py`:

```python
class Config:
    WHISPER_MODEL_NAME = "openai/whisper-small"  # Model to use
    LANGUAGE = "ar"                              # Arabic language
    SAMPLE_RATE = 16000                         # Audio sample rate
    MAX_LENGTH = 50                             # Max generation length
    SAMPLE_SIZE = 50                            # Quick sample size
    TOP_WORDS_COUNT = 10                        # Words to analyze
    MIN_WORD_SAMPLES = 3                        # Min samples per word
```

## Output

The evaluation generates:

### 1. Console Output

```
Using device: cuda
Loading model: openai/whisper-small
Loading data...
   Test data: 59 samples
   Sample data: 50 samples

Starting comprehensive evaluation...
==================================================
Evaluating Quick sample...
Results for Quick sample:
   Successful files: 50/50
   Word Error Rate (WER): 1.020
   Accuracy: -2.0%
```

### 2. JSON Reports

**`results/real_training_results.json`** - Complete detailed results
**`results/results_summary.json`** - Executive summary

Example summary:

```json
{
  "overall_performance": {
    "test_accuracy": 0.841,
    "test_wer": 0.159,
    "sample_accuracy": 0.98,
    "sample_wer": 0.02
  },
  "best_speaker": "Speaker_A",
  "worst_speaker": "Speaker_B",
  "best_word": "hello",
  "worst_word": "complicated_word"
}
```

## Performance Metrics

- **Word Error Rate (WER)**: Primary metric for speech recognition
- **Accuracy**: 1 - WER, showing correct transcription rate
- **Success Rate**: Percentage of files processed without errors
- **Speaker-wise Analysis**: Performance breakdown per speaker
- **Word-wise Analysis**: Accuracy for most common words

## Customization

### Model Configuration

Change the Whisper model variant in `config.py`:

```python
WHISPER_MODEL_NAME = "openai/whisper-base"  # or whisper-large
```

### Evaluation Parameters

Adjust evaluation settings:

```python
SAMPLE_SIZE = 100        # Larger quick sample
TOP_WORDS_COUNT = 20     # More words to analyze
MIN_WORD_SAMPLES = 5     # Higher threshold for word analysis
```

### Language Support

For other languages, update:

```python
LANGUAGE = "en"  # English
LANGUAGE = "fr"  # French
```
