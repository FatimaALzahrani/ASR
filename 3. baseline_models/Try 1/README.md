# Simple Audio Model Evaluator

A fast and simple evaluator for audio models on real data that handles unbalanced dataset issues.

## Overview

This project provides a comprehensive evaluation framework for audio classification models. It analyzes speaker performance, word recognition accuracy, and provides baseline model comparisons using simple feature extraction techniques.

## Features

- **Data Loading & Analysis**: Handles CSV datasets with audio file paths, words, and speaker information
- **Feature Extraction**: Extracts 22 audio features including MFCC, spectral features, and statistical measures
- **Model Evaluation**: Compares random, majority class, and k-NN baseline models
- **Speaker Analysis**: Analyzes performance per speaker
- **Word Analysis**: Analyzes word frequency and distribution
- **Comprehensive Reporting**: Generates detailed JSON reports with insights

## Requirements

```bash
pip install torch numpy pandas librosa scikit-learn
```

## Usage

### Command Line Usage

```bash
# Method 1: Pass CSV file as argument
python main.py training_dataset.csv

# Method 2: Show help
python main.py --help
```

### Programmatic Usage

```python
from main import SimpleEvaluator

# Initialize evaluator with your dataset path
evaluator = SimpleEvaluator("path/to/your/training_dataset.csv")

# Run complete evaluation
report = evaluator.run_evaluation()
```

### Configuration File Usage

You can also set the default path in `config.py`:

```python
class Config:
    DEFAULT_DATA_PATH = "path/to/your/training_dataset.csv"
    # ... other settings
```

Then run without arguments:

```bash
python main.py
```

### Quick Start

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare your data:**
   Make sure your CSV file has these columns:

   ```csv
   file_path,word,speaker
   /path/to/audio1.wav,hello,speaker1
   /path/to/audio2.wav,world,speaker2
   ```

3. **Run evaluation:**

   ```bash
   python main.py your_data.csv
   ```

4. **Check results:**
   The evaluation will generate `simple_evaluation_report.json`

### Dataset Format

Your CSV file should contain the following columns:

- `file_path`: Path to audio file
- `word`: The word being spoken
- `speaker`: Speaker identifier

### Example CSV:

```csv
file_path,word,speaker
/path/to/audio1.wav,hello,speaker1
/path/to/audio2.wav,world,speaker2
...
```

## Classes Overview

### DataLoader

- Loads and analyzes dataset
- Filters out rare words (< 3 samples)
- Sets up label encoders for words and speakers

### FeatureExtractor

- Extracts 22 audio features per file:
  - Energy (RMS)
  - Zero Crossing Rate
  - Spectral Centroid
  - Spectral Bandwidth
  - Spectral Rolloff
  - 13 MFCC coefficients
  - Statistical measures (std, max, min, duration)

### ModelEvaluator

- Evaluates three baseline models:
  - Random classifier
  - Majority class classifier
  - k-NN classifier (k=3)

### AudioAnalyzer

- Analyzes performance by speaker
- Analyzes word frequency distribution
- Provides accuracy metrics per speaker

### ReportGenerator

- Generates comprehensive JSON reports
- Creates insights and recommendations
- Prints formatted results

## Output

The evaluation generates:

- **Console output**: Real-time progress and results
- **simple_evaluation_report.json**: Comprehensive evaluation report

### Sample Output:

```
Using device: cpu
Simple Model Evaluator for Real Data
============================================================
Loading and analyzing data...
Loaded 1310 samples
Found 101 unique words
Found 5 speakers
Warning: 1 words have only one sample
Success: 100 words have 3+ samples
Filtered data to 1309 samples from common words
...
Random accuracy: 0.0092
Majority accuracy: 0.0214
k-NN accuracy: 0.2758
```

## Performance Metrics

The evaluator provides:

- **Accuracy scores** for different baseline models
- **Speaker-wise performance** analysis
- **Word frequency** distribution
- **Comparative insights** between models

## Data Filtering

The system automatically:

- Filters out words with less than 3 samples
- Maintains data balance across speakers
- Reports data quality statistics

## Configuration

The `config.py` file contains all configurable parameters:

```python
class Config:
    DEFAULT_DATA_PATH = "path/to/training_dataset.csv"
    AUDIO_SAMPLE_RATE = 16000
    AUDIO_DURATION = 3.0
    MIN_WORD_SAMPLES = 3
    KNN_NEIGHBORS = 3
    CV_FOLDS = 3
    MFCC_COEFFICIENTS = 13
    FEATURE_SIZE = 22
    OUTPUT_REPORT = "simple_evaluation_report.json"
    PROGRESS_INTERVAL = 100
```

You can modify these parameters to customize the evaluation process.

## Customization

Beyond the configuration file, you can also customize:

- Feature extraction algorithms in `FeatureExtractor`
- Model evaluation metrics in `ModelEvaluator`
- Analysis methods in `AudioAnalyzer`
- Report format and insights in `ReportGenerator`

## Error Handling

The system includes comprehensive error handling for common issues:

### File Not Found Error

```
Error: Data file not found: path/to/file.csv
```

**Solution**: Check that the file path is correct and the file exists.

### Missing Columns Error

```
Error: Missing required columns: ['word']. Required columns: ['file_path', 'word', 'speaker']
```

**Solution**: Ensure your CSV has the required columns: `file_path`, `word`, `speaker`.

### Invalid File Type Error

```
Error: Data file must be a CSV file.
```

**Solution**: Use a CSV file with `.csv` extension.

### No Data Path Provided

```
Error: Data path is required. Please provide it in one of these ways:
1. Pass as argument: python main.py data.csv
2. Set DEFAULT_DATA_PATH in config.py
3. Pass to SimpleEvaluator(data_path='data.csv')
```

**Solution**: Follow one of the suggested methods to provide the data path.

## Common Issues

1. **Audio files not found**: Make sure all audio file paths in your CSV are correct
2. **Permission errors**: Ensure you have read permissions for the audio files
3. **Memory issues**: For large datasets, consider processing in batches
4. **Library conflicts**: Use a virtual environment to avoid dependency conflicts

## Example Output

```
Using device: cpu
Simple Model Evaluator for Real Data
============================================================
Using data file: training_dataset.csv
Loading and analyzing data...
Loaded 1309 samples
Found 100 unique words
Found 5 speakers
...
Random accuracy: 0.0092
Majority accuracy: 0.0214
k-NN accuracy: 0.2758
```
