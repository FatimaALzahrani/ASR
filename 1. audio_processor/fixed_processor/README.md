# Fixed Enhanced Down Syndrome Speech Data Processor

A robust and comprehensive data processing pipeline specifically designed for Down syndrome children speech recognition research. This processor handles various data challenges including imbalanced datasets, quality issues, and speaker variations.

## Installation

1. Install required packages:

```bash
pip install -r requirements.txt
```

## Usage

1. Update data paths in `down_syndrome_processor.py` if needed
2. Run the comprehensive processing pipeline:

```bash
python main.py
```

## Key Features

### Enhanced Error Handling

- **Robust File Processing**: Graceful handling of corrupted or invalid audio files
- **Flexible Duration Constraints**: Adjustable minimum (0.3s) and maximum (15s) duration limits
- **Failed File Tracking**: Comprehensive logging of processing failures with reasons
- **JSON Serialization**: Automatic conversion of NumPy types for JSON compatibility

### Smart Data Splitting

- **Sample-Aware Splitting**: Intelligent handling of words with few samples
  - 1 sample: Training only
  - 2 samples: Training (1) + Test (1)
  - 3 samples: Training (2) + Test (1)
  - 4 samples: Training (2) + Validation (1) + Test (1)
  - 5+ samples: Proportional splitting with stratification when possible

### Comprehensive Quality Analysis

- **Multi-Metric Assessment**: RMS energy, SNR, clipping ratio, silence ratio
- **Spectral Analysis**: Spectral centroid and zero-crossing rate
- **Composite Quality Score**: Overall quality rating (0.0 - 1.0)
- **Error-Resistant Calculations**: Fallback values for corrupted audio

### Advanced Data Balance Analysis

- **Distribution Analysis**: Word, speaker, and quality distributions
- **Sample Categorization**: Classification by sample count
  - Single sample words
  - Few sample words (2-4)
  - Moderate sample words (5-14)
  - Many sample words (15+)

### Speaker Profile Management

- **Demographic Information**: Age, gender, IQ level, speech quality
- **File Range Mapping**: Automatic speaker identification from file numbers
- **Quality-Based Processing**: Different processing parameters per speaker quality

## Data Quality Metrics

### Audio Quality Assessment

- **RMS Energy**: Signal strength measurement
- **Signal-to-Noise Ratio**: Quality of recording environment
- **Clipping Detection**: Digital distortion identification
- **Silence Analysis**: Non-speech content measurement
- **Spectral Features**: Frequency domain characteristics

### Quality Score Calculation

```python
score = base_score(0.5)
if energy_optimal: score += 0.2
if snr_good: score += 0.2
if no_clipping: score += 0.1
if low_silence: score += 0.1
```

## Output Files

### Processed Data

- `data/processed/train.csv` - Training dataset
- `data/processed/validation.csv` - Validation dataset (if sufficient samples)
- `data/processed/test.csv` - Test dataset
- `data/processed/mappings.json` - ID mappings for categorical data

### Reports and Analysis

- `data/reports/balance_analysis.json` - Detailed distribution analysis
- `data/reports/processing_report.json` - Comprehensive processing summary
- `data/reports/failed_files.csv` - List of files that failed processing
- `data/reports/data_analysis.png` - Visual analysis charts

## Classes Overview

1. **NumpyTypeConverter**: Handles JSON serialization of NumPy data types
2. **SpeakerIdentifierFixed**: Enhanced speaker identification with error handling
3. **AudioQualityAnalyzerFixed**: Robust audio quality assessment
4. **AudioFileScannerFixed**: Comprehensive file scanning with error tracking
5. **DataBalanceAnalyzer**: Statistical analysis of data distribution
6. **ImprovedDataSplitter**: Smart splitting algorithm for imbalanced data
7. **DataVisualizerFixed**: Comprehensive data visualization
8. **ProcessedDataSaver**: Robust data export with multiple formats
9. **FixedDownSyndromeProcessor**: Main coordinator orchestrating all components

## Processing Pipeline

1. **File Discovery**: Scan audio directories for valid WAV files
2. **Quality Assessment**: Analyze each file for audio quality metrics
3. **Speaker Identification**: Map files to speakers using filename patterns
4. **Error Handling**: Track and report failed processing attempts
5. **Balance Analysis**: Analyze data distribution across words and speakers
6. **Smart Splitting**: Create balanced train/validation/test splits
7. **Visualization**: Generate comprehensive analysis charts
8. **Data Export**: Save processed data in multiple formats

## Error Handling Strategy

### File Processing Errors

- **Empty Files**: Detected and logged
- **Duration Issues**: Files outside duration range excluded
- **Loading Errors**: Corrupted files tracked with error reasons
- **Quality Issues**: Poor quality files flagged but not excluded

### Data Processing Resilience

- **Missing Values**: Handled with sensible defaults
- **Type Conversion**: Automatic NumPy to Python type conversion
- **JSON Serialization**: Robust handling of complex data structures
- **Memory Management**: Efficient processing of large datasets

## Configuration Options

### Duration Constraints

```python
min_duration = 0.3  # seconds
max_duration = 15.0  # seconds
target_duration = 3.0  # seconds
```

### Quality Thresholds

```python
min_samples_per_word = 1
excluded_words = ['sleep']  # Words to skip
```

### Speaker Ranges

```python
speaker_ranges = {
    range(0, 7): 'Ahmed',
    range(7, 14): 'Asem',
    # ... additional mappings
}
```
