# Down Syndrome Speech Recognition Data Processor

A comprehensive data preprocessing pipeline for Down Syndrome children's speech recognition research. This tool analyzes, processes, and prepares speech data for machine learning model training.

## Features

### 🔍 **Comprehensive Audio Analysis**

- Audio quality assessment (SNR, RMS energy, clipping detection)
- Spectral feature extraction
- Duration validation
- Speaker identification from filenames

### 📊 **Data Balance Analysis**

- Word frequency distribution
- Speaker sample distribution
- Speech quality categorization
- Rare and common word identification

### 🎯 **Smart Data Splitting**

- Stratified train/validation/test splits
- Speaker-specific dataset creation
- Balanced distribution maintenance
- Configurable split ratios

### 📈 **Rich Visualizations**

- Data distribution plots
- Quality analysis charts
- Balance assessment graphs
- Speaker performance comparisons

## Installation

1. Install required packages:

```bash
pip install -r requirements.txt
```

2. Organize your data structure:

```
data/
├── clean/
│   ├── word1/
│   │   ├── 0.wav
│   │   ├── 1.wav
│   │   └── ...
│   ├── word2/
│   │   ├── 7.wav
│   │   ├── 8.wav
│   │   └── ...
│   └── ...
└── [output directories will be created]
```

## Usage

### Basic Usage

```bash
python main.py
```

### Advanced Options

```bash
# Custom data directory
python main.py --data-root /path/to/my/data

# Custom split ratios
python main.py --test-size 0.25 --val-size 0.15

# Minimum samples per word
python main.py --min-samples 3

# Skip visualizations (faster processing)
python main.py --skip-visualizations

# Verbose output
python main.py --verbose
```

## Configuration

Edit `config.py` to customize:

### Audio Processing

- `TARGET_SAMPLE_RATE`: Target sampling rate (default: 16000 Hz)
- `TARGET_DURATION`: Target audio duration (default: 3.0 seconds)
- `MIN_DURATION`: Minimum acceptable duration (default: 0.5 seconds)
- `MAX_DURATION`: Maximum acceptable duration (default: 10.0 seconds)

### Data Splitting

- `TEST_SIZE`: Test set ratio (default: 0.2)
- `VALIDATION_SIZE`: Validation set ratio (default: 0.1)

### Speaker Information

- `SPEAKER_RANGES`: Filename ranges for speaker identification
- `SPEAKER_INFO`: Demographics and characteristics for each speaker

## Output Files

### Processed Data

- `data/processed/train.csv`: Training dataset
- `data/processed/validation.csv`: Validation dataset
- `data/processed/test.csv`: Test dataset
- `data/processed/mappings.json`: Word and speaker mappings
- `data/processed/speakers/[speaker]/`: Speaker-specific datasets

### Reports and Analysis

- `data/reports/processing_report.json`: Complete processing statistics
- `data/reports/balance_analysis.json`: Data balance metrics
- `data/reports/failed_files.csv`: Failed file processing log
- `data/reports/data_analysis.png`: Comprehensive data visualizations
- `data/reports/balance_analysis.png`: Balance analysis plots
- `data/reports/quality_analysis.png`: Audio quality analysis plots

## Speaker Information

The system recognizes 5 speakers based on filename ranges:

| Speaker | Age | Gender | IQ Level | Speech Quality | File Range |
| ------- | --- | ------ | -------- | -------------- | ---------- |
| أحمد    | 10  | Male   | 38       | Poor           | 0-6        |
| عاصم    | 11  | Male   | 55       | Excellent      | 7-13       |
| هيفاء   | 7   | Female | 64       | Good           | 14-20      |
| أسيل    | 16  | Female | 40       | Poor           | 21-28      |
| وسام    | 6   | Male   | Medium   | Medium         | 29-36      |

## Audio Quality Metrics

The system analyzes multiple audio quality metrics:

- **RMS Energy**: Overall signal strength
- **Signal-to-Noise Ratio (SNR)**: Audio clarity measurement
- **Clipping Ratio**: Digital distortion detection
- **Silence Ratio**: Non-speech content percentage
- **Spectral Features**: Frequency domain characteristics
- **Zero Crossing Rate**: Speech/silence discrimination
- **Overall Quality Score**: Composite quality rating (0-1)

## Data Validation

The processor includes robust validation:

- File format verification (.wav files)
- Duration range checking
- Speaker identification validation
- Audio loading error handling
- Quality threshold enforcement

## Customization

### Adding New Speakers

1. Update `SPEAKER_RANGES` in `config.py`
2. Add speaker info to `SPEAKER_INFO`
3. Ensure filename numbering follows the pattern

### Modifying Quality Thresholds

Edit `QUALITY_THRESHOLDS` in `config.py` to adjust:

- Minimum/maximum energy levels
- SNR requirements
- Clipping tolerance
- Silence ratio limits

### Custom Visualizations

Extend the `DataVisualizer` class in `visualization.py` to add:

- New plot types
- Custom metrics visualization
- Additional analysis charts

## Performance Considerations

- **Memory Usage**: Large datasets may require chunked processing
- **Processing Time**: Quality analysis is computationally intensive
- **Storage**: Visualizations and reports require additional disk space
- **CPU Usage**: Librosa audio analysis benefits from multiple cores
