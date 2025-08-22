# Down Syndrome Speech Recognition Data Processor (Fixed Version)

A comprehensive and robust data preprocessing pipeline for Down Syndrome children's speech recognition research. This version includes improved error handling and smart splitting strategies for small datasets.

## 🔧 Recent Fixes

### Smart Data Splitting
- **Adaptive splitting strategies** based on word frequency
- **Improved handling** of words with 1-2 samples  
- **Better error recovery** for stratification failures
- **Speaker-specific splits** with robust fallback methods

### Enhanced Validation
- **Comprehensive dataset validation** before processing
- **Detailed diagnostic reports** for data issues
- **Splitting feasibility analysis** 
- **Interactive validation** with override options

### Better Error Handling
- **Graceful degradation** when splits fail
- **Detailed warning messages** for problematic words
- **Fallback strategies** for edge cases
- **Validation reports** saved automatically

## Project Structure

```
project/
├── config.py               # Configuration settings
├── audio_analyzer.py       # Audio quality analysis
├── data_scanner.py         # Audio file scanning
├── data_balancer.py        # Data balance analysis
├── data_splitter.py        # Smart train/test/validation splitting
├── data_validator.py       # Dataset validation and diagnostics
├── visualization.py        # Data visualization
├── data_processor.py       # Main processing pipeline
├── main.py                # Entry point
├── example_usage.py       # Usage examples
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## 🆕 New Features

### Dataset Validation
- Validates dataset structure and completeness
- Identifies data quality issues automatically
- Provides recommendations for improvement
- Checks splitting feasibility before processing

### Smart Splitting Strategies
- **Single-sample words**: Go to training only
- **Small words (2-4 samples)**: Simple train/test split
- **Medium words (5-15 samples)**: Train/validation/test with adaptive ratios
- **Large words (15+ samples)**: Full stratified splitting

### Robust Error Recovery
- Automatic fallback when stratification fails
- Graceful handling of speaker imbalance
- Detailed logging of split failures
- Continues processing despite individual word failures

## Project Structure

```
project/
├── config.py               # Configuration settings
├── audio_analyzer.py       # Audio quality analysis
├── data_scanner.py         # Audio file scanning
├── data_balancer.py        # Data balance analysis
├── data_splitter.py        # Train/test/validation splitting
├── visualization.py        # Data visualization
├── data_processor.py       # Main processing pipeline
├── main.py                # Entry point
├── requirements.txt       # Dependencies
└── README.md             # This file
```

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
python main.py --data-root /path/to/your/data

# Custom split ratios (helpful for small datasets)
python main.py --test-size 0.15 --val-size 0.05

# Minimum samples per word (default: 2)
python main.py --min-samples 1

# Skip interactive validation prompts (for automation)
python main.py --skip-validation

# Skip visualizations (faster processing)
python main.py --skip-visualizations

# Verbose output for debugging
python main.py --verbose

# Recommended for small datasets
python main.py --test-size 0.15 --val-size 0.05 --min-samples 1 --skip-validation
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
|---------|-----|--------|----------|----------------|------------|
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

## ✅ Verifying Successful Processing

### Check Output Files
```bash
# Main dataset files
ls data/processed/
# Should see: train.csv, test.csv, [validation.csv], mappings.json

# Speaker-specific files  
ls data/processed/speakers/
# Should see directories for each speaker

# Reports and analysis
ls data/reports/
# Should see: *.json reports and *.png visualizations
```

### Validate Data Integrity
```python
import pandas as pd

# Check main splits
train_df = pd.read_csv('data/processed/train.csv')
test_df = pd.read_csv('data/processed/test.csv')

print(f"Training samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")
print(f"Unique words: {train_df['word'].nunique()}")
print(f"Speakers: {train_df['speaker'].unique()}")
```

### Success Criteria
- ✅ **train.csv exists** and has >500 samples
- ✅ **test.csv exists** and has >100 samples
- ✅ **No critical errors** in console output
- ✅ **validation_report.json** shows status "GOOD" or better
- ✅ **Speaker directories** created for available speakers

### What If Validation Set is Empty?
This is **normal** for small datasets:
- Training and testing will work fine
- Model validation can use cross-validation instead
- Consider collecting more data for future versions

## 🚨 Troubleshooting Common Issues

### "Failed to split word" Warnings
These warnings are **normal** for small datasets and are handled automatically:
- Words with 1-2 samples cannot be split - they go to training only
- The system uses adaptive strategies based on sample count
- Processing continues successfully despite these warnings

### Empty Validation Set
If validation set is empty:
- This happens when most words have very few samples
- Training and testing will still work normally
- Consider using `--val-size 0.05` for smaller validation ratio

### "The least populated class" Errors
Fixed in this version with improved error handling:
- Automatic fallback to non-stratified splitting
- Better detection of stratification feasibility
- Graceful degradation maintains data integrity

### Speaker-Specific Split Failures
- System automatically tries multiple fallback strategies
- Processing continues with general splits if speaker splits fail
- Individual speaker failures don't stop the entire process

## 📊 Understanding the Output

### Expected Warnings (Normal)
```
Warning: Failed to split word اخ: With n_samples=2...
Warning: Failed to split word جد: With n_samples=1...
```
These are **expected** for small datasets and handled automatically.

### Processing Success Indicators
```
Data split summary:
  train: 881 samples, 100 words, 5 speakers
  validation: 90 samples, 67 words, 5 speakers  
  test: 241 samples, 67 words, 5 speakers
```

### Files Created
- ✅ `train.csv`, `validation.csv`, `test.csv` - Main datasets
- ✅ `speakers/[speaker]/train.csv` - Speaker-specific data
- ✅ `validation_report.json` - Diagnostic information
- ✅ `*.png` files - Visualization plots

## Development

### Extending the Pipeline

1. **New Analysis Modules**: Add to respective analyzer classes
2. **Custom Filters**: Implement in `DataBalancer`
3. **Additional Splits**: Extend `DataSplitter`
4. **New Visualizations**: Add to `DataVisualizer`

### Testing

```bash
# Test with sample data
python main.py --data-root sample_data --verbose

# Validate configuration
python -c "import config; print('Config OK')"
```

## 💡 Best Practices for Small Datasets

### Recommended Settings
For datasets with many words having few samples:
```bash
python main.py \
  --test-size 0.15 \
  --val-size 0.05 \
  --min-samples 1 \
  --skip-validation
```

### Data Collection Tips
- **Aim for 10+ samples per word** for robust splitting
- **Balance speakers** across all words when possible  
- **Maintain consistent recording quality** 
- **Consider data augmentation** for underrepresented words

### Interpretation Guidelines
- **Warnings about failed splits are normal** for small datasets
- **Focus on training set size** - should have majority of data
- **Empty validation sets are acceptable** for very small datasets
- **Use cross-validation** during model training if needed

### Quality Thresholds
Acceptable ranges for small datasets:
- **Training samples**: 70-85% of total
- **Test samples**: 15-25% of total  
- **Validation samples**: 0-10% of total (can be empty)
- **Words per split**: Not all words need to be in every split

## Contributing

1. Follow the modular structure
2. Add comprehensive documentation
3. Include error handling
4. Write unit tests for new features
5. Update configuration options as needed

## License

[Specify your license here]

## Citation

If you use this preprocessor in your research, please cite:
[Add your citation information]