# Advanced Audio Preprocessor for Speech Recognition

A comprehensive audio processing pipeline designed to enhance audio quality for children with Down syndrome speech recognition research. This tool addresses noise reduction, volume normalization, duration adjustment, and speech enhancement.

## Installation

1. Install required packages:

```bash
pip install -r requirements.txt
```

## Usage

1. Update input/output paths in `audio_preprocessor.py`
2. Run the processing pipeline:

```bash
python main.py
```

## Features

### Audio Quality Analysis

- **Signal-to-Noise Ratio (SNR)** calculation
- **Clipping detection** and measurement
- **Silence ratio** analysis
- **Spectral characteristics** analysis
- **Zero-crossing rate** computation

### Noise Reduction

- **Low-pass filtering** to remove high-frequency noise
- **High-pass filtering** to remove low-frequency artifacts
- **Spectral gating** for advanced noise reduction
- **Adaptive thresholding** based on signal characteristics

### Volume Normalization

- **DC offset removal**
- **RMS energy normalization**
- **Peak limiting** to prevent clipping
- **Dynamic compression** for consistent levels

### Duration Adjustment

- **Smart truncation** from center for long files
- **Intelligent padding** for short files
- **File repetition** for very short recordings
- **Consistent target duration** across all files

### Speech Enhancement

- **Pre-emphasis filtering** for high-frequency clarity
- **Dynamic range compression** for weak signals
- **Adaptive gain control** for optimal levels

## Output Files

- `output_files/processed_audio/` - Directory containing enhanced audio files
- `output_files/processed_dataset.csv` - Dataset with quality metrics
- `output_files/audio_processing_report.json` - Comprehensive processing report

## Quality Metrics

- **RMS Energy**: Overall signal strength
- **SNR**: Signal-to-noise ratio in dB
- **Clipping Ratio**: Percentage of clipped samples
- **Silence Ratio**: Percentage of silent portions
- **Spectral Centroid**: Frequency distribution center
- **Spectral Bandwidth**: Frequency spread
- **Zero Crossing Rate**: Signal variability measure

## Classes Overview

1. **AudioQualityAnalyzer**: Comprehensive audio quality assessment
2. **NoiseReducer**: Multi-stage noise reduction algorithms
3. **VolumeNormalizer**: Advanced volume normalization techniques
4. **DurationAdjuster**: Smart duration standardization
5. **SpeechEnhancer**: Speech clarity improvement
6. **QualityMetricsCalculator**: Overall quality scoring system
7. **ProcessingReporter**: Report generation and analysis
8. **FileProcessor**: Individual file processing coordination
9. **SpeakerIdentifier**: Speaker identification from filenames
10. **FixedAudioPreprocessor**: Main processing orchestrator

## Processing Pipeline

1. **Input Validation**: Check file integrity and format
2. **Quality Analysis**: Initial quality assessment
3. **Sample Rate Conversion**: Standardize to target sample rate
4. **Noise Reduction**: Apply if SNR < 15dB
5. **Volume Normalization**: Standardize audio levels
6. **Speech Enhancement**: Improve clarity and intelligibility
7. **Duration Adjustment**: Standardize file duration
8. **Final Analysis**: Post-processing quality assessment
9. **Report Generation**: Comprehensive processing statistics

## Customization

- Modify target sample rate and duration in `audio_preprocessor.py`
- Adjust noise reduction thresholds in `noise_reducer.py`
- Customize volume normalization parameters in `volume_normalizer.py`
- Fine-tune speech enhancement settings in `speech_enhancer.py`
- Modify quality scoring weights in `quality_metrics_calculator.py`

## Performance Considerations

- **Batch Processing**: Processes files in batches for efficiency
- **Memory Management**: Optimized for large datasets
- **Error Handling**: Robust error recovery and reporting
- **Progress Tracking**: Real-time processing updates
