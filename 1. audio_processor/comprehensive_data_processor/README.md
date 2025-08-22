# Audio Data Processor for Children with Down Syndrome

A comprehensive tool for analyzing audio data from children with Down syndrome, providing detailed statistics, visualizations, and quality assessments.

## Installation & Usage

1. Install required packages:

```bash
pip install -r requirements.txt
```

2. Run the main script:

```bash
python main_entry.py
```

## Output Files

- `comprehensive_analysis_report.json` - Detailed analysis report
- `comprehensive_data_analysis.png` - Visualization charts
- `training_dataset.csv` - Training data export
- `training_metadata.json` - Metadata for training

## Features

- **Speaker Management**: Organizes data by speaker with demographic info
- **Audio Analysis**: Extracts acoustic features and quality metrics
- **Statistics**: Comprehensive statistical analysis
- **Visualizations**: 6 different charts showing data distributions
- **Quality Assessment**: Automated quality scoring for recordings
- **Report Generation**: Detailed JSON reports with recommendations
- **Data Export**: CSV format ready for machine learning training

## Classes Overview

1. **SpeakerManager**: Maps file numbers to speakers and stores demographic data
2. **AudioFileScanner**: Scans directories for audio files and extracts metadata
3. **AudioAnalyzer**: Analyzes audio features using librosa
4. **StatisticsCalculator**: Computes comprehensive statistics
5. **DataVisualizer**: Creates publication-ready charts
6. **ReportGenerator**: Generates detailed analysis reports
7. **DataExporter**: Exports data for training purposes
8. **ComprehensiveDataProcessor**: Orchestrates the entire analysis pipeline

## Tirmenal Output

![tirmenal screenshot](\output_files\ScreanShot.png)
