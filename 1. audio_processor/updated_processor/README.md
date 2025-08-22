# Updated Audio Data Processor for Speech Recognition

An updated data processing pipeline for children with Down syndrome speech recognition research, designed to handle word-based folder structures.

## Installation

1. Install required packages:

```bash
pip install -r requirements_updated.txt
```

## Usage

1. Update the data paths in `updated_data_processor.py`
2. Run the processing pipeline:

```bash
python main.py
```

## Expected Input Structure

```
audio_data/
├── word1/
│   ├── 001.wav
│   ├── 002.wav
│   └── ...
├── word2/
│   ├── 003.wav
│   ├── 004.wav
│   └── ...
└── ...
```

## Output Files

- `processed/train.csv` - Training dataset
- `processed/validation.csv` - Validation dataset
- `processed/test.csv` - Test dataset
- `processed/mappings.json` - ID mappings for words, speakers, quality levels
- `processed/statistics.json` - Comprehensive dataset statistics
- `transcripts/transcripts.csv` - Transcript data with word IDs
- `transcripts/speaker_info.csv` - Speaker demographic information

## Features

- **Speaker Identification**: Maps file numbers to specific speakers
- **Audio Validation**: Checks duration and format requirements
- **Smart Splitting**: Speaker-aware train/validation/test splits
- **Comprehensive Mapping**: Creates ID mappings for all categorical data
- **Statistical Analysis**: Detailed statistics on dataset composition
- **Multiple Export Formats**: CSV and JSON outputs for different use cases
- **Error Handling**: Robust processing with informative error messages

## Classes Overview

1. **SpeakerMapper**: Maps file numbers to speakers and manages demographic info
2. **AudioStructureScanner**: Scans word-based folder structure for audio files
3. **DatasetSplitter**: Creates train/validation/test splits with speaker awareness
4. **MappingCreator**: Generates ID mappings for categorical variables
5. **StatisticsComputer**: Calculates comprehensive dataset statistics
6. **DataSaver**: Handles all data export operations
7. **UpdatedDataProcessor**: Main orchestrator that coordinates all components

## Data Quality Checks

- Duration validation (0.5s - 30s)
- Audio format verification
- Speaker assignment validation
- Word folder structure validation
