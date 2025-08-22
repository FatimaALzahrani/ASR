# Ultimate Speech Recognition Trainer

A comprehensive speech recognition training system designed for children with Down syndrome.

## Features

- **Advanced Audio Processing**: Noise reduction, silence removal, and audio enhancement
- **Multi-Speaker Support**: Handles 6 different speakers with quality assessment
- **Word Recognition**: Supports 20 Arabic words including fruits, colors, numbers, and family terms
- **Automated Training Pipeline**: Complete data loading, preprocessing, training, and evaluation
- **Results Export**: JSON format results with detailed speaker and word analysis

## Installation

1. Clone or download the project files
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the main script:

```bash
python main.py
```

When prompted, enter the path to your audio data folder, or leave empty to use default paths.

## Audio File Requirements

- Supported formats: WAV, MP3, M4A, FLAC
- File naming should include numbers for speaker identification
- Audio files should contain single words for recognition

## Speaker Configuration

The system recognizes 6 speakers:

- أحمد (Files 0-6): Medium quality
- عاصم (Files 7-13): High quality
- هيفاء (Files 14-20): Medium quality
- أسيل (Files 21-28): Low quality
- وسام (Files 29-36): Medium quality
- مجهول (Files 37+): Medium quality

## Supported Words

The system recognizes these Arabic words:

- **Fruits**: موز, تفاح, برتقال, عنب, فراولة
- **Colors**: أحمر, أزرق, أخضر, أصفر, أبيض
- **Numbers**: واحد, اثنين, ثلاثة, أربعة, خمسة
- **Family**: أب, أم, أخ, أخت, جد

## Output

The system generates:

- `ultimate_results/ultimate_results.json`: Complete results with detailed analysis
- `ultimate_results/summary.json`: Summary statistics
- Console output with real-time progress and final accuracy

## Technical Details

- **Audio Processing**: 16kHz sampling rate, 30-second duration normalization
- **Model**: OpenAI Whisper Small model
- **Data Split**: 60% training, 20% validation, 20% testing
- **Evaluation**: Speaker-based and word-based accuracy analysis
