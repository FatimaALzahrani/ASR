# Specialized Audio Processor for Down Syndrome Children

A sophisticated audio processing pipeline specifically designed for children with Down syndrome speech recognition research. This processor addresses unique challenges in Down syndrome speech patterns through targeted enhancement techniques.

## Installation

1. Install required packages:

```bash
pip install -r requirements.txt
```

## Usage

1. Update input/output paths in `down_syndrome_audio_processor.py` if needed
2. Run the specialized processing pipeline:

```bash
python main.py
```

## Down Syndrome-Specific Features

### Recording Type Detection

- **Microphone vs Computer**: Automatic detection based on frequency analysis
- **Adaptive Processing**: Different enhancement strategies for each recording type
- **Noise Profile Analysis**: Tailored noise reduction based on recording characteristics

### Advanced Noise Processing

- **Microphone Noise Reduction**: Addresses background noise, breathing sounds, and artifacts
- **Computer Artifact Removal**: Handles digital distortions and compression artifacts
- **Spectral Subtraction**: Advanced frequency-domain noise reduction
- **Breathing Noise Reduction**: Specific filtering for respiratory sounds common in Down syndrome speech

### Articulation Enhancement

- **Low Frequency Emphasis**: Enhances important speech frequencies (1-4 kHz)
- **Weak Signal Amplification**: Boosts quiet speech segments
- **Pre-emphasis Filtering**: Improves speech clarity and intelligibility
- **Dynamic Range Compression**: Balances loud and soft speech portions

### Gentle Volume Normalization

- **Progressive Normalization**: Gradual adjustment to prevent sudden changes
- **Soft Limiting**: Prevents clipping while maintaining naturalness
- **RMS-based Targeting**: Consistent energy levels across recordings
- **Hyperbolic Tangent Compression**: Natural-sounding dynamic control

### Smart Duration Management

- **Activity Detection**: Identifies speech vs silence regions
- **Intelligent Trimming**: Preserves important speech content
- **Smart Padding**: Natural transitions for short recordings
- **Fade In/Out**: Smooth audio boundaries

## Speaker Profiles

The processor adapts processing parameters based on speech quality levels:

### Excellent Speakers

- Minimal noise reduction (30% strength)
- Higher normalization target (0.15 RMS)
- Lower silence threshold (0.01)

### Good Speakers

- Moderate noise reduction (50% strength)
- Standard normalization (0.12 RMS)
- Standard silence threshold (0.015)

### Medium Speakers

- Strong noise reduction (70% strength)
- Conservative normalization (0.10 RMS)
- Higher silence tolerance (0.02)

### Weak Speakers

- Maximum noise reduction (90% strength)
- Gentle normalization (0.08 RMS)
- Maximum silence tolerance (0.025)

## Enhancement Metrics

### Signal Quality Measures

- **SNR Improvement**: Signal-to-noise ratio enhancement in dB
- **RMS Improvement**: Energy normalization factor
- **Clipping Reduction**: Distortion elimination percentage
- **Spectral Analysis**: Frequency content optimization

### Processing Statistics

- Total files processed
- Recording type distribution
- Enhancement technique applications
- Quality improvement averages

## Output Files

- `data/enhanced/` - Directory with enhanced audio files
- `data/reports/audio_enhancement_report.json` - Comprehensive processing report
- Enhanced files maintain original structure with "enhanced\_" prefix

## Classes Overview

1. **RecordingTypeDetector**: Analyzes frequency characteristics to classify recording type
2. **AdvancedNoiseProcessor**: Multi-stage noise reduction with adaptive algorithms
3. **ArticulationEnhancer**: Speech clarity improvement for Down syndrome characteristics
4. **GentleVolumeNormalizer**: Careful volume adjustment preserving speech naturalness
5. **SmartDurationManager**: Intelligent duration standardization with content preservation
6. **EnhancementMetricsCalculator**: Comprehensive quality improvement measurement
7. **EnhancementReportGenerator**: Detailed analysis and reporting
8. **AudioFileProcessor**: Individual file processing coordination
9. **DownSyndromeAudioProcessor**: Main orchestrator with specialized parameters

## Processing Pipeline

1. **Input Validation**: File integrity and format verification
2. **Recording Classification**: Microphone vs computer detection
3. **Sample Rate Normalization**: Standardize to 16kHz
4. **Adaptive Noise Reduction**: Type-specific noise removal
5. **Articulation Enhancement**: Speech clarity improvement
6. **Gentle Normalization**: Volume standardization
7. **Smart Duration Adjustment**: Intelligent length standardization
8. **Quality Assessment**: Enhancement effectiveness measurement
9. **Report Generation**: Comprehensive processing analysis

## Down Syndrome Speech Considerations

### Common Challenges Addressed

- **Reduced Vocal Tract Size**: Frequency emphasis adjustments
- **Hypotonia Effects**: Gentle processing to preserve weak signals
- **Breathing Patterns**: Specialized respiratory noise reduction
- **Articulation Difficulties**: Targeted speech enhancement
- **Variable Volume**: Adaptive normalization strategies

### Processing Adaptations

- **Extended Silence Tolerance**: Accommodates speech timing variations
- **Low Frequency Emphasis**: Preserves important speech information
- **Gentle Dynamics**: Avoids aggressive processing that could distort speech
- **Breathing Artifact Removal**: Specific filtering for respiratory sounds
