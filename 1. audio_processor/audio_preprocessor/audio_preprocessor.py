import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from audio_quality_analyzer import AudioQualityAnalyzer
from noise_reducer import NoiseReducer
from volume_normalizer import VolumeNormalizer
from duration_adjuster import DurationAdjuster
from speech_enhancer import SpeechEnhancer
from quality_metrics_calculator import QualityMetricsCalculator
from processing_reporter import ProcessingReporter
from file_processor import FileProcessor
from speaker_identifier import SpeakerIdentifier


class FixedAudioPreprocessor:
    def __init__(self, input_path="C:/Users/فاطمة الزهراني/Desktop/ابحاث/الداون/Data/clean", 
                 output_path="processed_audio"):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)
        
        self.target_sr = 16000
        self.target_duration = 3.0
        self.target_length = int(self.target_sr * self.target_duration)
        
        self.processing_stats = {
            'total_files': 0,
            'processed_files': 0,
            'failed_files': 0,
            'noise_reduced_files': 0,
            'volume_normalized_files': 0,
            'duration_adjusted_files': 0
        }
        
        self.quality_metrics = []
        
        self.audio_quality_analyzer = AudioQualityAnalyzer()
        self.noise_reducer = NoiseReducer()
        self.volume_normalizer = VolumeNormalizer()
        self.duration_adjuster = DurationAdjuster(self.target_length)
        self.speech_enhancer = SpeechEnhancer()
        self.quality_metrics_calculator = QualityMetricsCalculator()
        self.processing_reporter = ProcessingReporter(self.quality_metrics_calculator)
        self.speaker_identifier = SpeakerIdentifier()
        self.file_processor = FileProcessor(
            self.audio_quality_analyzer, self.noise_reducer, self.volume_normalizer,
            self.duration_adjuster, self.speech_enhancer, self.target_sr
        )
    
    def process_all_files(self):
        print("Starting audio file processing...")
        print("=" * 60)
        
        all_files = []
        for word_dir in self.input_path.iterdir():
            if word_dir.is_dir():
                word = word_dir.name
                for audio_file in word_dir.glob("*.wav"):
                    file_number = self.speaker_identifier.extract_file_number(audio_file.stem)
                    speaker = self.speaker_identifier.get_speaker_from_number(file_number)
                    
                    all_files.append({
                        'input_file': audio_file,
                        'word': word,
                        'speaker': speaker
                    })
        
        self.processing_stats['total_files'] = len(all_files)
        print(f"Found {len(all_files)} files to process")
        
        for i, file_info in enumerate(all_files):
            input_file = file_info['input_file']
            word = file_info['word']
            speaker = file_info['speaker']
            
            output_file = self.output_path / word / f"{speaker}_{input_file.stem}_processed.wav"
            
            success, quality_info = self.file_processor.process_single_file(
                input_file, output_file, word, speaker, self.processing_stats
            )
            
            if success and quality_info:
                self.quality_metrics.append(quality_info)
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(all_files)} files...")
        
        print("All files processed successfully!")
        self.processing_reporter.print_processing_summary(self.processing_stats)
    
    def run_complete_processing(self):
        print("Advanced Audio Processor for Children with Down Syndrome")
        print("=" * 60)
        
        self.process_all_files()
        
        self.processing_reporter.analyze_quality_improvements(self.quality_metrics)
        
        settings = {
            'target_sr': self.target_sr,
            'target_duration': self.target_duration,
            'target_length': self.target_length
        }
        
        self.processing_reporter.save_processing_report(
            self.processing_stats, self.quality_metrics, settings
        )
        
        processed_df = self.processing_reporter.create_processed_dataset_csv(self.quality_metrics)
        
        print("\n" + "=" * 60)
        print("Complete processing finished successfully!")
        print("Generated files:")
        print("• processed_audio/ - processed audio data folder")
        print("• processed_dataset.csv - processed data file")
        print("• audio_processing_report.json - processing report")
        
        return processed_df