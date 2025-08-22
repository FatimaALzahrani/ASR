import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from pathlib import Path
from recording_type_detector import RecordingTypeDetector
from advanced_noise_processor import AdvancedNoiseProcessor
from articulation_enhancer import ArticulationEnhancer
from gentle_volume_normalizer import GentleVolumeNormalizer
from smart_duration_manager import SmartDurationManager
from enhancement_metrics_calculator import EnhancementMetricsCalculator
from enhancement_report_generator import EnhancementReportGenerator
from audio_file_processor import AudioFileProcessor


class DownSyndromeAudioProcessor:
    def __init__(self, input_dir="C:/Users/فاطمة الزهراني/Desktop/ابحاث/الداون/Data/clean", output_dir="data/enhanced"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.reports_dir = Path("data/reports")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        self.target_sr = 16000
        self.target_duration = 3.0
        self.target_length = int(self.target_sr * self.target_duration)
        
        self.down_syndrome_params = {
            'low_freq_emphasis': True,
            'extended_silence_tolerance': True,
            'gentle_normalization': True,
            'articulation_enhancement': True,
            'breathing_noise_reduction': True
        }
        
        self.speaker_profiles = {
            'excellent': {
                'noise_reduction_strength': 0.3,
                'normalization_target': 0.15,
                'silence_threshold': 0.01
            },
            'good': {
                'noise_reduction_strength': 0.5,
                'normalization_target': 0.12,
                'silence_threshold': 0.015
            },
            'medium': {
                'noise_reduction_strength': 0.7,
                'normalization_target': 0.10,
                'silence_threshold': 0.02
            },
            'weak': {
                'noise_reduction_strength': 0.9,
                'normalization_target': 0.08,
                'silence_threshold': 0.025
            }
        }
        
        self.enhancement_stats = {
            'total_processed': 0,
            'mic_recordings_processed': 0,
            'computer_recordings_processed': 0,
            'noise_reduced': 0,
            'volume_enhanced': 0,
            'articulation_improved': 0,
            'quality_improvements': []
        }
        
        self.recording_detector = RecordingTypeDetector()
        self.noise_processor = AdvancedNoiseProcessor(self.down_syndrome_params)
        self.articulation_enhancer = ArticulationEnhancer(self.down_syndrome_params)
        self.volume_normalizer = GentleVolumeNormalizer()
        self.duration_manager = SmartDurationManager(self.target_length)
        self.metrics_calculator = EnhancementMetricsCalculator()
        self.report_generator = EnhancementReportGenerator(self.reports_dir)
        self.file_processor = AudioFileProcessor(
            self.target_sr, self.recording_detector, self.noise_processor,
            self.articulation_enhancer, self.volume_normalizer, 
            self.duration_manager, self.metrics_calculator
        )
    
    def process_all_files(self) -> bool:
        print("Starting enhanced audio processing...")
        print("=" * 60)
        
        speaker_qualities = {}
        transcripts_path = Path("data/processed/train.csv")
        if transcripts_path.exists():
            try:
                df = pd.read_csv(transcripts_path, encoding='utf-8')
                # Map Arabic quality levels to English
                quality_mapping = {
                    'ممتاز': 'excellent',
                    'جيد': 'good', 
                    'متوسط': 'medium',
                    'ضعيف': 'weak'
                }
                speaker_qualities = {
                    filename: quality_mapping.get(quality, 'medium') 
                    for filename, quality in zip(df['filename'], df['speech_quality'])
                }
                print(f"Loaded speaker information for {len(speaker_qualities)} speakers")
            except:
                print("Warning: Could not load speaker information, using default settings")
        
        all_files = []
        for word_folder in self.input_dir.iterdir():
            if word_folder.is_dir():
                for audio_file in word_folder.glob("*.wav"):
                    all_files.append({
                        'input_path': audio_file,
                        'word': word_folder.name,
                        'quality': speaker_qualities.get(audio_file.name, 'medium')
                    })
        
        if not all_files:
            print("Error: No audio files found")
            return False
        
        print(f"Found {len(all_files)} files for processing")
        
        success_count = 0
        for i, file_info in enumerate(all_files):
            input_path = file_info['input_path']
            word = file_info['word']
            quality = file_info['quality']
            
            profile = self.speaker_profiles.get(quality, self.speaker_profiles['medium'])
            output_path = self.output_dir / word / f"enhanced_{input_path.name}"
            
            if self.file_processor.process_single_audio(
                input_path, output_path, profile, self.enhancement_stats
            ):
                success_count += 1
            
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(all_files)} files...")
        
        print(f"Processing completed! ({success_count}/{len(all_files)} files succeeded)")
        
        self.report_generator.generate_enhancement_report(
            self.enhancement_stats, self.down_syndrome_params
        )
        
        return success_count > 0