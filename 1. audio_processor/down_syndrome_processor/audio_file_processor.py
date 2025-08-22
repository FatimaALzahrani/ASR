import librosa
import soundfile as sf
from pathlib import Path


class AudioFileProcessor:
    def __init__(self, target_sr, recording_detector, noise_processor, 
                 articulation_enhancer, volume_normalizer, duration_manager, 
                 metrics_calculator):
        self.target_sr = target_sr
        self.recording_detector = recording_detector
        self.noise_processor = noise_processor
        self.articulation_enhancer = articulation_enhancer
        self.volume_normalizer = volume_normalizer
        self.duration_manager = duration_manager
        self.metrics_calculator = metrics_calculator
    
    def process_single_audio(self, input_path: Path, output_path: Path, 
                           speaker_profile: dict, enhancement_stats: dict) -> bool:
        try:
            audio, sr = librosa.load(input_path, sr=None)
            
            if len(audio) == 0:
                print(f"Warning: Empty file: {input_path}")
                return False
            
            original_audio = audio.copy()
            
            if sr != self.target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
                sr = self.target_sr
            
            recording_type = self.recording_detector.detect_recording_type(audio, sr)
            
            audio = self.noise_processor.advanced_noise_reduction(
                audio, sr, recording_type, speaker_profile['noise_reduction_strength']
            )
            enhancement_stats['noise_reduced'] += 1
            
            audio = self.articulation_enhancer.enhance_articulation(audio, sr)
            enhancement_stats['articulation_improved'] += 1
            
            audio = self.volume_normalizer.gentle_normalization(
                audio, speaker_profile['normalization_target']
            )
            enhancement_stats['volume_enhanced'] += 1
            
            audio = self.duration_manager.smart_duration_adjustment(
                audio, sr, speaker_profile['silence_threshold']
            )
            
            if recording_type == 'microphone':
                enhancement_stats['mic_recordings_processed'] += 1
            else:
                enhancement_stats['computer_recordings_processed'] += 1
            
            if sr == self.target_sr:
                original_resampled = librosa.resample(
                    original_audio, 
                    orig_sr=librosa.load(input_path, sr=None)[1], 
                    target_sr=self.target_sr
                )
                min_length = min(len(original_resampled), len(audio))
                metrics = self.metrics_calculator.calculate_enhancement_metrics(
                    original_resampled[:min_length], audio[:min_length], sr
                )
                enhancement_stats['quality_improvements'].append(metrics)
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(output_path), audio, sr)
            
            enhancement_stats['total_processed'] += 1
            
            return True
            
        except Exception as e:
            print(f"Error processing {input_path}: {e}")
            return False