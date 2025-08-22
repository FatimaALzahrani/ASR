import librosa
import soundfile as sf


class FileProcessor:
    def __init__(self, audio_quality_analyzer, noise_reducer, volume_normalizer, 
                 duration_adjuster, speech_enhancer, target_sr):
        self.audio_quality_analyzer = audio_quality_analyzer
        self.noise_reducer = noise_reducer
        self.volume_normalizer = volume_normalizer
        self.duration_adjuster = duration_adjuster
        self.speech_enhancer = speech_enhancer
        self.target_sr = target_sr
    
    def process_single_file(self, input_file, output_file, word, speaker, processing_stats):
        try:
            audio, sr = librosa.load(input_file, sr=None)
            
            if len(audio) == 0:
                print(f"Warning: Empty file: {input_file}")
                return False, None
            
            initial_quality = self.audio_quality_analyzer.analyze_audio_quality(audio, sr)
            
            if sr != self.target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
                sr = self.target_sr
            
            processed_audio = audio.copy()
            
            if initial_quality['snr'] < 15:
                processed_audio = self.noise_reducer.simple_noise_reduction(processed_audio, sr)
                processing_stats['noise_reduced_files'] += 1
            
            processed_audio = self.volume_normalizer.normalize_volume(processed_audio)
            processing_stats['volume_normalized_files'] += 1
            
            processed_audio = self.speech_enhancer.enhance_speech(processed_audio, sr)
            
            if len(processed_audio) != self.duration_adjuster.target_length:
                processed_audio = self.duration_adjuster.adjust_duration(processed_audio, sr)
                processing_stats['duration_adjusted_files'] += 1
            
            final_quality = self.audio_quality_analyzer.analyze_audio_quality(processed_audio, sr)
            
            output_file.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(output_file), processed_audio, sr)
            
            quality_info = {
                'file_path': str(input_file),
                'output_path': str(output_file),
                'word': word,
                'speaker': speaker,
                'initial_quality': initial_quality,
                'final_quality': final_quality,
                'improvement': {}
            }
            
            if initial_quality and final_quality:
                quality_info['improvement'] = {
                    'snr_improvement': final_quality['snr'] - initial_quality['snr'],
                    'energy_change': final_quality['rms_energy'] / (initial_quality['rms_energy'] + 1e-10),
                    'clipping_reduction': initial_quality['clipping_ratio'] - final_quality['clipping_ratio']
                }
            
            processing_stats['processed_files'] += 1
            return True, quality_info
            
        except Exception as e:
            print(f"Failed to process {input_file}: {e}")
            processing_stats['failed_files'] += 1
            return False, None