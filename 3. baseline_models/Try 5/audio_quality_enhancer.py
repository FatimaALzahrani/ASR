import os
import numpy as np
import librosa
import soundfile as sf
import warnings
warnings.filterwarnings('ignore')

from speech_boundary_detector import SpeechBoundaryDetector
from noise_reducer import NoiseReducer
from volume_normalizer import VolumeNormalizer

class AudioQualityEnhancer:
    def __init__(self, target_sr=22050, target_duration=None):
        self.target_sr = target_sr
        self.target_duration = target_duration
        self.boundary_detector = SpeechBoundaryDetector()
        self.noise_reducer = NoiseReducer()
        self.volume_normalizer = VolumeNormalizer()
        
    def smart_trim(self, audio, sr):
        start_time, end_time = self.boundary_detector.detect_boundaries(audio, sr)
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        trimmed_audio = audio[start_sample:end_sample]
        return trimmed_audio, start_time, end_time
    
    def enhance_audio_file(self, input_path, output_path, apply_noise_reduction=True, 
                          apply_normalization=True, apply_smart_trim=True):
        try:
            audio, sr = librosa.load(input_path, sr=self.target_sr)
            original_duration = len(audio) / sr
            enhanced_audio = audio.copy()
            processing_info = {
                'original_duration': original_duration,
                'original_rms': np.sqrt(np.mean(audio**2)),
                'processing_steps': []
            }
            
            if apply_smart_trim:
                enhanced_audio, start_time, end_time = self.smart_trim(enhanced_audio, sr)
                processing_info['processing_steps'].append('smart_trim')
                processing_info['trim_start'] = start_time
                processing_info['trim_end'] = end_time
                processing_info['trimmed_duration'] = len(enhanced_audio) / sr
            
            if apply_noise_reduction:
                enhanced_audio = self.noise_reducer.remove_noise(enhanced_audio, sr)
                processing_info['processing_steps'].append('noise_reduction')
            
            if apply_normalization:
                enhanced_audio = self.volume_normalizer.normalize(enhanced_audio)
                processing_info['processing_steps'].append('normalization')
                processing_info['final_rms'] = np.sqrt(np.mean(enhanced_audio**2))
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            sf.write(output_path, enhanced_audio, sr)
            
            processing_info['success'] = True
            processing_info['final_duration'] = len(enhanced_audio) / sr
            
            return processing_info
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'original_duration': 0,
                'processing_steps': []
            }