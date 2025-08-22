import os
import json
from typing import Dict, List, Optional, Any
import logging

from acoustic_model import AcousticModel
from language_model import LanguageModel
from auto_correction import AutoCorrection

logger = logging.getLogger(__name__)

class ProfessionalASRSystem:
    def __init__(self, sample_rate: int = 22050, duration: float = 3.0):
        self.acoustic_model = AcousticModel(sample_rate, duration)
        self.language_model = LanguageModel()
        self.auto_correction = None
        self.is_trained = False
        
        logger.info("Professional ASR system initialized")
    
    def train(self, audio_data_path: str, text_corpus: List[str] = None, 
              min_samples_per_word: int = 3) -> Dict[str, Any]:
        logger.info("Starting complete ASR system training")
        
        acoustic_results = self.acoustic_model.train_acoustic_models(
            audio_data_path, min_samples_per_word
        )
        
        if text_corpus:
            self.language_model.train(text_corpus)
        else:
            words = list(self.acoustic_model.label_encoder.classes_)
            simple_corpus = [' '.join(words) for _ in range(10)]
            self.language_model.train(simple_corpus)
        
        self.auto_correction = AutoCorrection(self.language_model)
        
        self.is_trained = True
        
        logger.info("Complete ASR system training completed")
        
        return {
            'acoustic_model': acoustic_results,
            'language_model_vocab_size': len(self.language_model.vocabulary),
            'auto_correction_ready': self.auto_correction is not None
        }
    
    def recognize_speech(self, audio_file_path: str, speaker: Optional[str] = None,
                        apply_correction: bool = True) -> Dict[str, Any]:
        if not self.is_trained:
            raise ValueError("System must be trained before recognition")
        
        try:
            predicted_word, acoustic_confidence = self.acoustic_model.predict(
                audio_file_path, speaker
            )
            
            result = {
                'acoustic_prediction': predicted_word,
                'acoustic_confidence': acoustic_confidence,
                'final_prediction': predicted_word,
                'final_confidence': acoustic_confidence,
                'corrections_applied': False
            }
            
            if apply_correction and self.auto_correction:
                corrected_word, correction_confidence = self.auto_correction.correct_word(
                    predicted_word
                )
                
                if corrected_word != predicted_word:
                    final_confidence = (acoustic_confidence + correction_confidence) / 2
                    
                    result.update({
                        'final_prediction': corrected_word,
                        'final_confidence': final_confidence,
                        'corrections_applied': True,
                        'correction_confidence': correction_confidence
                    })
            
            return result
            
        except Exception as e:
            logger.error(f"Speech recognition error: {str(e)}")
            return {
                'acoustic_prediction': 'unknown',
                'acoustic_confidence': 0.0,
                'final_prediction': 'unknown',
                'final_confidence': 0.0,
                'corrections_applied': False,
                'error': str(e)
            }
    
    def save_system(self, base_path: str):
        os.makedirs(base_path, exist_ok=True)
        
        acoustic_path = os.path.join(base_path, 'acoustic_model.pkl')
        self.acoustic_model.save_model(acoustic_path)
        
        language_path = os.path.join(base_path, 'language_model.pkl')
        self.language_model.save_model(language_path)
        
        metadata = {
            'is_trained': self.is_trained,
            'sample_rate': self.acoustic_model.sample_rate,
            'duration': self.acoustic_model.duration,
            'vocabulary_size': len(self.language_model.vocabulary),
            'speakers': list(self.acoustic_model.speaker_mapping.keys())
        }
        
        metadata_path = os.path.join(base_path, 'system_metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Complete ASR system saved to: {base_path}")
    
    def load_system(self, base_path: str):
        acoustic_path = os.path.join(base_path, 'acoustic_model.pkl')
        self.acoustic_model.load_model(acoustic_path)
        
        language_path = os.path.join(base_path, 'language_model.pkl')
        self.language_model.load_model(language_path)
        
        self.auto_correction = AutoCorrection(self.language_model)
        
        metadata_path = os.path.join(base_path, 'system_metadata.json')
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        self.is_trained = metadata['is_trained']
        
        logger.info(f"Complete ASR system loaded from: {base_path}")
