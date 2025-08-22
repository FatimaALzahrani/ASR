import os
import pickle
from datetime import datetime
import pandas as pd
from feature_extractor import AdvancedFeatureExtractor
from audio_utils import load_audio, validate_audio_file
from file_utils import get_latest_file, save_json
from settings import RESULTS_PATH, SPEAKER_INFO


class IntegratedSpeechSystem:
    def __init__(self, models_path=None):
        self.models_path = models_path or RESULTS_PATH
        self.acoustic_model = None
        self.language_model = None
        self.corrector = None
        self.feature_extractor = None
        self.scaler = None
        self.label_encoder = None
        self.selected_features = None
        self.confidence_threshold = 0.5
        
        self.speakers_info = SPEAKER_INFO
        
        print("Integrated Speech Recognition System initialized")
        print(f"Models path: {self.models_path}")
    
    def load_models(self):
        print("Loading trained models...")
        
        try:
            acoustic_files = get_latest_file(f"{self.models_path}/acoustic_models", "acoustic_model_*.pkl")
            
            if acoustic_files:
                with open(acoustic_files, 'rb') as f:
                    acoustic_data = pickle.load(f)
                
                self.acoustic_model = acoustic_data['acoustic_model']
                self.scaler = acoustic_data['scaler']
                self.label_encoder = acoustic_data['label_encoder']
                self.selected_features = acoustic_data['selected_features']
                self.confidence_threshold = acoustic_data['confidence_threshold']
                
                print(f"Loaded acoustic model: {os.path.basename(acoustic_files)}")
            else:
                print("Acoustic model not found")
                return False
            
            language_files = get_latest_file(f"{self.models_path}/language_models", "language_model_*.pkl")
            
            if language_files:
                with open(language_files, 'rb') as f:
                    self.language_model = pickle.load(f)
                
                print(f"Loaded language model: {os.path.basename(language_files)}")
            else:
                print("Language model not found")
                return False
            
            corrector_files = get_latest_file(f"{self.models_path}/correction_models", "corrector_*.pkl")
            
            if corrector_files:
                with open(corrector_files, 'rb') as f:
                    self.corrector = pickle.load(f)
                
                print(f"Loaded corrector: {os.path.basename(corrector_files)}")
            else:
                print("Corrector not found")
                return False
            
            self.feature_extractor = AdvancedFeatureExtractor()
            print("Feature extractor initialized")
            
            print("All models loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            return False
    
    def extract_audio_features(self, audio_file_path):
        try:
            audio, sr = load_audio(audio_file_path)
            if audio is None:
                return None
            
            features = self.feature_extractor.extract_features(audio, sr)
            return features
            
        except Exception as e:
            print(f"Feature extraction error: {str(e)}")
            return None
    
    def predict_acoustic_model(self, features):
        try:
            feature_df = pd.DataFrame([features])
            
            if self.selected_features:
                available_features = [f for f in self.selected_features if f in feature_df.columns]
                if len(available_features) < len(self.selected_features) * 0.8:
                    print(f"Missing features: {len(available_features)}/{len(self.selected_features)}")
                
                feature_df = feature_df[available_features]
            
            feature_df = feature_df.fillna(feature_df.mean())
            features_scaled = self.scaler.transform(feature_df)
            
            predictions = self.acoustic_model.predict(features_scaled)
            probabilities = self.acoustic_model.predict_proba(features_scaled)
            
            predicted_words = self.label_encoder.inverse_transform(predictions)
            max_probs = max(probabilities[0])
            
            top_indices = sorted(range(len(probabilities[0])), key=lambda i: probabilities[0][i], reverse=True)[:5]
            alternatives = []
            
            for idx in top_indices:
                word = self.label_encoder.inverse_transform([idx])[0]
                confidence = probabilities[0][idx]
                alternatives.append({
                    'word': word,
                    'confidence': float(confidence)
                })
            
            return {
                'predicted_word': predicted_words[0],
                'confidence': float(max_probs),
                'alternatives': alternatives,
                'high_confidence': max_probs >= self.confidence_threshold
            }
            
        except Exception as e:
            print(f"Acoustic prediction error: {str(e)}")
            return None
    
    def apply_language_model_correction(self, acoustic_result, context=None):
        try:
            predicted_word = acoustic_result['predicted_word']
            confidence = acoustic_result['confidence']
            
            if confidence >= self.confidence_threshold and predicted_word in self.language_model['vocabulary']:
                return {
                    'original_prediction': predicted_word,
                    'corrected_word': predicted_word,
                    'correction_applied': False,
                    'correction_confidence': confidence,
                    'method': 'high_confidence_no_correction'
                }
            
            correction_result = self.apply_auto_correction(predicted_word, context)
            return correction_result
            
        except Exception as e:
            print(f"Language model correction error: {str(e)}")
            return {
                'original_prediction': acoustic_result['predicted_word'],
                'corrected_word': acoustic_result['predicted_word'],
                'correction_applied': False,
                'correction_confidence': acoustic_result['confidence'],
                'method': 'error_fallback'
            }
    
    def apply_auto_correction(self, word, context=None):
        try:
            vocabulary = set(self.language_model['vocabulary'])
            
            if word in vocabulary:
                return {
                    'original_prediction': word,
                    'corrected_word': word,
                    'correction_applied': False,
                    'correction_confidence': 1.0,
                    'method': 'exact_match'
                }
            
            from difflib import get_close_matches, SequenceMatcher
            
            close_matches = get_close_matches(word, vocabulary, n=5, cutoff=0.6)
            
            if close_matches:
                best_match = close_matches[0]
                similarity = SequenceMatcher(None, word, best_match).ratio()
                
                return {
                    'original_prediction': word,
                    'corrected_word': best_match,
                    'correction_applied': True,
                    'correction_confidence': similarity,
                    'method': 'similarity_correction',
                    'alternatives': close_matches[1:4]
                }
            else:
                return {
                    'original_prediction': word,
                    'corrected_word': word,
                    'correction_applied': False,
                    'correction_confidence': 0.0,
                    'method': 'no_correction_found'
                }
                
        except Exception as e:
            print(f"Auto correction error: {str(e)}")
            return {
                'original_prediction': word,
                'corrected_word': word,
                'correction_applied': False,
                'correction_confidence': 0.0,
                'method': 'error_fallback'
            }
    
    def recognize_speech(self, audio_file_path, speaker=None, context=None):
        print(f"Starting speech recognition...")
        print(f"File: {audio_file_path}")
        if speaker:
            print(f"Speaker: {speaker}")
        
        try:
            is_valid, validation_msg = validate_audio_file(audio_file_path)
            if not is_valid:
                return {
                    'success': False,
                    'error': validation_msg,
                    'file_path': audio_file_path
                }
            
            print("Extracting audio features...")
            features = self.extract_audio_features(audio_file_path)
            
            if features is None:
                return {
                    'success': False,
                    'error': 'Feature extraction failed'
                }
            
            print("Applying acoustic model...")
            acoustic_result = self.predict_acoustic_model(features)
            
            if acoustic_result is None:
                return {
                    'success': False,
                    'error': 'Acoustic prediction failed'
                }
            
            print("Applying language model and correction...")
            correction_result = self.apply_language_model_correction(acoustic_result, context)
            
            final_result = {
                'success': True,
                'timestamp': datetime.now().isoformat(),
                'file_path': audio_file_path,
                'speaker': speaker,
                'context': context,
                
                'acoustic_prediction': acoustic_result['predicted_word'],
                'acoustic_confidence': acoustic_result['confidence'],
                'acoustic_alternatives': acoustic_result['alternatives'],
                'high_confidence': acoustic_result['high_confidence'],
                
                'final_word': correction_result['corrected_word'],
                'correction_applied': correction_result['correction_applied'],
                'correction_confidence': correction_result['correction_confidence'],
                'correction_method': correction_result['method'],
                
                'speaker_info': self.speakers_info.get(speaker, {}) if speaker else {},
                'processing_steps': [
                    'feature_extraction',
                    'acoustic_prediction',
                    'language_model_correction'
                ]
            }
            
            if 'alternatives' in correction_result:
                final_result['correction_alternatives'] = correction_result['alternatives']
            
            print(f"Recognition completed: '{final_result['final_word']}'")
            print(f"Confidence: {final_result['correction_confidence']:.3f}")
            
            return final_result
            
        except Exception as e:
            print(f"Speech recognition error: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def recognize_speech_sequence(self, audio_files, speaker=None):
        print(f"Recognizing sequence of {len(audio_files)} audio files...")
        
        results = []
        context = []
        
        for i, audio_file in enumerate(audio_files):
            print(f"\n--- File {i+1}/{len(audio_files)} ---")
            
            result = self.recognize_speech(audio_file, speaker, context[-3:])
            results.append(result)
            
            if result['success']:
                context.append(result['final_word'])
        
        successful_recognitions = [r for r in results if r['success']]
        success_rate = len(successful_recognitions) / len(results) if results else 0
        
        recognized_text = ' '.join([r['final_word'] for r in successful_recognitions])
        
        sequence_result = {
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'speaker': speaker,
            'total_files': len(audio_files),
            'successful_recognitions': len(successful_recognitions),
            'success_rate': success_rate,
            'recognized_text': recognized_text,
            'individual_results': results,
            'speaker_info': self.speakers_info.get(speaker, {}) if speaker else {}
        }
        
        print(f"\nSequence recognition completed!")
        print(f"Success rate: {success_rate:.2%}")
        print(f"Recognized text: '{recognized_text}'")
        
        return sequence_result
    
    def get_speaker_statistics(self, speaker):
        if speaker not in self.speakers_info:
            return None
        
        stats = {
            'speaker_name': speaker,
            'basic_info': self.speakers_info[speaker],
            'vocabulary_size': len(self.language_model.get('speaker_vocabularies', {}).get(speaker, {})),
            'total_recordings': sum(self.language_model.get('speaker_vocabularies', {}).get(speaker, {}).values()),
            'most_common_words': list(self.language_model.get('speaker_vocabularies', {}).get(speaker, {}).keys())[:10]
        }
        
        return stats
    
    def test_system_with_sample(self):
        print("Testing system with sample files...")
        
        test_files = []
        
        if os.path.exists("processed_data"):
            for root, dirs, files in os.walk("processed_data"):
                for file in files[:5]:
                    if file.endswith('.wav'):
                        test_files.append(os.path.join(root, file))
        
        if not test_files:
            print("No test files found")
            return False
        
        print(f"Found {len(test_files)} test files")
        
        test_results = []
        
        for test_file in test_files[:3]:
            result = self.recognize_speech(test_file)
            test_results.append(result)
            
            if result['success']:
                print(f"✓ {os.path.basename(test_file)}: '{result['final_word']}'")
            else:
                print(f"✗ {os.path.basename(test_file)}: {result.get('error', 'Unknown error')}")
        
        test_summary = {
            'timestamp': datetime.now().isoformat(),
            'total_tests': len(test_results),
            'successful_tests': len([r for r in test_results if r['success']]),
            'success_rate': len([r for r in test_results if r['success']]) / len(test_results),
            'test_results': test_results
        }
        
        os.makedirs("system_tests", exist_ok=True)
        save_json(test_summary, "system_tests/integration_test_results.json")
        
        print(f"\nTest results:")
        print(f"Successful: {test_summary['successful_tests']}/{test_summary['total_tests']}")
        print(f"Success rate: {test_summary['success_rate']:.2%}")
        
        return test_summary['success_rate'] > 0.5


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Integrated Speech Recognition System')
    parser.add_argument('--audio-file', type=str, help='Audio file to recognize')
    parser.add_argument('--speaker', type=str, help='Speaker name')
    parser.add_argument('--models-path', type=str, default=RESULTS_PATH, help='Path to trained models')
    parser.add_argument('--test', action='store_true', help='Run system test')
    
    args = parser.parse_args()
    
    system = IntegratedSpeechSystem(models_path=args.models_path)
    
    if not system.load_models():
        print("Failed to load models")
        return 1
    
    if args.test:
        success = system.test_system_with_sample()
        return 0 if success else 1
    
    if args.audio_file:
        if not os.path.exists(args.audio_file):
            print(f"Audio file not found: {args.audio_file}")
            return 1
        
        result = system.recognize_speech(args.audio_file, args.speaker)
        
        if result['success']:
            print(f"\nRecognition Result:")
            print(f"Word: {result['final_word']}")
            print(f"Confidence: {result['correction_confidence']:.3f}")
            print(f"Method: {result['correction_method']}")
            
            if result['correction_applied']:
                print(f"Original: {result['acoustic_prediction']}")
                print(f"Corrected: {result['final_word']}")
        else:
            print(f"Recognition failed: {result['error']}")
            return 1
    else:
        print("System ready for speech recognition")
        print("Use --audio-file to recognize a file or --test to run tests")
    
    return 0


if __name__ == "__main__":
    exit(main())