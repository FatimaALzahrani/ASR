import json
import numpy as np
from collections import Counter

from dataset_loader import DatasetLoader
from model_trainer import ModelTrainer
from results_comparator import ResultsComparator

class EnhancedTrainingPipeline:
    def __init__(self):
        self.model_trainer = ModelTrainer()
        self.results_comparator = ResultsComparator()
    
    def run_training_comparison(self, data_dir="output_files/enhanced_audio_data"):
        print("Comparing model performance before and after data enhancement")
        print("="*60)
        
        enhanced_results = {}
        
        try:
            print(f"\n{'='*20} Enhanced Basic Model {'='*20}")
            loader = DatasetLoader(max_words=30)
            X, y, speakers, files_info = loader.load_dataset(data_dir)
            accuracy, results = self.model_trainer.train_model(X, y, "enhanced_basic_model")
            enhanced_results['Enhanced Basic Model'] = accuracy * 100
            
            print(f"\n{'='*20} Enhanced Without Aseel {'='*20}")
            loader = DatasetLoader(exclude_speakers=['أسيل'], max_words=30)
            X, y, speakers, files_info = loader.load_dataset(data_dir)
            accuracy, results = self.model_trainer.train_model(X, y, "enhanced_without_aseel")
            enhanced_results['Enhanced Without Aseel'] = accuracy * 100
            
            print(f"\n{'='*20} Enhanced Without Ahmed {'='*20}")
            loader = DatasetLoader(exclude_speakers=['أحمد'], max_words=30)
            X, y, speakers, files_info = loader.load_dataset(data_dir)
            accuracy, results = self.model_trainer.train_model(X, y, "enhanced_without_ahmed")
            enhanced_results['Enhanced Without Ahmed'] = accuracy * 100
            
            for speaker in ['عاصم', 'أسيل']:
                print(f"\n{'='*15} Enhanced {speaker} Specialized Model {'='*15}")
                try:
                    loader = DatasetLoader(max_words=20)
                    X, y, speakers_array, files_info = loader.load_dataset(data_dir)
                    
                    speaker_mask = speakers_array == speaker
                    if np.sum(speaker_mask) > 30:
                        X_speaker = X[speaker_mask]
                        y_speaker = y[speaker_mask]
                        
                        word_counts = Counter(y_speaker)
                        if len(word_counts) >= 5:
                            accuracy, results = self.model_trainer.train_model(X_speaker, y_speaker, f"enhanced_{speaker}_specialized")
                            enhanced_results[f'Enhanced {speaker} Specialized Model'] = accuracy * 100
                        else:
                            print(f"Insufficient words for speaker {speaker}: {len(word_counts)} words")
                            enhanced_results[f'Enhanced {speaker} Specialized Model'] = 0
                    else:
                        print(f"Insufficient data for speaker {speaker}: {np.sum(speaker_mask)} samples")
                        enhanced_results[f'Enhanced {speaker} Specialized Model'] = 0
                except Exception as e:
                    print(f"Error for {speaker} model: {e}")
                    enhanced_results[f'Enhanced {speaker} Specialized Model'] = 0
            
            comparison_results = self.results_comparator.compare_results(enhanced_results)
            
            final_results = {
                'previous_results': self.results_comparator.previous_results,
                'enhanced_results': enhanced_results,
                'comparison_results': comparison_results,
                'summary': {
                    'best_previous': max(self.results_comparator.previous_results.values()),
                    'best_enhanced': max(enhanced_results.values()) if enhanced_results else 0,
                    'average_improvement': np.mean([data['improvement'] for data in comparison_results.values()]) if comparison_results else 0,
                    'models_improved': sum(1 for data in comparison_results.values() if data['improvement'] > 0),
                    'total_models': len(comparison_results)
                }
            }
            
            with open('enhanced_models_final_comparison.json', 'w', encoding='utf-8') as f:
                json.dump(final_results, f, ensure_ascii=False, indent=2)
            
            print(f"Results saved to: enhanced_models_final_comparison.json")
            
            return final_results
            
        except Exception as e:
            print(f"Training error: {e}")
            import traceback
            traceback.print_exc()
            return None