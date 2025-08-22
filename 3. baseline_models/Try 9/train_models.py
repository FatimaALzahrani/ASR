import os
import sys
from datetime import datetime
from acoustic_model import AdvancedAcousticModel
from language_model import LanguageModel
from corrector_model import AutoCorrector
from ensemble_model import EnsembleModel
from file_utils import ensure_directory_exists, save_json
from settings import FEATURES_PATH, RESULTS_PATH


class ModelTrainer:
    def __init__(self, features_path=None, results_path=None):
        self.features_path = features_path or FEATURES_PATH
        self.results_path = results_path or RESULTS_PATH
        
        ensure_directory_exists(self.results_path)
        ensure_directory_exists(f"{self.results_path}/acoustic_models")
        ensure_directory_exists(f"{self.results_path}/language_models")
        ensure_directory_exists(f"{self.results_path}/correction_models")
        ensure_directory_exists(f"{self.results_path}/ensemble_models")
        ensure_directory_exists(f"{self.results_path}/reports")
        
        self.training_results = {}
        
        print("Model Trainer initialized")
        print(f"Features path: {self.features_path}")
        print(f"Results path: {self.results_path}")
    
    def train_acoustic_model(self):
        print("\n" + "="*60)
        print("TRAINING ACOUSTIC MODEL")
        print("="*60)
        
        try:
            acoustic_model = AdvancedAcousticModel(
                features_path=self.features_path,
                results_path=self.results_path
            )
            
            success = acoustic_model.train()
            
            if success:
                model_file = acoustic_model.save_model()
                report_file = acoustic_model.save_results()
                
                self.training_results['acoustic_model'] = {
                    'success': True,
                    'model_file': model_file,
                    'report_file': report_file,
                    'evaluation': acoustic_model.evaluate()
                }
                
                print("Acoustic model training completed successfully")
                return True
            else:
                self.training_results['acoustic_model'] = {
                    'success': False,
                    'error': 'Training failed'
                }
                print("Acoustic model training failed")
                return False
                
        except Exception as e:
            error_msg = f"Acoustic model training error: {str(e)}"
            print(error_msg)
            self.training_results['acoustic_model'] = {
                'success': False,
                'error': error_msg
            }
            return False
    
    def train_language_model(self):
        print("\n" + "="*60)
        print("TRAINING LANGUAGE MODEL")
        print("="*60)
        
        try:
            language_model = LanguageModel(
                features_path=self.features_path,
                results_path=self.results_path
            )
            
            success = language_model.train()
            
            if success:
                model_file = language_model.save_model()
                report_file = language_model.save_results()
                
                self.training_results['language_model'] = {
                    'success': True,
                    'model_file': model_file,
                    'report_file': report_file,
                    'evaluation': language_model.evaluate()
                }
                
                print("Language model training completed successfully")
                return True
            else:
                self.training_results['language_model'] = {
                    'success': False,
                    'error': 'Training failed'
                }
                print("Language model training failed")
                return False
                
        except Exception as e:
            error_msg = f"Language model training error: {str(e)}"
            print(error_msg)
            self.training_results['language_model'] = {
                'success': False,
                'error': error_msg
            }
            return False
    
    def train_corrector_model(self):
        print("\n" + "="*60)
        print("TRAINING CORRECTOR MODEL")
        print("="*60)
        
        try:
            corrector_model = AutoCorrector(
                features_path=self.features_path,
                results_path=self.results_path
            )
            
            success = corrector_model.train()
            
            if success:
                model_file = corrector_model.save_model()
                report_file = corrector_model.save_results()
                
                self.training_results['corrector_model'] = {
                    'success': True,
                    'model_file': model_file,
                    'report_file': report_file,
                    'evaluation': corrector_model.evaluate()
                }
                
                print("Corrector model training completed successfully")
                return True
            else:
                self.training_results['corrector_model'] = {
                    'success': False,
                    'error': 'Training failed'
                }
                print("Corrector model training failed")
                return False
                
        except Exception as e:
            error_msg = f"Corrector model training error: {str(e)}"
            print(error_msg)
            self.training_results['corrector_model'] = {
                'success': False,
                'error': error_msg
            }
            return False
    
    def train_ensemble_model(self):
        print("\n" + "="*60)
        print("TRAINING ENSEMBLE MODEL")
        print("="*60)
        
        try:
            ensemble_model = EnsembleModel(
                features_path=self.features_path,
                results_path=self.results_path
            )
            
            success = ensemble_model.train()
            
            if success:
                model_file = ensemble_model.save_model()
                report_file = ensemble_model.save_results()
                
                self.training_results['ensemble_model'] = {
                    'success': True,
                    'model_file': model_file,
                    'report_file': report_file,
                    'evaluation': ensemble_model.evaluate()
                }
                
                print("Ensemble model training completed successfully")
                return True
            else:
                self.training_results['ensemble_model'] = {
                    'success': False,
                    'error': 'Training failed'
                }
                print("Ensemble model training failed")
                return False
                
        except Exception as e:
            error_msg = f"Ensemble model training error: {str(e)}"
            print(error_msg)
            self.training_results['ensemble_model'] = {
                'success': False,
                'error': error_msg
            }
            return False
    
    def generate_training_summary(self):
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        
        total_models = len(self.training_results)
        successful_models = len([r for r in self.training_results.values() if r['success']])
        
        print(f"Total models trained: {total_models}")
        print(f"Successful: {successful_models}")
        print(f"Failed: {total_models - successful_models}")
        print(f"Success rate: {(successful_models/total_models)*100:.1f}%")
        
        print("\nModel-wise results:")
        for model_name, result in self.training_results.items():
            status = "SUCCESS" if result['success'] else "FAILED"
            print(f"  {model_name}: {status}")
            
            if result['success'] and 'evaluation' in result and result['evaluation']:
                eval_data = result['evaluation']
                if isinstance(eval_data, dict):
                    if 'accuracy' in eval_data:
                        print(f"    Accuracy: {eval_data['accuracy']:.4f}")
                    if 'f1_score' in eval_data:
                        print(f"    F1-Score: {eval_data['f1_score']:.4f}")
            
            if not result['success'] and 'error' in result:
                print(f"    Error: {result['error']}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = f"{self.results_path}/reports/training_summary_{timestamp}.json"
        
        summary_data = {
            'timestamp': datetime.now().isoformat(),
            'total_models': total_models,
            'successful_models': successful_models,
            'success_rate': (successful_models/total_models)*100,
            'results': self.training_results
        }
        
        save_json(summary_data, summary_file)
        print(f"\nTraining summary saved: {summary_file}")
        
        return summary_data
    
    def run_complete_training(self, models=None):
        if models is None:
            models = ['acoustic', 'language', 'corrector', 'ensemble']
        
        print("Speech Recognition System - Model Training")
        print("="*60)
        print("Designed for children with Down syndrome")
        print("="*60)
        
        start_time = datetime.now()
        
        try:
            if 'acoustic' in models:
                self.train_acoustic_model()
            
            if 'language' in models:
                self.train_language_model()
            
            if 'corrector' in models:
                self.train_corrector_model()
            
            if 'ensemble' in models:
                self.train_ensemble_model()
            
            end_time = datetime.now()
            training_duration = end_time - start_time
            
            print(f"\nTotal training time: {training_duration}")
            
            summary = self.generate_training_summary()
            
            if summary['successful_models'] > 0:
                print("\nTraining completed with some success!")
                print(f"Check results directory: {self.results_path}")
                return True
            else:
                print("\nAll model training failed!")
                return False
                
        except Exception as e:
            print(f"Critical training error: {str(e)}")
            return False
    
    def train_individual_model(self, model_type):
        model_type = model_type.lower()
        
        if model_type == 'acoustic':
            return self.train_acoustic_model()
        elif model_type == 'language':
            return self.train_language_model()
        elif model_type == 'corrector':
            return self.train_corrector_model()
        elif model_type == 'ensemble':
            return self.train_ensemble_model()
        else:
            print(f"Unknown model type: {model_type}")
            return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train speech recognition models')
    parser.add_argument('--models', nargs='+', 
                       choices=['acoustic', 'language', 'corrector', 'ensemble', 'all'],
                       default=['all'],
                       help='Models to train')
    parser.add_argument('--features-path', type=str, default=FEATURES_PATH,
                       help='Path to features directory')
    parser.add_argument('--results-path', type=str, default=RESULTS_PATH,
                       help='Path to results directory')
    
    args = parser.parse_args()
    
    if 'all' in args.models:
        models_to_train = ['acoustic', 'language', 'corrector', 'ensemble']
    else:
        models_to_train = args.models
    
    trainer = ModelTrainer(
        features_path=args.features_path,
        results_path=args.results_path
    )
    
    success = trainer.run_complete_training(models_to_train)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())