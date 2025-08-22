import numpy as np
from data_loader import DataLoader
from feature_extractor import FeatureExtractor
from model_factory import ModelFactory
from model_trainer import ModelTrainer
from hyperparameter_tuner import HyperparameterTuner
from results_analyzer import ResultsAnalyzer
from results_saver import ResultsSaver


class ImprovedModelsTrainer:
    def __init__(self, file_path='C:/Users/فاطمة الزهراني/Desktop/ابحاث/الداون/Codes/Real Codes/01_data_processing/processed_dataset.csv'):
        self.file_path = file_path
        self.data_loader = DataLoader(file_path)
        self.feature_extractor = FeatureExtractor()
        self.model_factory = ModelFactory()
        self.results_analyzer = ResultsAnalyzer()
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.train_speakers = None
        self.test_speakers = None
        
    def run_complete_pipeline(self):
        print("Starting improved models development with strong regularization")
        print("=" * 70)
        
        if not self.data_loader.load_data():
            return
        
        train_data, test_data = self.data_loader.split_by_speakers()
        
        self.X_train, self.X_test, self.y_train, self.y_test, common_words = (
            self.feature_extractor.preprocess_data(train_data, test_data)
        )
        
        self.train_speakers = self.data_loader.train_speakers
        self.test_speakers = self.data_loader.test_speakers
        
        print(f"Data loaded successfully:")
        print(f"- Training: {len(self.X_train)} samples")
        print(f"- Testing: {len(self.X_test)} samples")
        print(f"- Features: {self.X_train.shape[1]}")
        print(f"- Words: {len(common_words)}")
        
        model_trainer = ModelTrainer(self.X_train, self.X_test, self.y_train, self.y_test)
        
        base_models = self.model_factory.create_regularized_models()
        base_results = model_trainer.train_and_evaluate_models(base_models)
        
        ensemble_models = self.model_factory.create_ensemble_models(base_models)
        ensemble_results = model_trainer.train_and_evaluate_models(ensemble_models)
        
        hyperparameter_tuner = HyperparameterTuner(self.X_train, self.y_train)
        tuned_models = hyperparameter_tuner.tune_hyperparameters()
        tuned_results = model_trainer.train_and_evaluate_models(tuned_models)
        
        all_results = {**base_results, **ensemble_results, **tuned_results}
        
        self.results_analyzer.create_comprehensive_plots(all_results)
        
        results_saver = ResultsSaver(self.train_speakers, self.test_speakers, 
                                   self.X_train, self.X_test)
        comprehensive_results = results_saver.save_comprehensive_results(all_results)
        
        print("\n" + "=" * 70)
        print("Improved models development completed")
        
        print("\nBest Results:")
        print(f"- Best model: {comprehensive_results['summary']['best_model']}")
        print(f"- Best test accuracy: {comprehensive_results['summary']['best_test_accuracy']:.4f}")
        print(f"- Best CV score: {comprehensive_results['summary']['best_cv_score']:.4f}")
        
        print(f"\nModels with low overfitting:")
        for model in comprehensive_results['summary']['models_with_low_overfitting']:
            print(f"  - {model}")
        
        print("\nComprehensive summary:")
        valid_results = [r for r in all_results.values() if r is not None]
        if valid_results:
            avg_test_acc = np.mean([r['test_accuracy'] for r in valid_results])
            avg_cv_score = np.mean([r['cv_mean'] for r in valid_results])
            avg_overfitting = np.mean([r['overfitting_gap'] for r in valid_results])
            
            print(f"- Average test accuracy: {avg_test_acc:.4f}")
            print(f"- Average CV score: {avg_cv_score:.4f}")
            print(f"- Average overfitting gap: {avg_overfitting:.4f}")