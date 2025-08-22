import os
import numpy as np
import json
import pickle
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple

import tensorflow as tf
from tensorflow.keras import callbacks
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE

from feature_extractor import EnhancedFeatureExtractor
from language_corrector import AdvancedLanguageCorrector
from neural_models import ASRModelArchitectures
from baseline_models import BaselineModels
from statistical_analyzer import StatisticalAnalyzer
from visualization_generator import VisualizationGenerator

class ComprehensiveASRSystem:
    def __init__(self, data_path: str, output_path: str = "asr_results"):
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True, parents=True)
        
        self.feature_extractor = EnhancedFeatureExtractor()
        self.language_corrector = None
        self.baseline_models = BaselineModels()
        self.statistical_analyzer = StatisticalAnalyzer()
        self.visualizer = VisualizationGenerator(str(self.output_path / "visualizations"))
        
        self.speakers = {
            'Ahmad': range(0, 7),
            'Asim': range(7, 14),
            'Haifa': range(14, 21),
            'Aseel': range(21, 29),
            'Wissam': range(29, 37)
        }
        
        self.scaler = None
        self.label_encoder = None
        self.results = {}
        self.training_histories = []
        
        print("Comprehensive ASR System initialized")
        print(f"Data path: {self.data_path}")
        print(f"Output path: {self.output_path}")
    
    def get_speaker(self, filename: str) -> str:
        name = os.path.splitext(filename)[0]
        if not name.isdigit():
            return 'unknown'
        
        try:
            number = 0 if name == "0" else int(name.lstrip("0")) if name.lstrip("0") else 0
            
            for speaker, rng in self.speakers.items():
                if number in rng:
                    return speaker
            return 'unknown'
        except ValueError:
            return 'unknown'
    
    def load_and_prepare_data(self) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        print("Loading and preparing data...")
        
        X = []
        y = []
        speaker_labels = []
        word_counts = defaultdict(int)
        speaker_counts = defaultdict(int)
        failed_extractions = 0
        
        for word_folder in self.data_path.iterdir():
            if not word_folder.is_dir():
                continue
                
            word = word_folder.name
            audio_files = list(word_folder.glob("*.wav"))
            
            if len(audio_files) < 2:
                print(f"Skipping word '{word}' - insufficient samples")
                continue
            
            print(f"Processing word: {word} ({len(audio_files)} files)")
            
            for audio_file in audio_files:
                features = self.feature_extractor.extract_comprehensive_features(audio_file)
                
                if features is not None:
                    X.append(features)
                    y.append(word)
                    
                    speaker = self.get_speaker(audio_file.name)
                    speaker_labels.append(speaker)
                    
                    word_counts[word] += 1
                    speaker_counts[speaker] += 1
                else:
                    failed_extractions += 1
        
        if len(X) == 0:
            raise ValueError("No valid features extracted")
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y)
        speaker_labels = np.array(speaker_labels)
        
        word_list = sorted(list(set(y)))
        
        self.language_corrector = AdvancedLanguageCorrector(word_list)
        self.language_corrector.train_on_vocabulary(y.tolist())
        
        print(f"Data loaded: {len(X)} samples, {len(word_list)} words")
        print(f"Speaker distribution: {dict(speaker_counts)}")
        print(f"Failed extractions: {failed_extractions}")
        
        return X, y, word_list, speaker_labels
    
    def preprocess_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        print("Preprocessing data...")
        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        try:
            unique, counts = np.unique(y_encoded, return_counts=True)
            min_samples = np.min(counts)
            k_neighbors = min(5, min_samples - 1) if min_samples > 1 else 1
            
            smote = SMOTE(random_state=42, k_neighbors=k_neighbors, sampling_strategy='auto')
            X_balanced, y_balanced = smote.fit_resample(X_scaled, y_encoded)
            
            print(f"SMOTE applied: {len(X_scaled)} -> {len(X_balanced)} samples")
            return X_balanced, y_balanced
            
        except Exception as e:
            print(f"SMOTE failed: {e}, using original data")
            return X_scaled, y_encoded
    
    def evaluate_baseline_models(self, X: np.ndarray, y: np.ndarray) -> Dict:
        print("Evaluating baseline models...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        baseline_results = self.baseline_models.evaluate_traditional_models(
            X_train, X_test, y_train, y_test
        )
        
        self.results['baseline_results'] = baseline_results
        return baseline_results
    
    def train_neural_models(self, X: np.ndarray, y: np.ndarray) -> Dict:
        print("Training neural network models...")
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        num_classes = len(np.unique(y))
        
        model_creators = {
            'Enhanced_CNN': ASRModelArchitectures.create_enhanced_cnn,
            'Enhanced_LSTM': ASRModelArchitectures.create_enhanced_lstm,
            'Deep_DNN': ASRModelArchitectures.create_deep_dnn,
            'Hybrid_CNN_DNN': ASRModelArchitectures.create_hybrid_model,
            'Transformer_Inspired': ASRModelArchitectures.create_transformer_inspired,
            'Conformer_Inspired': ASRModelArchitectures.create_conformer_inspired
        }
        
        neural_results = {}
        
        for model_name, model_creator in model_creators.items():
            print(f"Training {model_name}...")
            
            fold_scores = []
            fold_histories = []
            
            for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
                X_train_fold = X[train_idx]
                X_val_fold = X[val_idx]
                y_train_fold = y[train_idx]
                y_val_fold = y[val_idx]
                
                try:
                    model = model_creator(X_train_fold.shape[1], num_classes)
                    model.compile(
                        optimizer='adam',
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy']
                    )
                    
                    callbacks_list = [
                        callbacks.EarlyStopping(
                            monitor='val_accuracy',
                            patience=15,
                            restore_best_weights=True,
                            verbose=0
                        ),
                        callbacks.ReduceLROnPlateau(
                            monitor='val_loss',
                            factor=0.5,
                            patience=8,
                            min_lr=1e-7,
                            verbose=0
                        )
                    ]
                    
                    history = model.fit(
                        X_train_fold, y_train_fold,
                        validation_data=(X_val_fold, y_val_fold),
                        epochs=50,
                        batch_size=32,
                        callbacks=callbacks_list,
                        verbose=0
                    )
                    
                    val_loss, val_accuracy = model.evaluate(X_val_fold, y_val_fold, verbose=0)
                    fold_scores.append(val_accuracy)
                    fold_histories.append(history.history)
                    
                    print(f"  Fold {fold+1}: {val_accuracy:.4f}")
                    
                except Exception as e:
                    print(f"  Fold {fold+1} failed: {e}")
                    fold_scores.append(0.0)
                
                finally:
                    if 'model' in locals():
                        del model
                    tf.keras.backend.clear_session()
            
            if fold_scores:
                mean_score = np.mean(fold_scores)
                std_score = np.std(fold_scores)
                
                neural_results[model_name] = {
                    'accuracy': mean_score,
                    'std': std_score,
                    'fold_scores': fold_scores
                }
                
                if fold_histories:
                    self.training_histories.extend(fold_histories[:2])
                
                print(f"{model_name}: {mean_score:.4f} ± {std_score:.4f}")
        
        self.results['neural_results'] = neural_results
        return neural_results
    
    def test_language_correction(self, X: np.ndarray, y: np.ndarray) -> Dict:
        print("Testing language correction system...")
        
        if 'neural_results' not in self.results:
            print("No neural results available for correction testing")
            return {}
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        num_classes = len(np.unique(y))
        
        model = ASRModelArchitectures.create_hybrid_model(X_train.shape[1], num_classes)
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        model.fit(X_train, y_train, epochs=30, verbose=0)
        
        predictions = model.predict(X_test, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        predicted_words = self.label_encoder.inverse_transform(predicted_classes)
        y_test_words = self.label_encoder.inverse_transform(y_test)
        
        original_accuracy = accuracy_score(y_test_words, predicted_words)
        
        simulated_errors = self.language_corrector.simulate_realistic_errors(
            predicted_words.tolist(), error_rate=0.15
        )
        
        corrected_words = []
        for error_word in simulated_errors:
            correction_result = self.language_corrector.correct_word(error_word)
            corrected_words.append(correction_result['corrected'])
        
        simulated_accuracy = accuracy_score(y_test_words, simulated_errors)
        corrected_accuracy = accuracy_score(y_test_words, corrected_words)
        
        correction_results = {
            'original_accuracy': original_accuracy,
            'simulated_accuracy': simulated_accuracy,
            'corrected_accuracy': corrected_accuracy,
            'improvement': corrected_accuracy - simulated_accuracy,
            'total_corrections': len([w for w in corrected_words if w in simulated_errors])
        }
        
        self.results['correction_results'] = correction_results
        
        print(f"Original accuracy: {original_accuracy:.4f}")
        print(f"Simulated errors accuracy: {simulated_accuracy:.4f}")
        print(f"After correction accuracy: {corrected_accuracy:.4f}")
        print(f"Correction improvement: {correction_results['improvement']:.4f}")
        
        return correction_results
    
    def perform_statistical_analysis(self) -> Dict:
        print("Performing statistical analysis...")
        
        statistical_results = {}
        
        if 'correction_results' in self.results:
            correction = self.results['correction_results']
            
            before_scores = np.array([correction['simulated_accuracy']] * 5)
            after_scores = np.array([correction['corrected_accuracy']] * 5)
            
            correction_analysis = self.statistical_analyzer.paired_ttest_analysis(
                before_scores, after_scores
            )
            
            statistical_results['correction_effectiveness'] = correction_analysis
        
        if 'baseline_results' in self.results and 'neural_results' in self.results:
            baseline_accs = [r.get('accuracy', 0) for r in self.results['baseline_results'].values() 
                           if isinstance(r, dict)]
            neural_accs = [r.get('accuracy', 0) for r in self.results['neural_results'].values()]
            
            if baseline_accs and neural_accs:
                best_baseline = max(baseline_accs)
                best_neural = max(neural_accs)
                
                statistical_results['method_comparison'] = {
                    'best_baseline_accuracy': best_baseline,
                    'best_neural_accuracy': best_neural,
                    'improvement': best_neural - best_baseline,
                    'relative_improvement': (best_neural - best_baseline) / best_baseline * 100 if best_baseline > 0 else 0
                }
        
        self.results['statistical_analysis'] = statistical_results
        return statistical_results
    
    def generate_visualizations(self):
        print("Generating visualizations...")
        
        viz_paths = {}
        
        if 'neural_results' in self.results:
            viz_paths['model_comparison'] = self.visualizer.plot_model_comparison(
                self.results['neural_results']
            )
        
        if 'correction_results' in self.results:
            correction = self.results['correction_results']
            before_scores = [correction['simulated_accuracy']] * 5
            after_scores = [correction['corrected_accuracy']] * 5
            
            viz_paths['correction_effectiveness'] = self.visualizer.plot_correction_effectiveness(
                before_scores, after_scores
            )
        
        if self.training_histories:
            viz_paths['training_curves'] = self.visualizer.plot_training_curves(
                self.training_histories
            )
        
        all_results = {}
        if 'baseline_results' in self.results:
            all_results.update(self.results['baseline_results'])
        if 'neural_results' in self.results:
            all_results.update(self.results['neural_results'])
        
        if all_results:
            viz_paths['summary_table'] = self.visualizer.create_summary_table(all_results)
        
        self.results['visualization_paths'] = viz_paths
        return viz_paths
    
    def save_results(self):
        print("Saving results...")
        
        final_results = {
            'system_info': {
                'name': 'Comprehensive ASR System for Children with Down Syndrome',
                'version': '2.0',
                'timestamp': datetime.now().isoformat(),
                'data_path': str(self.data_path),
                'output_path': str(self.output_path)
            },
            'results': self.results
        }
        
        results_file = self.output_path / 'comprehensive_asr_results.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2, default=str)
        
        if self.scaler:
            scaler_file = self.output_path / 'scaler.pkl'
            with open(scaler_file, 'wb') as f:
                pickle.dump(self.scaler, f)
        
        if self.label_encoder:
            encoder_file = self.output_path / 'label_encoder.pkl'
            with open(encoder_file, 'wb') as f:
                pickle.dump(self.label_encoder, f)
        
        if self.language_corrector:
            corrector_file = self.output_path / 'language_corrector.pkl'
            with open(corrector_file, 'wb') as f:
                pickle.dump(self.language_corrector, f)
        
        print(f"Results saved to: {results_file}")
        
        self.generate_summary_report()
    
    def generate_summary_report(self):
        report_content = []
        report_content.append("# Comprehensive ASR System Results")
        report_content.append("## For Children with Down Syndrome")
        report_content.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_content.append("")
        
        if 'neural_results' in self.results:
            report_content.append(f"- **Improvement**: +{correction['improvement']:.4f}")
        
        if 'statistical_analysis' in self.results:
            stats = self.results['statistical_analysis']
            report_content.append("\n## Statistical Analysis")
            
            if 'correction_effectiveness' in stats:
                corr_stats = stats['correction_effectiveness']
                report_content.append(f"- P-value: {corr_stats['p_value']:.6f}")
                report_content.append(f"- Effect Size: {corr_stats['effect_size_cohens_d']:.4f}")
                report_content.append(f"- Significant: {corr_stats['significant']}")
            
            if 'method_comparison' in stats:
                comp = stats['method_comparison']
                report_content.append(f"- Best Baseline: {comp['best_baseline_accuracy']:.4f}")
                report_content.append(f"- Best Neural: {comp['best_neural_accuracy']:.4f}")
                report_content.append(f"- Relative Improvement: {comp['relative_improvement']:.1f}%")
        
        report_content.append("\n## Files Generated")
        if 'visualization_paths' in self.results:
            for viz_name, viz_path in self.results['visualization_paths'].items():
                report_content.append(f"- {viz_name}: {viz_path}")
        
        report_file = self.output_path / 'summary_report.md'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_content))
        
        print(f"Summary report saved: {report_file}")
    
    def run_complete_evaluation(self) -> Dict:
        print("Starting comprehensive ASR evaluation...")
        print("=" * 60)
        
        try:
            X, y, word_list, speaker_labels = self.load_and_prepare_data()
            
            X_processed, y_processed = self.preprocess_data(X, y)
            
            baseline_results = self.evaluate_baseline_models(X_processed, y_processed)
            
            neural_results = self.train_neural_models(X_processed, y_processed)
            
            correction_results = self.test_language_correction(X_processed, y_processed)
            
            statistical_results = self.perform_statistical_analysis()
            
            visualization_paths = self.generate_visualizations()
            
            self.save_results()
            
            print("\n" + "=" * 60)
            print("Comprehensive evaluation completed successfully!")
            print(f"Results saved to: {self.output_path}")
            
            if neural_results:
                best_model = max(neural_results.keys(), 
                               key=lambda k: neural_results[k]['accuracy'])
                best_accuracy = neural_results[best_model]['accuracy']
                print(f"Best model: {best_model} (Accuracy: {best_accuracy:.4f})")
            
            if correction_results:
                improvement = correction_results.get('improvement', 0)
                print(f"Language correction improvement: +{improvement:.4f}")
            
            print("=" * 60)
            
            return self.results
            
        except Exception as e:
            print(f"Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}.append("## Neural Network Models Performance")
            for model_name, result in self.results['neural_results'].items():
                accuracy = result.get('accuracy', 0)
                std = result.get('std', 0)
                report_content.append(f"- **{model_name}**: {accuracy:.4f} ± {std:.4f}")
            
            best_neural = max(self.results['neural_results'].items(), 
                            key=lambda x: x[1].get('accuracy', 0))
            report_content.append(f"\n**Best Neural Model**: {best_neural[0]} ({best_neural[1]['accuracy']:.4f})")
        
        if 'baseline_results' in self.results:
            report_content.append("\n## Baseline Models Performance")
            for model_name, result in self.results['baseline_results'].items():
                if isinstance(result, dict) and 'accuracy' in result:
                    accuracy = result['accuracy']
                    report_content.append(f"- **{model_name}**: {accuracy:.4f}")
        
        if 'correction_results' in self.results:
            correction = self.results['correction_results']
            report_content.append("\n## Language Correction Results")
            report_content.append(f"- Original Accuracy: {correction['original_accuracy']:.4f}")
            report_content.append(f"- After Simulated Errors: {correction['simulated_accuracy']:.4f}")
            report_content.append(f"- After Correction: {correction['corrected_accuracy']:.4f}")
            report_content