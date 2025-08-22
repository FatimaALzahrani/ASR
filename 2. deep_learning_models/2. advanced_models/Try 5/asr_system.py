import os
import numpy as np
import pickle
import json
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import (MinMaxScaler, Normalizer, PowerTransformer, 
                                 QuantileTransformer, StandardScaler, 
                                 LabelEncoder, RobustScaler)
from sklearn.model_selection import train_test_split
import tensorflow as tf

from data_loader import DataLoader
from data_balancer import DataBalancer
from model_trainer import ModelTrainer


class ASRSystem:
    def __init__(self, data_path="C:/Users/فاطمة الزهراني/Desktop/ابحاث/الداون/Data/clean", 
                 output_path="output_files"):
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)
        
        self.data_loader = DataLoader(data_path)
        self.data_balancer = DataBalancer()
        self.model_trainer = ModelTrainer()
        
        self.word_frequencies = {}
        self.scaler = None
        self.label_encoder = None
        
        self._setup_tensorflow()
        
    def _setup_tensorflow(self):
        tf.config.optimizer.set_jit(True)
        
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"GPU enabled: {len(gpus)} GPU(s) available")
            except RuntimeError as e:
                print(f"GPU setup error: {e}")
        else:
            print("Using CPU for training")
    
    def run_complete_evaluation(self):
        print("Starting ASR system evaluation with advanced deep learning...")
        print("="*80)
        
        df, word_counts, speaker_word_counts = self.data_loader.load_data()
        
        if df is None or len(df) == 0:
            print("ERROR: Failed to load data!")
            return None
        
        feature_cols = [col for col in df.columns if col not in ['file_path', 'word', 'speaker']]
        X = df[feature_cols].values
        y = df['word'].values
        speakers = df['speaker'].values
        
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        scalers = {
            'StandardScaler': StandardScaler(),
            'RobustScaler': RobustScaler(),
            'MinMaxScaler': MinMaxScaler(),
            'Normalizer': Normalizer(),
            'PowerTransformer': PowerTransformer(method='yeo-johnson'),
            'QuantileTransformer': QuantileTransformer(output_distribution='uniform')
        }
        
        best_scaler = None
        best_scaler_score = 0
        
        print("Testing different scaling methods...")
        for scaler_name, scaler in scalers.items():
            try:
                X_scaled = scaler.fit_transform(X)
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
                )
                
                quick_model = tf.keras.Sequential([
                    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
                    tf.keras.layers.Dropout(0.3),
                    tf.keras.layers.Dense(64, activation='relu'),
                    tf.keras.layers.Dense(len(np.unique(y_encoded)), activation='softmax')
                ])
                
                quick_model.compile(
                    optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )
                
                quick_model.fit(X_train, y_train, epochs=10, verbose=0, validation_split=0.2)
                _, score = quick_model.evaluate(X_test, y_test, verbose=0)
                
                print(f"  {scaler_name}: {score:.4f}")
                
                if score > best_scaler_score:
                    best_scaler_score = score
                    best_scaler = scaler
                    self.scaler = scaler
                    
            except Exception as e:
                print(f"  {scaler_name}: Error")
                continue
        
        X_scaled = best_scaler.fit_transform(X)
        
        print(f"\nDataset Information:")
        print(f"  Samples: {len(df):,}")
        print(f"  Features: {len(feature_cols):,}")
        print(f"  Words: {len(np.unique(y)):,}")
        print(f"  Speakers: {len(np.unique(speakers)):,}")
        print(f"  Best scaler: {type(best_scaler).__name__}")
        print(f"  Best scaler score: {best_scaler_score:.4f}")
        
        print(f"\nApplying intelligent data balancing...")
        X_balanced, y_balanced, speakers_balanced = self.data_balancer.balance_data(
            X_scaled, y_encoded, speakers
        )
        
        print(f"After balancing: {len(X_balanced):,} samples")
        
        print(f"\nTraining advanced deep learning models...")
        basic_results = self.model_trainer.train_deep_models(X_balanced, y_balanced, speakers_balanced)
        
        advanced_results = self.model_trainer.train_advanced_models(X_balanced, y_balanced)
        
        all_results = {**basic_results, **advanced_results}
        
        best_model_name = max(all_results, key=all_results.get)
        best_accuracy = all_results[best_model_name]
        
        accuracy_stats = {
            'mean': np.mean(list(all_results.values())),
            'std': np.std(list(all_results.values())),
            'median': np.median(list(all_results.values())),
            'max': np.max(list(all_results.values())),
            'min': np.min(list(all_results.values()))
        }
        
        final_results = {
            'dataset_info': {
                'original_samples': len(df),
                'balanced_samples': len(X_balanced),
                'features': len(feature_cols),
                'words': len(np.unique(y)),
                'speakers': len(np.unique(speakers)),
                'word_frequencies': dict(word_counts),
                'speaker_distribution': {
                    speaker: {
                        'total_samples': sum(words.values()),
                        'unique_words': len(words),
                        'words': dict(words)
                    } for speaker, words in speaker_word_counts.items()
                },
                'scaler_used': type(best_scaler).__name__,
                'scaler_score': best_scaler_score
            },
            'model_results': all_results,
            'accuracy_statistics': accuracy_stats,
            'absolute_best': {
                'model': best_model_name,
                'accuracy': best_accuracy,
                'improvement_over_baseline': (best_accuracy - 0.5) * 100
            },
            'model_rankings': sorted(all_results.items(), key=lambda x: x[1], reverse=True),
            'deep_learning_metrics': {
                'models_trained': len(all_results),
                'successful_models': len([r for r in all_results.values() if r > 0]),
                'average_accuracy': accuracy_stats['mean'],
                'accuracy_variance': accuracy_stats['std']
            },
            'evaluation_timestamp': datetime.now().isoformat(),
            'system_info': {
                'tensorflow_version': tf.__version__,
                'gpu_available': len(tf.config.experimental.list_physical_devices('GPU')) > 0,
                'gpu_count': len(tf.config.experimental.list_physical_devices('GPU'))
            }
        }
        
        results_file = self.output_path / 'deep_learning_asr_results.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        
        models_to_save = {
            'models': self.model_trainer.models,
            'model_history': self.model_trainer.history,
            'scaler': best_scaler,
            'label_encoder': self.label_encoder,
            'feature_columns': feature_cols,
            'model_results': all_results
        }
        
        models_file = self.output_path / 'deep_learning_models.pkl'
        with open(models_file, 'wb') as f:
            pickle.dump(models_to_save, f)
        
        for model_name, model in self.model_trainer.models.items():
            try:
                model_path = self.output_path / f'{model_name}_model.keras'
                model.save(model_path)
                print(f"Saved {model_name} to {model_path}")
            except Exception as e:
                print(f"Could not save {model_name}: {e}")
        
        self._print_detailed_results(final_results)
        
        return final_results
    
    def _print_detailed_results(self, results):
        print(f"\n" + "="*80)
        print(f"ASR System Evaluation Results with Advanced Deep Learning")
        print(f"="*80)
        
        dataset_info = results['dataset_info']
        print(f"\nDataset Information:")
        print(f"   Samples: {dataset_info['original_samples']:,} -> {dataset_info['balanced_samples']:,}")
        print(f"   Features: {dataset_info['features']:,}")
        print(f"   Words: {dataset_info['words']:,}")
        print(f"   Speakers: {dataset_info['speakers']:,}")
        print(f"   Scaler: {dataset_info['scaler_used']}")
        
        stats = results['accuracy_statistics']
        print(f"\nAccuracy Statistics:")
        print(f"   Maximum: {stats['max']*100:.2f}%")
        print(f"   Mean: {stats['mean']*100:.2f}%")
        print(f"   Median: {stats['median']*100:.2f}%")
        print(f"   Minimum: {stats['min']*100:.2f}%")
        print(f"   Standard Deviation: {stats['std']*100:.2f}%")
        
        best = results['absolute_best']
        print(f"\nBest Result:")
        print(f"   Model: {best['model']}")
        print(f"   Accuracy: {best['accuracy']*100:.2f}%")
        print(f"   Improvement: +{best['improvement_over_baseline']:.2f}%")
        
        print(f"\nModel Rankings (by accuracy):")
        for i, (name, acc) in enumerate(results['model_rankings'][:5], 1):
            emoji = "1st" if i == 1 else "2nd" if i == 2 else "3rd" if i == 3 else f"{i}th"
            status = "Excellent" if acc > 0.9 else "Very Good" if acc > 0.8 else "Good" if acc > 0.7 else "Fair"
            print(f"   {emoji:3s} {name:<30}: {acc*100:6.2f}% ({status})")
        
        system_info = results['system_info']
        print(f"\nSystem Information:")
        print(f"   TensorFlow: {system_info['tensorflow_version']}")
        print(f"   GPU Available: {'Yes' if system_info['gpu_available'] else 'No'}")
        if system_info['gpu_count'] > 0:
            print(f"   GPU Count: {system_info['gpu_count']}")
        
        dl_metrics = results['deep_learning_metrics']
        print(f"\nTraining Summary:")
        print(f"   Models Trained: {dl_metrics['models_trained']}")
        print(f"   Successful Models: {dl_metrics['successful_models']}")
        print(f"   Average Accuracy: {dl_metrics['average_accuracy']*100:.2f}%")
        
        print(f"\nSaved Files:")
        print(f"   Results: deep_learning_asr_results.json")
        print(f"   Models: deep_learning_models.pkl")
        print(f"   Individual Models: *_model.keras")
        
        print(f"="*80)
        print(f"Evaluation completed successfully! Best accuracy: {best['accuracy']*100:.2f}%")
        print(f"="*80)
    
    def load_trained_models(self, models_file=None):
        if models_file is None:
            models_file = self.output_path / 'deep_learning_models.pkl'
        
        try:
            with open(models_file, 'rb') as f:
                saved_data = pickle.load(f)
            
            self.model_trainer.models = saved_data['models']
            self.scaler = saved_data['scaler']
            self.label_encoder = saved_data['label_encoder']
            
            print(f"Successfully loaded {len(self.model_trainer.models)} models")
            return True
            
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    
    def predict_word(self, audio_file_path, model_name=None):
        if not self.model_trainer.models:
            print("No models loaded! Use load_trained_models() first")
            return None
        
        try:
            from feature_extractor import FeatureExtractor
            extractor = FeatureExtractor()
            features = extractor.extract_features(audio_file_path)
            
            if features is None:
                print(f"Failed to extract features from {audio_file_path}")
                return None
            
            feature_array = np.array(list(features.values())).reshape(1, -1)
            feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=0.0, neginf=0.0)
            feature_scaled = self.scaler.transform(feature_array)
            
            if model_name is None:
                model_name = list(self.model_trainer.models.keys())[0]
            
            model = self.model_trainer.models.get(model_name)
            if model is None:
                print(f"Model {model_name} not found")
                return None
            
            prediction = model.predict(feature_scaled, verbose=0)
            predicted_class = np.argmax(prediction, axis=1)[0]
            confidence = np.max(prediction)
            
            predicted_word = self.label_encoder.inverse_transform([predicted_class])[0]
            
            result = {
                'predicted_word': predicted_word,
                'confidence': float(confidence),
                'model_used': model_name,
                'probabilities': {
                    self.label_encoder.inverse_transform([i])[0]: float(prob)
                    for i, prob in enumerate(prediction[0])
                }
            }
            
            print(f"Prediction: {predicted_word} (confidence: {confidence*100:.1f}%)")
            return result
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return None