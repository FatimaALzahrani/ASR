import os
import json
import pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from datetime import datetime
from data_loader import ComprehensiveDataLoader
from feature_extractor import RobustFeatureExtractor
from config import Config

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available")


class UltimateASRSystem:
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.data_loader = ComprehensiveDataLoader(data_path)
        self.feature_extractor = RobustFeatureExtractor()
        self.models = {}
        self.scaler = None
        self.label_encoder = None
        self.feature_names = None
        self.results = {}
        
    def load_and_process_data(self):
        print("Loading data...")
        
        data_result = self.data_loader.scan_audio_files()
        
        if data_result is None:
            return None
            
        self.df, self.word_counts = data_result
        
        min_samples_per_word = Config.MIN_SAMPLES_PER_WORD
        valid_words = [word for word, count in self.word_counts.items() if count >= min_samples_per_word]
        
        self.filtered_df = self.df[self.df['word'].isin(valid_words)]
        
        print(f"Filtered data: {len(self.filtered_df)} samples")
        print(f"Words: {len(valid_words)}")
        
        return self.filtered_df
    
    def extract_all_features(self):
        print("Extracting features...")
        
        all_features = []
        total_files = len(self.filtered_df)
        
        for idx, row in self.filtered_df.iterrows():
            if idx % 50 == 0:
                print(f"Processing {idx}/{total_files}...")
            
            features = self.feature_extractor.extract_comprehensive_features(
                row['file_path'], row['word'], row['speaker']
            )
            
            combined_row = {
                'file_path': row['file_path'],
                'word': row['word'],
                'speaker': row['speaker'],
                **features
            }
            
            all_features.append(combined_row)
        
        self.feature_df = pd.DataFrame(all_features)
        
        print(f"Features extracted:")
        print(f"Success: {self.feature_extractor.features_extracted}")
        print(f"Failed: {self.feature_extractor.failed_extractions}")
        print(f"Total features: {len(self.feature_df.columns) - 3}")
        
        return self.feature_df
    
    def prepare_training_data(self):
        print("Preparing training data...")
        
        feature_cols = [col for col in self.feature_df.columns 
                       if col not in ['file_path', 'word', 'speaker']]
        
        X = self.feature_df[feature_cols].values
        y = self.feature_df['word'].values
        
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.feature_names = feature_cols
        
        print(f"Data prepared:")
        print(f"Samples: {X_scaled.shape[0]}")
        print(f"Features: {X_scaled.shape[1]}")
        print(f"Words: {len(self.label_encoder.classes_)}")
        
        return X_scaled, y_encoded
    
    def train_multiple_models(self, X, y):
        print("Training models...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE, stratify=y
        )
        
        models_config = {
            'ExtraTrees': ExtraTreesClassifier(
                random_state=Config.RANDOM_STATE, n_jobs=-1,
                **Config.MODEL_CONFIGS['ExtraTrees']
            ),
            
            'RandomForest': RandomForestClassifier(
                random_state=Config.RANDOM_STATE, n_jobs=-1,
                **Config.MODEL_CONFIGS['RandomForest']
            ),
            
            'GradientBoosting': GradientBoostingClassifier(
                random_state=Config.RANDOM_STATE,
                **Config.MODEL_CONFIGS['GradientBoosting']
            ),
            
            'MLP': MLPClassifier(
                random_state=Config.RANDOM_STATE,
                **Config.MODEL_CONFIGS['MLP']
            ),
            
            'SVM': SVC(
                random_state=Config.RANDOM_STATE,
                **Config.MODEL_CONFIGS['SVM']
            )
        }
        
        if XGBOOST_AVAILABLE:
            models_config['XGBoost'] = xgb.XGBClassifier(
                random_state=Config.RANDOM_STATE,
                **Config.MODEL_CONFIGS['XGBoost']
            )
        
        if LIGHTGBM_AVAILABLE:
            models_config['LightGBM'] = lgb.LGBMClassifier(
                random_state=Config.RANDOM_STATE,
                **Config.MODEL_CONFIGS['LightGBM']
            )
        
        results = {}
        
        for name, model in models_config.items():
            print(f"Training {name}...")
            
            try:
                model.fit(X_train, y_train)
                
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)
                
                train_acc = accuracy_score(y_train, train_pred)
                test_acc = accuracy_score(y_test, test_pred)
                
                cv_scores = cross_val_score(model, X, y, cv=Config.CROSS_VALIDATION_FOLDS, scoring='accuracy')
                cv_mean = np.mean(cv_scores)
                cv_std = np.std(cv_scores)
                
                results[name] = {
                    'model': model,
                    'train_accuracy': train_acc,
                    'test_accuracy': test_acc,
                    'cv_mean': cv_mean,
                    'cv_std': cv_std
                }
                
                print(f"{name}: Test={test_acc:.4f}, CV={cv_mean:.4f}±{cv_std:.4f}")
                
            except Exception as e:
                print(f"{name} failed: {e}")
                continue
        
        self.models = {name: result['model'] for name, result in results.items()}
        self.results = results
        
        return results
    
    def create_ensemble(self):
        print("Creating ensemble...")
        
        if not self.results:
            print("No models to ensemble")
            return None
        
        sorted_models = sorted(self.results.items(), 
                              key=lambda x: x[1]['cv_mean'], reverse=True)
        
        print("Model ranking:")
        for i, (name, result) in enumerate(sorted_models, 1):
            print(f"{i}. {name}: {result['cv_mean']:.4f}±{result['cv_std']:.4f}")
        
        top_models = sorted_models[:min(5, len(sorted_models))]
        
        estimators = [(name, result['model']) for name, result in top_models]
        
        ensemble_model = VotingClassifier(
            estimators=estimators, 
            voting='soft',
            n_jobs=-1
        )
        
        X, y = self.prepare_training_data()
        ensemble_model.fit(X, y)
        
        ensemble_scores = cross_val_score(ensemble_model, X, y, cv=Config.CROSS_VALIDATION_FOLDS, scoring='accuracy')
        ensemble_mean = np.mean(ensemble_scores)
        ensemble_std = np.std(ensemble_scores)
        
        print(f"Ensemble performance: {ensemble_mean:.4f}±{ensemble_std:.4f}")
        
        self.models['Ensemble'] = ensemble_model
        self.results['Ensemble'] = {
            'model': ensemble_model,
            'cv_mean': ensemble_mean,
            'cv_std': ensemble_std
        }
        
        return ensemble_model
    
    def save_complete_system(self, output_dir=None):
        print("Saving system...")
        
        output_path = Path(output_dir or Config.OUTPUT_DIRECTORY)
        output_path.mkdir(exist_ok=True)
        
        system_data = {
            'models': self.models,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'results': self.results,
            'word_counts': self.word_counts,
            'speaker_profiles': self.data_loader.speaker_profiles,
            'training_summary': {
                'total_samples': len(self.feature_df),
                'total_features': len(self.feature_names),
                'total_words': len(self.label_encoder.classes_),
                'feature_extraction_success': self.feature_extractor.features_extracted,
                'feature_extraction_failed': self.feature_extractor.failed_extractions
            }
        }
        
        with open(output_path / 'complete_asr_system.pkl', 'wb') as f:
            pickle.dump(system_data, f)
        
        self.feature_df.to_csv(output_path / 'processed_features.csv', 
                               index=False, encoding='utf-8-sig')
        
        results_report = {
            'timestamp': datetime.now().isoformat(),
            'dataset_info': system_data['training_summary'],
            'model_performance': {
                name: {
                    'cv_mean': result.get('cv_mean', 0),
                    'cv_std': result.get('cv_std', 0),
                    'test_accuracy': result.get('test_accuracy', 0)
                }
                for name, result in self.results.items()
            },
            'best_model': max(self.results.items(), 
                             key=lambda x: x[1].get('cv_mean', 0))[0],
            'words_list': self.label_encoder.classes_.tolist()
        }
        
        with open(output_path / 'training_report.json', 'w', encoding='utf-8') as f:
            json.dump(results_report, f, ensure_ascii=False, indent=2)
        
        print(f"System saved to: {output_path}")
        
        return output_path
    
    def predict_audio(self, audio_file, speaker_name=None):
        if not self.models or self.scaler is None:
            print("System not trained")
            return None
        
        features = self.feature_extractor.extract_comprehensive_features(
            audio_file, speaker=speaker_name
        )
        
        feature_vector = np.array([features.get(name, 0.0) for name in self.feature_names])
        feature_vector = feature_vector.reshape(1, -1)
        
        feature_scaled = self.scaler.transform(feature_vector)
        
        best_model_name = max(self.results.items(), 
                             key=lambda x: x[1].get('cv_mean', 0))[0]
        best_model = self.models[best_model_name]
        
        prediction = best_model.predict(feature_scaled)[0]
        
        if hasattr(best_model, 'predict_proba'):
            probabilities = best_model.predict_proba(feature_scaled)[0]
            confidence = np.max(probabilities)
            
            top_3_indices = np.argsort(probabilities)[::-1][:3]
            top_3 = [(self.label_encoder.inverse_transform([idx])[0], probabilities[idx]) 
                     for idx in top_3_indices]
        else:
            confidence = 0.5
            top_3 = [(self.label_encoder.inverse_transform([prediction])[0], confidence)]
        
        predicted_word = self.label_encoder.inverse_transform([prediction])[0]
        
        return {
            'predicted_word': predicted_word,
            'confidence': confidence,
            'top_3': top_3,
            'model_used': best_model_name
        }
    
    def run_complete_pipeline(self):
        print("Starting ASR system")
        print("=" * 60)
        
        if self.load_and_process_data() is None:
            print("Failed to load data!")
            return None
        
        if len(self.extract_all_features()) == 0:
            print("Failed to extract features!")
            return None
        
        X, y = self.prepare_training_data()
        
        results = self.train_multiple_models(X, y)
        
        ensemble = self.create_ensemble()
        
        output_path = self.save_complete_system()
        
        self.display_final_results()
        
        return output_path
    
    def display_final_results(self):
        print("\n" + "=" * 60)
        print("Final Results")
        print("=" * 60)
        
        print(f"Dataset info:")
        print(f"Total samples: {len(self.feature_df)}")
        print(f"Total features: {len(self.feature_names)}")
        print(f"Total words: {len(self.label_encoder.classes_)}")
        print(f"Total speakers: {self.feature_df['speaker'].nunique()}")
        
        print(f"\nModel performance:")
        sorted_results = sorted(self.results.items(), 
                               key=lambda x: x[1].get('cv_mean', 0), reverse=True)
        
        for i, (name, result) in enumerate(sorted_results, 1):
            cv_mean = result.get('cv_mean', 0)
            cv_std = result.get('cv_std', 0)
            print(f"{i}. {name}: {cv_mean:.4f}±{cv_std:.4f} ({cv_mean*100:.2f}%)")
        
        best_model_name, best_result = sorted_results[0]
        print(f"\nBest model: {best_model_name}")
        print(f"Accuracy: {best_result.get('cv_mean', 0)*100:.2f}%")
        
        print(f"\nFeature extraction stats:")
        print(f"Success: {self.feature_extractor.features_extracted}")
        print(f"Failed: {self.feature_extractor.failed_extractions}")
        success_rate = (self.feature_extractor.features_extracted / 
                       (self.feature_extractor.features_extracted + self.feature_extractor.failed_extractions))
        print(f"Success rate: {success_rate*100:.1f}%")
        
        print(f"\nSystem completed successfully!")