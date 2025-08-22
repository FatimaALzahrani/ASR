import os
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from datetime import datetime

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from data_loader import RealisticDataLoader
from feature_extractor import ConservativeFeatureExtractor
from overfitting_detector import OverfittingDetector

class RealisticASRSystem:
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.data_loader = RealisticDataLoader(data_path)
        self.feature_extractor = ConservativeFeatureExtractor()
        self.overfitting_detector = OverfittingDetector()
        
        self.models = {}
        self.results = {}
        self.scaler = None
        self.label_encoder = None
        self.feature_selector = None
        self.selected_features = None
        
    def load_and_process_data(self):
        print("Loading and processing data...")
        
        data_result = self.data_loader.scan_audio_files_with_sessions()
        
        if data_result is None:
            return None
            
        self.df, self.word_counts = data_result
        
        min_samples_per_word = 5
        valid_words = [word for word, count in self.word_counts.items() 
                      if count >= min_samples_per_word]
        
        self.filtered_df = self.df[self.df['word'].isin(valid_words)]
        
        print(f"Filtered dataset:")
        print(f"   Samples: {len(self.filtered_df)}")
        print(f"   Words: {len(valid_words)}")
        print(f"   Avg samples/word: {len(self.filtered_df)/len(valid_words):.1f}")
        
        return self.filtered_df
    
    def extract_conservative_features(self):
        print("Extracting conservative features...")
        
        all_features = []
        total_files = len(self.filtered_df)
        
        for idx, row in self.filtered_df.iterrows():
            if idx % 100 == 0:
                print(f"   Processing {idx}/{total_files}...")
            
            features = self.feature_extractor.extract_essential_features(
                row['file_path'], row['word'], row['speaker']
            )
            
            combined_row = {
                'file_path': row['file_path'],
                'word': row['word'],
                'speaker': row['speaker'],
                'session_id': row['session_id'],
                **features
            }
            
            all_features.append(combined_row)
        
        self.feature_df = pd.DataFrame(all_features)
        
        feature_count = len([col for col in self.feature_df.columns 
                           if col not in ['file_path', 'word', 'speaker', 'session_id']])
        
        print(f"Feature extraction completed:")
        print(f"   Successful: {self.feature_extractor.features_extracted}")
        print(f"   Failed: {self.feature_extractor.failed_extractions}")
        print(f"   Features: {feature_count}")
        print(f"   Features/samples ratio: {feature_count/len(self.feature_df):.3f}")
        
        return self.feature_df
    
    def prepare_realistic_training_data(self, max_features=25):
        print("Preparing training data...")
        
        feature_cols = [col for col in self.feature_df.columns 
                       if col not in ['file_path', 'word', 'speaker', 'session_id']]
        
        X = self.feature_df[feature_cols].values
        y = self.feature_df['word'].values
        speakers = self.feature_df['speaker'].values
        sessions = self.feature_df['session_id'].values
        
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        print(f"Selecting best {max_features} features from {len(feature_cols)}...")
        
        self.feature_selector = SelectKBest(f_classif, k=min(max_features, len(feature_cols)))
        X_selected = self.feature_selector.fit_transform(X_scaled, y_encoded)
        
        selected_indices = self.feature_selector.get_support(indices=True)
        self.selected_features = [feature_cols[i] for i in selected_indices]
        
        print(f"Training data ready:")
        print(f"   Samples: {X_selected.shape[0]}")
        print(f"   Selected features: {X_selected.shape[1]}")
        print(f"   Words: {len(self.label_encoder.classes_)}")
        print(f"   Safe ratio: {X_selected.shape[1]/X_selected.shape[0]:.4f} (should be < 0.1)")
        
        if X_selected.shape[1]/X_selected.shape[0] > 0.1:
            print("   Warning: Feature/sample ratio still high!")
        
        return X_selected, y_encoded, speakers, sessions
    
    def train_conservative_models(self, X, y, speakers, sessions):
        print("Training conservative models...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        models_config = {
            'Conservative_ExtraTrees': ExtraTreesClassifier(
                n_estimators=20,
                max_depth=4,
                min_samples_split=8,
                min_samples_leaf=5,
                max_features=5,
                bootstrap=True,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            
            'Conservative_RandomForest': RandomForestClassifier(
                n_estimators=15,
                max_depth=3,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features=3,
                bootstrap=True,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            
            'Regularized_LogisticRegression': LogisticRegression(
                C=0.1,
                penalty='l2',
                max_iter=1000,
                class_weight='balanced',
                random_state=42
            ),
            
            'Simple_SVM': SVC(
                kernel='rbf',
                C=0.1,
                gamma='scale',
                class_weight='balanced',
                probability=True,
                random_state=42
            )
        }
        
        if XGBOOST_AVAILABLE:
            models_config['Conservative_XGBoost'] = xgb.XGBClassifier(
                n_estimators=30,
                max_depth=3,
                learning_rate=0.3,
                subsample=0.8,
                colsample_bytree=0.6,
                reg_alpha=1.0,
                reg_lambda=1.0,
                random_state=42,
                eval_metric='mlogloss',
                verbosity=0
            )
        
        results = {}
        
        for name, model in models_config.items():
            print(f"   Training {name}...")
            
            try:
                model.fit(X_train, y_train)
                
                test_pred = model.predict(X_test)
                test_accuracy = accuracy_score(y_test, test_pred)
                
                cv_scores = cross_val_score(
                    model, X, y, 
                    cv=3,
                    scoring='accuracy'
                )
                
                overfitting_analysis = self.overfitting_detector.detect_overfitting(
                    cv_scores, test_accuracy
                )
                
                results[name] = {
                    'model': model,
                    'test_accuracy': test_accuracy,
                    'cv_scores': cv_scores,
                    'overfitting_analysis': overfitting_analysis
                }
                
                print(f"      {name}:")
                print(f"         Test: {test_accuracy:.4f}")
                print(f"         CV: {overfitting_analysis['cv_mean']:.4f} ± {overfitting_analysis['cv_std']:.4f}")
                print(f"         Overfitting: {overfitting_analysis['severity']}")
                
                if overfitting_analysis['is_overfitted']:
                    print(f"         Warning: Generalization issue!")
                    recommendations = self.overfitting_detector.recommend_fixes(overfitting_analysis)
                    print(f"         Recommendation: {recommendations[0] if recommendations else 'None'}")
                else:
                    print(f"         Status: Healthy model")
                
            except Exception as e:
                print(f"      {name}: Error - {e}")
                continue
        
        self.models = {name: result['model'] for name, result in results.items()}
        self.results = results
        
        return results
    
    def get_realistic_performance_report(self):
        print("\n" + "="*80)
        print("Realistic ASR Performance Report")
        print("="*80)
        
        if not self.results:
            print("No results available for evaluation")
            return None
        
        healthy_models = []
        overfitted_models = []
        
        for name, result in self.results.items():
            analysis = result['overfitting_analysis']
            
            if analysis['is_overfitted']:
                overfitted_models.append((name, result))
            else:
                healthy_models.append((name, result))
        
        healthy_models.sort(key=lambda x: x[1]['overfitting_analysis']['cv_mean'], reverse=True)
        overfitted_models.sort(key=lambda x: x[1]['overfitting_analysis']['cv_mean'], reverse=True)
        
        print("Healthy Models (no overfitting):")
        if healthy_models:
            for i, (name, result) in enumerate(healthy_models, 1):
                analysis = result['overfitting_analysis']
                print(f"   {i}. {name}")
                print(f"      CV: {analysis['cv_mean']:.3f} ± {analysis['cv_std']:.3f}")
                print(f"      Test: {analysis['test_score']:.3f}")
                print(f"      Status: Reliable for production")
                print()
        else:
            print("   No healthy models found!")
        
        print("Overfitted Models:")
        if overfitted_models:
            for i, (name, result) in enumerate(overfitted_models, 1):
                analysis = result['overfitting_analysis']
                print(f"   {i}. {name}")
                print(f"      CV: {analysis['cv_mean']:.3f} ± {analysis['cv_std']:.3f}")
                print(f"      Test: {analysis['test_score']:.3f}")
                print(f"      Overfitting level: {analysis['severity']}")
                print()
        
        print("Performance Analysis:")
        
        if healthy_models:
            best_healthy = healthy_models[0]
            best_cv = best_healthy[1]['overfitting_analysis']['cv_mean']
            print(f"   Best reliable model: {best_healthy[0]}")
            print(f"   Realistic accuracy: {best_cv:.3f} ({best_cv*100:.1f}%)")
            
            if best_cv >= 0.80:
                print("   Performance: Excellent (above 80%)")
            elif best_cv >= 0.70:
                print("   Performance: Very good (70-80%)")
            elif best_cv >= 0.60:
                print("   Performance: Acceptable (60-70%)")
            else:
                print("   Performance: Needs improvement (below 60%)")
                
        else:
            print("   All models suffer from overfitting!")
            print("   Suggested solutions:")
            print("      - Collect more data")
            print("      - Reduce features further")
            print("      - Use simpler models")
        
        print(f"\nData Quality Assessment:")
        print(f"   Samples: {len(self.feature_df)}")
        print(f"   Features: {len(self.selected_features)}")
        print(f"   Words: {len(self.label_encoder.classes_)}")
        
        samples_per_word = len(self.feature_df) / len(self.label_encoder.classes_)
        features_to_samples = len(self.selected_features) / len(self.feature_df)
        
        print(f"   Samples per word: {samples_per_word:.1f}")
        print(f"   Features/samples ratio: {features_to_samples:.4f}")
        
        if samples_per_word < 10:
            print("   Warning: Few samples per word (minimum: 10)")
        if features_to_samples > 0.1:
            print("   Warning: High features/samples ratio (prefer < 0.1)")
        
        return {
            'healthy_models': healthy_models,
            'overfitted_models': overfitted_models,
            'best_model': healthy_models[0] if healthy_models else None,
            'data_quality': {
                'samples_per_word': samples_per_word,
                'features_to_samples_ratio': features_to_samples
            }
        }
    
    def save_realistic_system(self, output_dir="realistic_asr_output"):
        print("Saving realistic system...")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        system_data = {
            'models': self.models,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_selector': self.feature_selector,
            'selected_features': self.selected_features,
            'results': self.results,
            'word_counts': self.word_counts,
            'speaker_profiles': self.data_loader.speaker_profiles,
            'training_summary': {
                'total_samples': len(self.feature_df),
                'selected_features': len(self.selected_features),
                'total_words': len(self.label_encoder.classes_),
                'extraction_success': self.feature_extractor.features_extracted,
                'extraction_failed': self.feature_extractor.failed_extractions,
                'realistic_training': True
            }
        }
        
        with open(output_path / 'realistic_asr_system.pkl', 'wb') as f:
            pickle.dump(system_data, f)
        
        self.feature_df.to_csv(output_path / 'processed_features_realistic.csv', 
                               index=False, encoding='utf-8-sig')
        
        performance_report = self.get_realistic_performance_report()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'system_type': 'Realistic ASR (Anti-Overfitting)',
            'dataset_info': system_data['training_summary'],
            'performance_analysis': {
                name: {
                    'cv_mean': result['overfitting_analysis']['cv_mean'],
                    'cv_std': result['overfitting_analysis']['cv_std'],
                    'test_accuracy': result['overfitting_analysis']['test_score'],
                    'is_overfitted': result['overfitting_analysis']['is_overfitted'],
                    'overfitting_severity': result['overfitting_analysis']['severity']
                }
                for name, result in self.results.items()
            },
            'best_reliable_model': performance_report['best_model'][0] if performance_report and performance_report['best_model'] else None,
            'selected_features': self.selected_features,
            'recommendations': [
                "Use only healthy models for production",
                "Test on new data before deployment",
                "Monitor performance in real usage"
            ]
        }
        
        with open(output_path / 'realistic_training_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"System saved to: {output_path}")
        
        return output_path
    
    def predict_with_confidence(self, audio_file, speaker_name=None):
        if not self.models or self.scaler is None:
            print("System not trained")
            return None
        
        healthy_models = [
            (name, result) for name, result in self.results.items()
            if not result['overfitting_analysis']['is_overfitted']
        ]
        
        if healthy_models:
            best_model_name = max(healthy_models, 
                                key=lambda x: x[1]['overfitting_analysis']['cv_mean'])[0]
            model_status = 'healthy'
        else:
            best_model_name = min(
                self.results.items(),
                key=lambda x: x[1]['overfitting_analysis']['indicators']['cv_test_gap']
            )[0]
            model_status = 'overfitted'
            print("Warning: Using overfitted model")
        
        features = self.feature_extractor.extract_essential_features(
            audio_file, speaker=speaker_name
        )
        
        feature_cols = [col for col in self.feature_df.columns 
                       if col not in ['file_path', 'word', 'speaker', 'session_id']]
        
        feature_vector = np.array([features.get(name, 0.0) for name in feature_cols])
        feature_vector = feature_vector.reshape(1, -1)
        feature_scaled = self.scaler.transform(feature_vector)
        feature_selected = self.feature_selector.transform(feature_scaled)
        
        best_model = self.models[best_model_name]
        prediction = best_model.predict(feature_selected)[0]
        
        if hasattr(best_model, 'predict_proba'):
            probabilities = best_model.predict_proba(feature_selected)[0]
            confidence = np.max(probabilities)
            
            top_3_indices = np.argsort(probabilities)[::-1][:3]
            top_3 = [(self.label_encoder.inverse_transform([idx])[0], probabilities[idx]) 
                     for idx in top_3_indices]
        else:
            confidence = 0.5
            top_3 = [(self.label_encoder.inverse_transform([prediction])[0], confidence)]
        
        predicted_word = self.label_encoder.inverse_transform([prediction])[0]
        
        reliability = 'high' if model_status == 'healthy' and confidence > 0.7 else 'medium'
        if model_status == 'overfitted' or confidence < 0.5:
            reliability = 'low'
        
        return {
            'predicted_word': predicted_word,
            'confidence': confidence,
            'top_3': top_3,
            'model_used': best_model_name,
            'model_status': model_status,
            'prediction_reliability': reliability,
            'warning': 'Prediction from overfitted model' if model_status == 'overfitted' else None
        }
    
    def run_realistic_pipeline(self):
        print("Starting Realistic ASR System (Anti-Overfitting)")
        print("=" * 80)
        
        if self.load_and_process_data() is None:
            print("Failed to load data!")
            return None
        
        if len(self.extract_conservative_features()) == 0:
            print("Failed to extract features!")
            return None
        
        X, y, speakers, sessions = self.prepare_realistic_training_data(max_features=20)
        
        results = self.train_conservative_models(X, y, speakers, sessions)
        
        performance_report = self.get_realistic_performance_report()
        
        output_path = self.save_realistic_system()
        
        print("\n" + "="*80)
        print("Realistic System Summary")
        print("="*80)
        
        if performance_report and performance_report['best_model']:
            best_model_name, best_result = performance_report['best_model']
            best_cv = best_result['overfitting_analysis']['cv_mean']
            
            print(f"Best reliable model: {best_model_name}")
            print(f"Realistic accuracy: {best_cv:.3f} ({best_cv*100:.1f}%)")
            print(f"These are realistic and reliable results for practical use!")
        else:
            print("No reliable models found - data needs improvement")
        
        return output_path