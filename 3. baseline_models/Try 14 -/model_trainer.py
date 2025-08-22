import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb

class ModelTrainer:
    
    def __init__(self):
        self.speaker_models = {}
    
    def train_model(self, speaker_name, df, data_processor, speaker_profile):
        print(f"Training model for speaker: {speaker_name}")
        
        print(f"Samples: {len(df)}")
        print(f"Words: {len(df['word'].unique())}")
        print(f"Quality: {speaker_profile['quality']}")
        
        min_samples = speaker_profile.get("min_samples", 15)
        if len(df) < min_samples:
            print(f"Insufficient samples for training ({len(df)} < {min_samples})")
            return None
        
        try:
            feature_cols = [col for col in df.columns if col not in [
                'file_path', 'word', 'speaker', 'name', 'quality', 'clarity', 
                'min_samples', 'preferred_models'
            ]]
            X = df[feature_cols].values
            y = df['word'].values
            
            print(f"Initial features: {len(feature_cols)}")
            
            word_counts = pd.Series(y).value_counts()
            words_to_keep = word_counts[word_counts >= 2].index
            
            if len(words_to_keep) < 2:
                print(f"Not enough words with multiple samples")
                return None
            
            mask = pd.Series(y).isin(words_to_keep)
            X = X[mask]
            y = y[mask]
            
            print(f"Kept {len(words_to_keep)} words out of {len(word_counts)} words")
            print(f"Final samples: {len(y)}")
            
            if len(y) < 10:
                print(f"Insufficient samples after cleaning ({len(y)} < 10)")
                return None
            
            X_processed = data_processor.prepare_data(X, y, speaker_name)
            y_encoded, label_encoder = data_processor.encode_labels(y)
            data_processor.save_scaler(speaker_name, label_encoder)
            
            test_size = min(0.3, max(0.15, 1.0 - 8/len(y)))
            
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_processed, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
                )
                print(f"Split with stratify")
            except ValueError:
                try:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_processed, y_encoded, test_size=test_size, random_state=42
                    )
                    print(f"Split without stratify")
                except Exception:
                    print(f"Split failed completely")
                    return None
            
            print(f"Training: {len(X_train)} samples, Testing: {len(X_test)} samples")
            
            preferred_models = speaker_profile.get("preferred_models", ["RandomForest"])
            
            available_models = self._get_available_models()
            
            models_to_train = []
            for model_name in preferred_models:
                if model_name in available_models:
                    models_to_train.append((model_name, available_models[model_name]))
            
            if len(models_to_train) < 2:
                models_to_train.append(('RandomForest', available_models['RandomForest']))
                if 'LogisticRegression' not in [m[0] for m in models_to_train]:
                    models_to_train.append(('LogisticRegression', available_models['LogisticRegression']))
            
            best_model, best_score, best_name, model_results, successful_models = self._train_individual_models(
                models_to_train, X_train, X_test, y_train, y_test
            )
            
            if len(successful_models) >= 2 and best_score > 0.3:
                ensemble_model, ensemble_score, ensemble_name = self._train_ensemble(
                    successful_models, model_results, X_train, X_test, y_train, y_test
                )
                
                if ensemble_score > best_score:
                    best_model = ensemble_model
                    best_score = ensemble_score
                    best_name = ensemble_name
                    model_results[best_name] = ensemble_score
            
            if best_model is not None:
                self.speaker_models[speaker_name] = {
                    'model': best_model,
                    'model_name': best_name,
                    'accuracy': best_score,
                    'feature_count': X_processed.shape[1],
                    'word_count': len(np.unique(y)),
                    'sample_count': len(y),
                    'profile': speaker_profile
                }
                
                print(f"Best model: {best_name} - {best_score:.4f} ({best_score*100:.2f}%)")
                
                return {
                    'speaker': speaker_name,
                    'best_model': best_name,
                    'best_accuracy': best_score,
                    'all_results': model_results,
                    'samples': len(y),
                    'words': len(np.unique(y)),
                    'quality': speaker_profile.get('quality', 'medium'),
                    'clarity': speaker_profile.get('clarity', 0.5)
                }
            else:
                print(f"All models failed to train")
                return None
                
        except Exception as e:
            print(f"Training failed: {e}")
            return None
    
    def _get_available_models(self):
        return {
            'RandomForest': RandomForestClassifier(
                n_estimators=300, max_depth=12, min_samples_split=3,
                min_samples_leaf=2, class_weight='balanced', random_state=42, n_jobs=-1
            ),
            'ExtraTrees': ExtraTreesClassifier(
                n_estimators=300, max_depth=12, min_samples_split=3,
                min_samples_leaf=2, class_weight='balanced', random_state=42, n_jobs=-1
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                random_state=42, eval_metric='mlogloss', verbosity=0
            ),
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=200, max_depth=8, learning_rate=0.1,
                random_state=42, verbosity=-1
            ),
            'LogisticRegression': LogisticRegression(
                random_state=42, max_iter=1000, class_weight='balanced'
            ),
            'SVM': SVC(
                kernel='rbf', C=1, gamma='scale', class_weight='balanced',
                probability=True, random_state=42
            )
        }
    
    def _train_individual_models(self, models_to_train, X_train, X_test, y_train, y_test):
        best_model = None
        best_score = 0
        best_name = ""
        model_results = {}
        successful_models = []
        
        for name, model in models_to_train:
            try:
                print(f"Training {name}...")
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                model_results[name] = accuracy
                successful_models.append((name, model))
                
                print(f"{name}: {accuracy:.4f} ({accuracy*100:.2f}%)")
                
                if accuracy > best_score:
                    best_score = accuracy
                    best_model = model
                    best_name = name
                    
            except Exception as e:
                print(f"Error in {name}: {e}")
                model_results[name] = 0.0
        
        return best_model, best_score, best_name, model_results, successful_models
    
    def _train_ensemble(self, successful_models, model_results, X_train, X_test, y_train, y_test):
        try:
            print(f"Creating Ensemble...")
            
            sorted_models = sorted(successful_models, key=lambda x: model_results[x[0]], reverse=True)
            top_models = sorted_models[:min(3, len(sorted_models))]
            
            voting_type = 'soft'
            for name, model in top_models:
                if not hasattr(model, 'predict_proba'):
                    voting_type = 'hard'
                    break
            
            voting_ensemble = VotingClassifier(
                estimators=top_models,
                voting=voting_type
            )
            voting_ensemble.fit(X_train, y_train)
            y_pred_ensemble = voting_ensemble.predict(X_test)
            ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
            
            print(f"Ensemble ({voting_type}): {ensemble_accuracy:.4f} ({ensemble_accuracy*100:.2f}%)")
            
            return voting_ensemble, ensemble_accuracy, f"Ensemble_{voting_type}"
            
        except Exception as e:
            print(f"Ensemble creation failed: {e}")
            return None, 0, ""
    
    def get_model(self, speaker_name):
        return self.speaker_models.get(speaker_name, None)
    
    def get_all_models(self):
        return self.speaker_models
