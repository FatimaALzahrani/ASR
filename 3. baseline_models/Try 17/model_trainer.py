import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import xgboost as xgb
import lightgbm as lgb

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.scalers = {}
    
    def train_enhanced_model(self, speaker_name, df):
        print(f"Training enhanced model for speaker: {speaker_name}")
        
        speaker_profile = self._get_speaker_profile_from_df(df)
        quality = speaker_profile.get("quality", "متوسط")
        clarity = speaker_profile.get("clarity", 0.5)
        
        print(f"Samples: {len(df)}")
        print(f"Words: {len(df['word'].unique())}")
        print(f"Quality: {quality} (clarity: {clarity:.2f})")
        
        try:
            feature_cols = [col for col in df.columns if col not in [
                'file_path', 'word', 'speaker', 'name', 'quality', 'clarity', 
                'strategy', 'target_words', 'augment_factor'
            ]]
            
            X = df[feature_cols].values
            y = df['word'].values
            
            print(f"Features: {len(feature_cols)}")
            
            word_counts = pd.Series(y).value_counts()
            words_to_keep = word_counts[word_counts >= 2].index
            
            if len(words_to_keep) < 2:
                print(f"Not enough words with multiple samples")
                return None
            
            mask = pd.Series(y).isin(words_to_keep)
            X = X[mask]
            y = y[mask]
            
            print(f"Final words: {len(words_to_keep)}")
            print(f"Final samples: {len(y)}")
            
            imputer = SimpleImputer(strategy='constant', fill_value=0.0)
            X_imputed = imputer.fit_transform(X)
            X_clean = np.nan_to_num(X_imputed, nan=0.0, posinf=1e6, neginf=-1e6)
            
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X_clean)
            
            n_features_target = min(150, X_scaled.shape[1]//2, len(y)//3)
            n_features_target = max(20, n_features_target)
            
            try:
                selector = SelectKBest(score_func=mutual_info_classif, k=n_features_target)
                X_selected = selector.fit_transform(X_scaled, y)
            except:
                try:
                    selector = SelectKBest(score_func=f_classif, k=n_features_target)
                    X_selected = selector.fit_transform(X_scaled, y)
                except:
                    X_selected = X_scaled[:, :n_features_target]
                    selector = None
            
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
            
            test_size = min(0.25, max(0.15, 1.0 - 10/len(y)))
            
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_selected, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
                )
            except:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_selected, y_encoded, test_size=test_size, random_state=42
                )
            
            print(f"Training: {len(X_train)} samples, testing: {len(X_test)} samples")
            
            models = self._get_models_for_quality(quality)
            
            model_results = {}
            successful_models = []
            
            for name, model in models:
                try:
                    print(f"Training {name}...")
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    model_results[name] = accuracy
                    successful_models.append((name, model))
                    
                    print(f"{name}: {accuracy:.4f} ({accuracy*100:.2f}%)")
                    
                except Exception as e:
                    print(f"Error in {name}: {e}")
                    model_results[name] = 0.0
            
            best_model, best_name, best_score = self._create_ensemble(successful_models, model_results, X_train, y_train, X_test, y_test)
            
            if best_model is not None:
                self.models[speaker_name] = {
                    'model': best_model,
                    'model_name': best_name,
                    'accuracy': best_score,
                    'feature_count': X_selected.shape[1],
                    'word_count': len(words_to_keep),
                    'sample_count': len(y),
                    'profile': speaker_profile
                }
                
                self.scalers[speaker_name] = {
                    'imputer': imputer,
                    'scaler': scaler,
                    'selector': selector,
                    'label_encoder': label_encoder
                }
                
                print(f"Best model: {best_name} - {best_score:.4f} ({best_score*100:.2f}%)")
                
                return {
                    'speaker': speaker_name,
                    'best_model': best_name,
                    'best_accuracy': best_score,
                    'all_results': model_results,
                    'samples': len(y),
                    'words': len(words_to_keep),
                    'quality': quality,
                    'clarity': clarity
                }
            else:
                print(f"Failed to train all models")
                return None
                
        except Exception as e:
            print(f"General training error: {e}")
            return None
    
    def _get_speaker_profile_from_df(self, df):
        return {
            "quality": df.iloc[0].get("quality", "متوسط"),
            "clarity": df.iloc[0].get("clarity", 0.5)
        }
    
    def _get_models_for_quality(self, quality):
        if quality == "عالي":
            return [
                ('ExtraTrees', ExtraTreesClassifier(n_estimators=500, max_depth=15, random_state=42, n_jobs=-1)),
                ('RandomForest', RandomForestClassifier(n_estimators=500, max_depth=15, random_state=42, n_jobs=-1)),
                ('XGBoost', xgb.XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1, random_state=42)),
                ('GradientBoosting', GradientBoostingClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, random_state=42))
            ]
        elif quality in ["متوسط-عالي", "متوسط"]:
            return [
                ('RandomForest', RandomForestClassifier(n_estimators=400, max_depth=12, random_state=42, n_jobs=-1)),
                ('ExtraTrees', ExtraTreesClassifier(n_estimators=400, max_depth=12, random_state=42, n_jobs=-1)),
                ('XGBoost', xgb.XGBClassifier(n_estimators=250, max_depth=6, learning_rate=0.15, random_state=42)),
                ('LightGBM', lgb.LGBMClassifier(n_estimators=250, max_depth=8, learning_rate=0.15, random_state=42, verbosity=-1))
            ]
        else:
            return [
                ('RandomForest', RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_split=5, random_state=42, n_jobs=-1)),
                ('ExtraTrees', ExtraTreesClassifier(n_estimators=300, max_depth=10, min_samples_split=5, random_state=42, n_jobs=-1)),
                ('LogisticRegression', LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')),
                ('SVM', SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42))
            ]
    
    def _create_ensemble(self, successful_models, model_results, X_train, y_train, X_test, y_test):
        best_model = None
        best_score = 0
        best_name = ""
        
        if len(successful_models) >= 2:
            try:
                print(f"Creating enhanced ensemble...")
                
                sorted_models = sorted(successful_models, key=lambda x: model_results[x[0]], reverse=True)
                top_models = sorted_models[:min(4, len(sorted_models))]
                
                weights = [model_results[name] for name, _ in top_models]
                total_weight = sum(weights)
                if total_weight > 0:
                    normalized_weights = [w/total_weight for w in weights]
                else:
                    normalized_weights = [1/len(top_models)] * len(top_models)
                
                voting_ensemble = VotingClassifier(
                    estimators=top_models,
                    voting='soft' if all(hasattr(model, 'predict_proba') for _, model in top_models) else 'hard'
                )
                
                voting_ensemble.fit(X_train, y_train)
                y_pred_ensemble = voting_ensemble.predict(X_test)
                ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
                
                print(f"Ensemble: {ensemble_accuracy:.4f} ({ensemble_accuracy*100:.2f}%)")
                
                all_models = successful_models + [("Ensemble", voting_ensemble)]
                all_results = {**model_results, "Ensemble": ensemble_accuracy}
                
                best_name = max(all_results.keys(), key=lambda x: all_results[x])
                best_score = all_results[best_name]
                
                if best_name == "Ensemble":
                    best_model = voting_ensemble
                else:
                    best_model = next(model for name, model in successful_models if name == best_name)
                    
            except Exception as e:
                print(f"Ensemble error: {e}")
                if successful_models:
                    best_name = max(model_results.keys(), key=lambda x: model_results[x])
                    best_score = model_results[best_name]
                    best_model = next(model for name, model in successful_models if name == best_name)
        
        elif successful_models:
            best_name = max(model_results.keys(), key=lambda x: model_results[x])
            best_score = model_results[best_name]
            best_model = next(model for name, model in successful_models if name == best_name)
        
        return best_model, best_name, best_score