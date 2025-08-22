#!/usr/bin/env python3

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.impute import SimpleImputer
import xgboost as xgb
import lightgbm as lgb
from collections import Counter

class ModelTrainer:
    
    def train_advanced_model(self, speaker_name, df):
        print(f"\nTraining advanced model for speaker: {speaker_name}")
        
        speaker_profile = self.get_speaker_profile(speaker_name, df)
        quality = speaker_profile.get("overall_quality", "متوسط")
        clarity = speaker_profile.get("clarity", 0.5)
        iq = speaker_profile.get("iq", 50)
        age = speaker_profile.get("age", "5")
        
        print(f"   Samples: {len(df)}")
        print(f"   Words: {len(df['word'].unique())}")
        print(f"   Quality: {quality} (clarity: {clarity:.2f})")
        print(f"   IQ: {iq}, Age: {age}")
        
        try:
            non_feature_cols = [
                'file_path', 'word', 'speaker', 'name', 'age', 'gender', 'iq',
                'overall_quality', 'clarity', 'strategy', 'target_words', 'augment_factor'
            ]
            feature_cols = [col for col in df.columns if col not in non_feature_cols]
            
            X = df[feature_cols].values
            y = df['word'].values
            
            print(f"   Features: {len(feature_cols)}")
            
            word_counts = pd.Series(y).value_counts()
            words_to_keep = word_counts[word_counts >= 3].index
            
            if len(words_to_keep) < 5:
                print(f"   Insufficient words with multiple samples")
                return None
            
            mask = pd.Series(y).isin(words_to_keep)
            X = X[mask]
            y = y[mask]
            
            print(f"   Final words: {len(words_to_keep)}")
            print(f"   Final samples: {len(y)}")
            
            print(f"   Processing data...")
            
            X_cleaned = self.remove_outliers(X)
            
            imputer = SimpleImputer(strategy='median')
            X_imputed = imputer.fit_transform(X_cleaned)
            X_clean = np.nan_to_num(X_imputed, nan=0.0, posinf=1e6, neginf=-1e6)
            
            scaler = self.select_optimal_scaler(X_clean, speaker_profile)
            X_scaled = scaler.fit_transform(X_clean)
            
            print(f"   Feature selection...")
            X_selected, selector = self.advanced_feature_selection(X_scaled, y, speaker_profile)
            
            print(f"   Selected features: {X_selected.shape[1]} from {X_scaled.shape[1]}")
            
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
            
            test_size = self.calculate_optimal_test_size(len(y), len(words_to_keep))
            
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_selected, y_encoded, test_size=test_size, random_state=42, 
                    stratify=y_encoded
                )
            except:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_selected, y_encoded, test_size=test_size, random_state=42
                )
            
            print(f"   Train: {len(X_train)} samples, Test: {len(X_test)} samples")
            
            print(f"   Training advanced models...")
            
            models = self.get_advanced_models(speaker_profile, len(X_train), len(words_to_keep))
            model_results = {}
            successful_models = []
            
            for name, model in models:
                try:
                    print(f"     {name}...")
                    
                    if len(y_train) > 30:
                        cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
                        cv_mean = cv_scores.mean()
                        print(f"       CV: {cv_mean:.4f} (±{cv_scores.std()*2:.4f})")
                    
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    precision = precision_score(y_test, y_pred, average='weighted')
                    recall = recall_score(y_test, y_pred, average='weighted')
                    
                    model_results[name] = {
                        'accuracy': accuracy,
                        'f1_score': f1,
                        'precision': precision,
                        'recall': recall
                    }
                    successful_models.append((name, model))
                    
                    print(f"       Acc: {accuracy:.4f}, F1: {f1:.4f}, Prec: {precision:.4f}, Rec: {recall:.4f}")
                    
                except Exception as e:
                    print(f"       Error with {name}: {e}")
                    model_results[name] = {'accuracy': 0.0, 'f1_score': 0.0, 'precision': 0.0, 'recall': 0.0}
            
            best_model, best_name, best_results = self.create_advanced_ensemble(
                successful_models, model_results, X_train, y_train, X_test, y_test
            )
            
            if best_model is not None:
                print(f"   Best model: {best_name}")
                print(f"       Accuracy: {best_results['accuracy']:.4f} ({best_results['accuracy']*100:.2f}%)")
                print(f"       F1: {best_results['f1_score']:.4f}")
                
                if len(y_test) > 0:
                    self.analyze_errors(best_model, X_test, y_test, label_encoder, speaker_name)
                
                return {
                    'speaker': speaker_name,
                    'best_model': best_name,
                    'results': best_results,
                    'samples': len(y),
                    'words': len(words_to_keep),
                    'quality': quality,
                    'clarity': clarity,
                    'word_list': list(words_to_keep),
                    'model': best_model,
                    'feature_count': X_selected.shape[1],
                    'word_count': len(words_to_keep),
                    'sample_count': len(y),
                    'profile': speaker_profile,
                    'scalers': {
                        'imputer': imputer,
                        'scaler': scaler,
                        'selector': selector,
                        'label_encoder': label_encoder,
                        'feature_columns': feature_cols
                    }
                }
            else:
                print(f"   All models failed to train")
                return None
                
        except Exception as e:
            print(f"   General training error: {e}")
            return None
    
    def get_speaker_profile(self, speaker_name, df):
        if len(df) > 0:
            first_row = df.iloc[0]
            return {
                'name': first_row.get('name', speaker_name),
                'age': first_row.get('age', '5'),
                'overall_quality': first_row.get('overall_quality', 'متوسط'),
                'clarity': first_row.get('clarity', 0.5),
                'iq': first_row.get('iq', 50)
            }
        return {'name': speaker_name, 'age': '5', 'overall_quality': 'متوسط', 'clarity': 0.5, 'iq': 50}
    
    def remove_outliers(self, X, contamination=0.05):
        try:
            from sklearn.ensemble import IsolationForest
            
            iso_forest = IsolationForest(contamination=contamination, random_state=42)
            outlier_labels = iso_forest.fit_predict(X)
            
            normal_mask = outlier_labels == 1
            X_cleaned = X[normal_mask]
            
            return X_cleaned
        except:
            return X
    
    def select_optimal_scaler(self, X, speaker_profile):
        try:
            quality = speaker_profile.get("overall_quality", "متوسط")
            
            if quality == "ممتاز":
                return StandardScaler()
            elif quality == "ضعيف":
                return RobustScaler()
            else:
                return MinMaxScaler()
        except:
            return RobustScaler()
    
    def advanced_feature_selection(self, X, y, speaker_profile):
        try:
            n_samples, n_features = X.shape
            n_classes = len(np.unique(y))
            
            optimal_features = min(
                200,
                n_features // 2,
                n_samples // 4,
                n_classes * 15
            )
            optimal_features = max(30, optimal_features)
            
            selectors_to_try = [
                ('mutual_info', SelectKBest(score_func=mutual_info_classif, k=optimal_features)),
                ('f_classif', SelectKBest(score_func=f_classif, k=optimal_features)),
            ]
            
            if n_samples > 100:
                try:
                    rf_estimator = RandomForestClassifier(n_estimators=50, random_state=42)
                    rfe = RFE(estimator=rf_estimator, n_features_to_select=optimal_features)
                    selectors_to_try.append(('rfe', rfe))
                except:
                    pass
            
            best_selector = None
            best_score = 0
            
            for name, selector in selectors_to_try:
                try:
                    X_selected = selector.fit_transform(X, y)
                    
                    rf = RandomForestClassifier(n_estimators=50, random_state=42)
                    score = cross_val_score(rf, X_selected, y, cv=3, scoring='accuracy').mean()
                    
                    if score > best_score:
                        best_score = score
                        best_selector = selector
                        
                except Exception as e:
                    continue
            
            if best_selector is not None:
                X_selected = best_selector.transform(X)
                return X_selected, best_selector
            else:
                n_features_fallback = min(100, X.shape[1])
                return X[:, :n_features_fallback], None
                
        except Exception as e:
            n_features_fallback = min(100, X.shape[1])
            return X[:, :n_features_fallback], None
    
    def calculate_optimal_test_size(self, n_samples, n_classes):
        min_test_per_class = 2
        min_train_per_class = 3
        
        max_test_size = 0.3
        min_test_size = 0.15
        
        required_test = n_classes * min_test_per_class
        required_train = n_classes * min_train_per_class
        
        if n_samples < required_test + required_train:
            return min_test_size
        
        optimal_test_size = required_test / n_samples
        return max(min_test_size, min(max_test_size, optimal_test_size))
    
    def get_advanced_models(self, speaker_profile, n_samples, n_classes):
        quality = speaker_profile.get("overall_quality", "متوسط")
        
        models = []
        
        if quality == "ممتاز":
            models.append(('RandomForest', RandomForestClassifier(
                n_estimators=800, max_depth=20, min_samples_split=2, 
                min_samples_leaf=1, random_state=42, n_jobs=-1
            )))
        else:
            models.append(('RandomForest', RandomForestClassifier(
                n_estimators=500, max_depth=15, min_samples_split=3, 
                min_samples_leaf=2, random_state=42, n_jobs=-1
            )))
        
        models.append(('ExtraTrees', ExtraTreesClassifier(
            n_estimators=600, max_depth=18, min_samples_split=2,
            random_state=42, n_jobs=-1
        )))
        
        try:
            if quality == "ممتاز":
                models.append(('XGBoost', xgb.XGBClassifier(
                    n_estimators=500, max_depth=10, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, random_state=42
                )))
            else:
                models.append(('XGBoost', xgb.XGBClassifier(
                    n_estimators=300, max_depth=8, learning_rate=0.1,
                    random_state=42
                )))
        except:
            pass
        
        try:
            models.append(('LightGBM', lgb.LGBMClassifier(
                n_estimators=400, max_depth=12, learning_rate=0.08,
                random_state=42, verbosity=-1
            )))
        except:
            pass
        
        models.append(('GradientBoosting', GradientBoostingClassifier(
            n_estimators=400, max_depth=8, learning_rate=0.08, random_state=42
        )))
        
        if n_samples < 500:
            models.append(('SVM', SVC(
                kernel='rbf', C=10, gamma='scale', probability=True, random_state=42
            )))
        
        models.append(('LogisticRegression', LogisticRegression(
            random_state=42, max_iter=2000, class_weight='balanced'
        )))
        
        return models
    
    def create_advanced_ensemble(self, successful_models, model_results, X_train, y_train, X_test, y_test):
        try:
            if len(successful_models) < 2:
                if successful_models:
                    name, model = successful_models[0]
                    results = model_results[name]
                    return model, name, results
                else:
                    return None, None, None
            
            print(f"     Creating ensemble from {len(successful_models)} models...")
            
            sorted_models = sorted(
                successful_models, 
                key=lambda x: model_results[x[0]]['accuracy'], 
                reverse=True
            )
            
            top_models = sorted_models[:min(4, len(sorted_models))]
            
            ensemble_methods = []
            
            try:
                voting_soft = VotingClassifier(
                    estimators=top_models, voting='soft'
                )
                voting_soft.fit(X_train, y_train)
                
                y_pred = voting_soft.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                
                ensemble_methods.append(('VotingSoft', voting_soft, {
                    'accuracy': accuracy, 'f1_score': f1, 'precision': precision, 'recall': recall
                }))
                
                print(f"       VotingSoft: {accuracy:.4f}")
            except:
                pass
            
            try:
                weights = [model_results[name]['accuracy'] for name, _ in top_models]
                total_weight = sum(weights)
                normalized_weights = [w/total_weight for w in weights] if total_weight > 0 else None
                
                if normalized_weights:
                    voting_weighted = VotingClassifier(
                        estimators=top_models, voting='soft', weights=normalized_weights
                    )
                    voting_weighted.fit(X_train, y_train)
                    
                    y_pred = voting_weighted.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    precision = precision_score(y_test, y_pred, average='weighted')
                    recall = recall_score(y_test, y_pred, average='weighted')
                    
                    ensemble_methods.append(('VotingWeighted', voting_weighted, {
                        'accuracy': accuracy, 'f1_score': f1, 'precision': precision, 'recall': recall
                    }))
                    
                    print(f"       VotingWeighted: {accuracy:.4f}")
            except:
                pass
            
            all_candidates = []
            
            for name, model in successful_models:
                all_candidates.append((name, model, model_results[name]))
            
            for name, model, results in ensemble_methods:
                all_candidates.append((name, model, results))
            
            best_candidate = max(
                all_candidates, 
                key=lambda x: x[2]['accuracy'] * 0.7 + x[2]['f1_score'] * 0.3
            )
            
            best_name, best_model, best_results = best_candidate
            
            print(f"       Best: {best_name}")
            
            return best_model, best_name, best_results
            
        except Exception as e:
            print(f"       Ensemble error: {e}")
            if successful_models:
                best_name = max(model_results.keys(), key=lambda x: model_results[x]['accuracy'])
                best_model = next(model for name, model in successful_models if name == best_name)
                best_results = model_results[best_name]
                return best_model, best_name, best_results
            else:
                return None, None, None
    
    def analyze_errors(self, model, X_test, y_test, label_encoder, speaker_name):
        try:
            y_pred = model.predict(X_test)
            
            cm = confusion_matrix(y_test, y_pred)
            
            errors = []
            for i, (true_label, pred_label) in enumerate(zip(y_test, y_pred)):
                if true_label != pred_label:
                    true_word = label_encoder.inverse_transform([true_label])[0]
                    pred_word = label_encoder.inverse_transform([pred_label])[0]
                    errors.append((true_word, pred_word))
            
            if errors:
                error_counts = Counter(errors)
                most_common_errors = error_counts.most_common(3)
                
                print(f"     Error analysis:")
                for (true_word, pred_word), count in most_common_errors:
                    print(f"       '{true_word}' → '{pred_word}': {count} times")
            
        except Exception as e:
            pass
