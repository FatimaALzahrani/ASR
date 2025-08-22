import numpy as np
import pickle
from datetime import datetime
from sklearn.calibration import LabelEncoder
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from base_model import BaseModel
from data_processor import DataProcessor

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


class EnsembleModel(BaseModel):
    def __init__(self, features_path="features", results_path="model_results"):
        super().__init__(features_path, results_path)
        self.data_processor = DataProcessor()
        self.models = {}
        self.speaker_models = {}
        self.speaker_results = {}
        
        print("Ensemble Model initialized")
    
    def prepare_data(self, target_column='word', test_size=0.2, random_state=42):
        print("Preparing data...")
        
        X, y_encoded, valid_mask = self.data_processor.prepare_features(self.df, target_column)
        
        X_train, X_test, y_train, y_test = self.data_processor.split_data(X, y_encoded, test_size, random_state)
        X_train_scaled, X_test_scaled = self.data_processor.scale_features(X_train, X_test)
        
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
        self.X_train_scaled, self.X_test_scaled = X_train_scaled, X_test_scaled
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_traditional_models(self):
        print("Training traditional models...")
        
        traditional_models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100, random_state=42, n_jobs=-1
            ),
            'SVM': SVC(
                kernel='rbf', random_state=42, probability=True
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100, random_state=42
            ),
            'K-Nearest Neighbors': KNeighborsClassifier(
                n_neighbors=5
            ),
            'Naive Bayes': GaussianNB()
        }
        
        for name, model in traditional_models.items():
            print(f"Training {name}...")
            
            try:
                model.fit(self.X_train_scaled, self.y_train)
                
                y_pred = model.predict(self.X_test_scaled)
                y_pred_proba = model.predict_proba(self.X_test_scaled) if hasattr(model, 'predict_proba') else None
                
                accuracy = accuracy_score(self.y_test, y_pred)
                f1 = f1_score(self.y_test, y_pred, average='weighted')
                
                cv_scores = cross_val_score(
                    model, self.X_train_scaled, self.y_train, 
                    cv=5, scoring='accuracy'
                )
                
                self.models[name] = model
                self.results[name] = {
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                print(f"  Test accuracy: {accuracy:.4f}")
                print(f"  F1-Score: {f1:.4f}")
                print(f"  Cross-validation: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                
            except Exception as e:
                print(f"  Error training {name}: {str(e)}")
    
    def train_advanced_models(self):
        print("Training advanced models...")
        
        if XGBOOST_AVAILABLE:
            print("Training XGBoost...")
            try:
                xgb_model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    eval_metric='mlogloss'
                )
                
                xgb_model.fit(self.X_train_scaled, self.y_train)
                y_pred = xgb_model.predict(self.X_test_scaled)
                y_pred_proba = xgb_model.predict_proba(self.X_test_scaled)
                
                accuracy = accuracy_score(self.y_test, y_pred)
                f1 = f1_score(self.y_test, y_pred, average='weighted')
                
                self.models['XGBoost'] = xgb_model
                self.results['XGBoost'] = {
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                print(f"  Test accuracy: {accuracy:.4f}")
                print(f"  F1-Score: {f1:.4f}")
                
            except Exception as e:
                print(f"  Error training XGBoost: {str(e)}")
        
        if LIGHTGBM_AVAILABLE:
            print("Training LightGBM...")
            try:
                lgb_model = lgb.LGBMClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    verbose=-1
                )
                
                lgb_model.fit(self.X_train_scaled, self.y_train)
                y_pred = lgb_model.predict(self.X_test_scaled)
                y_pred_proba = lgb_model.predict_proba(self.X_test_scaled)
                
                accuracy = accuracy_score(self.y_test, y_pred)
                f1 = f1_score(self.y_test, y_pred, average='weighted')
                
                self.models['LightGBM'] = lgb_model
                self.results['LightGBM'] = {
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                print(f"  Test accuracy: {accuracy:.4f}")
                print(f"  F1-Score: {f1:.4f}")
                
            except Exception as e:
                print(f"  Error training LightGBM: {str(e)}")
        
        if CATBOOST_AVAILABLE:
            print("Training CatBoost...")
            try:
                cat_model = CatBoostClassifier(
                    iterations=100,
                    depth=6,
                    learning_rate=0.1,
                    random_seed=42,
                    verbose=False
                )
                
                cat_model.fit(self.X_train_scaled, self.y_train)
                y_pred = cat_model.predict(self.X_test_scaled)
                y_pred_proba = cat_model.predict_proba(self.X_test_scaled)
                
                accuracy = accuracy_score(self.y_test, y_pred)
                f1 = f1_score(self.y_test, y_pred, average='weighted')
                
                self.models['CatBoost'] = cat_model
                self.results['CatBoost'] = {
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                print(f"  Test accuracy: {accuracy:.4f}")
                print(f"  F1-Score: {f1:.4f}")
                
            except Exception as e:
                print(f"  Error training CatBoost: {str(e)}")
    
    def train_speaker_specific_models(self):
        print("Training speaker-specific models...")
        
        speakers = self.df['speaker'].unique()
        
        for speaker in speakers:
            print(f"Training model for speaker: {speaker}")
            
            speaker_data = self.data_processor.get_speaker_data(self.df, speaker)
            
            if len(speaker_data) < 10:
                print(f"  Insufficient data for speaker {speaker} ({len(speaker_data)} samples)")
                continue
            
            feature_columns = [col for col in speaker_data.columns 
                              if col not in ['file_path', 'word', 'speaker']]
            
            X_speaker = speaker_data[feature_columns].fillna(speaker_data[feature_columns].mean())
            y_speaker = speaker_data['word']
            
            le_speaker = LabelEncoder()
            y_speaker_encoded = le_speaker.fit_transform(y_speaker)
            
            if len(np.unique(y_speaker_encoded)) > 1 and len(speaker_data) >= 20:
                X_train_sp, X_test_sp, y_train_sp, y_test_sp = train_test_split(
                    X_speaker, y_speaker_encoded, test_size=0.3, 
                    random_state=42, stratify=y_speaker_encoded
                )
                
                scaler_sp = StandardScaler()
                X_train_sp_scaled = scaler_sp.fit_transform(X_train_sp)
                X_test_sp_scaled = scaler_sp.transform(X_test_sp)
                
                rf_speaker = RandomForestClassifier(
                    n_estimators=50, random_state=42, n_jobs=-1
                )
                rf_speaker.fit(X_train_sp_scaled, y_train_sp)
                
                y_pred_sp = rf_speaker.predict(X_test_sp_scaled)
                accuracy_sp = accuracy_score(y_test_sp, y_pred_sp)
                f1_sp = f1_score(y_test_sp, y_pred_sp, average='weighted')
                
                self.speaker_models[speaker] = {
                    'model': rf_speaker,
                    'scaler': scaler_sp,
                    'label_encoder': le_speaker
                }
                
                self.speaker_results[speaker] = {
                    'accuracy': accuracy_sp,
                    'f1_score': f1_sp,
                    'n_samples': len(speaker_data),
                    'n_words': len(np.unique(y_speaker_encoded))
                }
                
                print(f"  Model accuracy: {accuracy_sp:.4f}")
                print(f"  F1-Score: {f1_sp:.4f}")
                print(f"  Samples: {len(speaker_data)}")
                print(f"  Words: {len(np.unique(y_speaker_encoded))}")
            
            else:
                print(f"  Insufficient data for training and testing")
    
    def create_ensemble_model(self):
        print("Creating ensemble model...")
        
        if not self.results:
            print("No trained models for ensemble")
            return
        
        sorted_models = sorted(
            self.results.items(), 
            key=lambda x: x[1]['accuracy'], 
            reverse=True
        )
        
        top_models = sorted_models[:3]
        print("Top 3 models for ensemble:")
        for i, (name, results) in enumerate(top_models, 1):
            print(f"  {i}. {name}: {results['accuracy']:.4f}")
        
        ensemble_probs = np.zeros((len(self.y_test), len(np.unique(self.y_test))))
        
        for name, _ in top_models:
            if 'probabilities' in self.results[name] and self.results[name]['probabilities'] is not None:
                ensemble_probs += self.results[name]['probabilities']
        
        ensemble_probs /= len(top_models)
        ensemble_predictions = np.argmax(ensemble_probs, axis=1)
        
        ensemble_accuracy = accuracy_score(self.y_test, ensemble_predictions)
        ensemble_f1 = f1_score(self.y_test, ensemble_predictions, average='weighted')
        
        self.results['Ensemble'] = {
            'accuracy': ensemble_accuracy,
            'f1_score': ensemble_f1,
            'predictions': ensemble_predictions,
            'probabilities': ensemble_probs,
            'component_models': [name for name, _ in top_models]
        }
        
        print(f"Ensemble accuracy: {ensemble_accuracy:.4f}")
        print(f"Ensemble F1-Score: {ensemble_f1:.4f}")
    
    def train(self):
        print("Starting ensemble model training...")
        
        try:
            self.df = self.data_processor.load_features(self.features_path)
            self.prepare_data()
            
            self.train_traditional_models()
            self.train_advanced_models()
            self.train_speaker_specific_models()
            self.create_ensemble_model()
            
            self.is_trained = True
            print("Ensemble model training completed")
            return True
            
        except Exception as e:
            print(f"Training error: {str(e)}")
            return False
    
    def predict(self, X):
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        if 'Ensemble' in self.results:
            best_model_name = self.results['Ensemble']['component_models'][0]
            best_model = self.models[best_model_name]
            return best_model.predict(X)
        else:
            best_model_name = max(self.results.items(), key=lambda x: x[1]['accuracy'])[0]
            best_model = self.models[best_model_name]
            return best_model.predict(X)
    
    def evaluate(self):
        if 'Ensemble' in self.results:
            return {
                'accuracy': self.results['Ensemble']['accuracy'],
                'f1_score': self.results['Ensemble']['f1_score']
            }
        elif self.results:
            best_result = max(self.results.items(), key=lambda x: x[1]['accuracy'])[1]
            return {
                'accuracy': best_result['accuracy'],
                'f1_score': best_result['f1_score']
            }
        return None
    
    def save_model(self, filename=None):
        if not self.is_trained:
            print("Model not trained yet")
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for name, model in self.models.items():
            model_file = f"{self.results_path}/models/{name}_{timestamp}.pkl"
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
            print(f"Saved {name}")
        
        if hasattr(self, 'speaker_models'):
            speaker_models_file = f"{self.results_path}/models/speaker_models_{timestamp}.pkl"
            with open(speaker_models_file, 'wb') as f:
                pickle.dump(self.speaker_models, f)
            print("Saved speaker-specific models")
        
        utils_file = f"{self.results_path}/models/utils_{timestamp}.pkl"
        utils_data = {
            'scaler': self.data_processor.scaler,
            'label_encoder': self.data_processor.label_encoder
        }
        with open(utils_file, 'wb') as f:
            pickle.dump(utils_data, f)
        print("Saved utilities")
        
        return f"{self.results_path}/models/"