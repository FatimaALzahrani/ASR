import numpy as np
import pickle
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
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


class AdvancedAcousticModel(BaseModel):
    def __init__(self, features_path="features", results_path="advanced_results"):
        super().__init__(features_path, results_path)
        self.data_processor = DataProcessor()
        self.ensemble_model = None
        self.confidence_threshold = 0.5
        
        print("Advanced Acoustic Model initialized")
    
    def load_and_analyze_features(self):
        print("Loading features...")
        self.df = self.data_processor.load_features(self.features_path)
        
        distribution = self.data_processor.get_data_distribution(self.df)
        self.results['data_distribution'] = distribution
        
        print(f"Words: {distribution['unique_words']}")
        print(f"Speakers: {distribution['unique_speakers']}")
        
        return self.df
    
    def advanced_feature_engineering(self):
        print("Advanced feature engineering...")
        
        X, y, valid_mask = self.data_processor.prepare_features(self.df)
        
        X = self.data_processor.create_composite_features(X)
        
        X_selected, y_filtered, _ = self.data_processor.select_features(X, y)
        
        return X_selected, y_filtered, valid_mask
    
    def build_ensemble_model(self, X, y):
        print("Building ensemble acoustic model...")
        
        X_train, X_test, y_train, y_test = self.data_processor.split_data(X, y)
        X_train_scaled, X_test_scaled = self.data_processor.scale_features(X_train, X_test)
        
        print(f"Training: {len(X_train)} | Testing: {len(X_test)}")
        print(f"Classes: {len(np.unique(y))}")
        
        base_models = {
            'rf_1': RandomForestClassifier(
                n_estimators=100, max_depth=15, min_samples_split=5,
                random_state=42, n_jobs=-1
            ),
            'rf_2': RandomForestClassifier(
                n_estimators=150, max_depth=20, min_samples_split=3,
                random_state=123, n_jobs=-1
            ),
            'mlp': MLPClassifier(
                hidden_layer_sizes=(128, 64), activation='relu',
                solver='adam', alpha=0.001, learning_rate='adaptive',
                max_iter=500, random_state=42
            )
        }
        
        if XGBOOST_AVAILABLE:
            base_models['xgb'] = xgb.XGBClassifier(
                n_estimators=100, max_depth=8, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, random_state=42,
                eval_metric='mlogloss'
            )
        
        if LIGHTGBM_AVAILABLE:
            base_models['lgb'] = lgb.LGBMClassifier(
                n_estimators=100, max_depth=8, learning_rate=0.1,
                feature_fraction=0.8, bagging_fraction=0.8,
                random_state=42, verbose=-1
            )
        
        trained_models = {}
        individual_results = {}
        
        for name, model in base_models.items():
            print(f"Training {name}...")
            try:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)
                
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                trained_models[name] = model
                individual_results[name] = {
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                print(f"  Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
                
            except Exception as e:
                print(f"  Error training {name}: {str(e)}")
        
        if len(trained_models) >= 2:
            print("Creating ensemble model...")
            
            sorted_models = sorted(
                individual_results.items(), 
                key=lambda x: x[1]['accuracy'], 
                reverse=True
            )[:3]
            
            ensemble_models = [(name, trained_models[name]) for name, _ in sorted_models]
            
            self.ensemble_model = VotingClassifier(
                estimators=ensemble_models, voting='soft'
            )
            
            self.ensemble_model.fit(X_train_scaled, y_train)
            
            y_pred_ensemble = self.ensemble_model.predict(X_test_scaled)
            y_pred_proba_ensemble = self.ensemble_model.predict_proba(X_test_scaled)
            
            ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
            ensemble_f1 = f1_score(y_test, y_pred_ensemble, average='weighted')
            
            print(f"Ensemble accuracy: {ensemble_accuracy:.4f}")
            print(f"Ensemble F1: {ensemble_f1:.4f}")
            
            self.model = self.ensemble_model
            self.results['ensemble_accuracy'] = ensemble_accuracy
            self.results['ensemble_f1'] = ensemble_f1
            self.results['individual_results'] = individual_results
            self.results['best_models'] = [name for name, _ in sorted_models]
            
            self.calculate_confidence_thresholds(y_pred_proba_ensemble, y_test)
            
        else:
            print("Insufficient models for ensemble")
            return None
        
        return self.ensemble_model
    
    def calculate_confidence_thresholds(self, probabilities, true_labels):
        print("Calculating confidence thresholds...")
        
        max_probs = np.max(probabilities, axis=1)
        predictions = np.argmax(probabilities, axis=1)
        correct_mask = predictions == true_labels
        
        correct_confidences = max_probs[correct_mask]
        incorrect_confidences = max_probs[~correct_mask]
        
        if len(correct_confidences) > 0 and len(incorrect_confidences) > 0:
            thresholds = np.linspace(0.1, 0.9, 50)
            best_threshold = 0.5
            best_f1 = 0
            
            for threshold in thresholds:
                high_conf_mask = max_probs >= threshold
                if high_conf_mask.sum() > 0:
                    high_conf_preds = predictions[high_conf_mask]
                    high_conf_true = true_labels[high_conf_mask]
                    
                    if len(np.unique(high_conf_true)) > 1:
                        f1 = f1_score(high_conf_true, high_conf_preds, average='weighted')
                        if f1 > best_f1:
                            best_f1 = f1
                            best_threshold = threshold
            
            self.confidence_threshold = best_threshold
            print(f"  Optimal threshold: {best_threshold:.3f}")
        else:
            self.confidence_threshold = 0.5
            print(f"  Default threshold: 0.5")
    
    def predict_with_confidence(self, X):
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        if hasattr(self.data_processor, 'selected_features'):
            X_selected = X[self.data_processor.selected_features]
        else:
            X_selected = X
        
        X_scaled = self.data_processor.scaler.transform(X_selected)
        
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        max_probs = np.max(probabilities, axis=1)
        high_confidence_mask = max_probs >= self.confidence_threshold
        
        predicted_words = self.data_processor.label_encoder.inverse_transform(predictions)
        
        results = {
            'predictions': predicted_words,
            'probabilities': probabilities,
            'confidence_scores': max_probs,
            'high_confidence_mask': high_confidence_mask,
            'low_confidence_indices': np.where(~high_confidence_mask)[0].tolist()
        }
        
        return results
    
    def train(self):
        print("Starting acoustic model training...")
        
        try:
            self.load_and_analyze_features()
            X, y, valid_mask = self.advanced_feature_engineering()
            model = self.build_ensemble_model(X, y)
            
            if model is not None:
                self.is_trained = True
                print("Acoustic model training completed")
                return True
            else:
                print("Failed to build ensemble model")
                return False
                
        except Exception as e:
            print(f"Training error: {str(e)}")
            return False
    
    def predict(self, X):
        return self.predict_with_confidence(X)
    
    def evaluate(self):
        if 'ensemble_accuracy' in self.results:
            return {
                'accuracy': self.results['ensemble_accuracy'],
                'f1_score': self.results['ensemble_f1']
            }
        return None
    
    def save_model(self, filename=None):
        if not self.is_trained:
            print("Model not trained yet")
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if filename is None:
            filename = f"acoustic_model_{timestamp}.pkl"
        
        model_data = {
            'acoustic_model': self.model,
            'scaler': self.data_processor.scaler,
            'label_encoder': self.data_processor.label_encoder,
            'feature_selector': self.data_processor.feature_selector,
            'selected_features': self.data_processor.selected_features,
            'confidence_threshold': self.confidence_threshold,
            'results': self.results
        }
        
        filepath = f"{self.results_path}/acoustic_models/{filename}"
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Acoustic model saved: {filepath}")
        return filepath