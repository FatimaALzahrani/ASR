import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
from config import MODEL_PARAMS, SPEAKER_MODEL_PARAMS

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.speaker_models = {}
    
    def train_general_models(self, X, y):
        print("Training general models...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        results = {}
        
        models = [
            ('RandomForest', RandomForestClassifier(**MODEL_PARAMS['random_forest'])),
            ('ExtraTrees', ExtraTreesClassifier(**MODEL_PARAMS['extra_trees'])),
            ('XGBoost', xgb.XGBClassifier(**MODEL_PARAMS['xgboost'])),
            ('SVM', SVC(**MODEL_PARAMS['svm'])),
            ('LightGBM', lgb.LGBMClassifier(**MODEL_PARAMS['lightgbm'])),
            ('MLP', MLPClassifier(**MODEL_PARAMS['mlp']))
        ]
        
        trained_models = []
        for name, model in models:
            try:
                print(f"Training {name}...")
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                results[name] = accuracy
                trained_models.append((name, model))
                self.models[name] = model
                print(f"{name}: {accuracy*100:.2f}%")
            except Exception as e:
                print(f"{name} error: {e}")
                results[name] = 0
        
        if len(trained_models) >= 3:
            try:
                print("Training Voting Ensemble...")
                voting_ensemble = VotingClassifier(
                    estimators=trained_models,
                    voting='soft',
                    n_jobs=-1
                )
                voting_ensemble.fit(X_train, y_train)
                y_pred = voting_ensemble.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                results['VotingEnsemble'] = accuracy
                self.models['VotingEnsemble'] = voting_ensemble
                print(f"Voting Ensemble: {accuracy*100:.2f}%")
            except Exception as e:
                print(f"Voting Ensemble error: {e}")
                results['VotingEnsemble'] = 0
            
            try:
                print("Training Stacking Ensemble...")
                stacking_ensemble = StackingClassifier(
                    estimators=trained_models[:4],
                    final_estimator=LogisticRegression(random_state=42, max_iter=1000),
                    cv=5,
                    n_jobs=-1
                )
                stacking_ensemble.fit(X_train, y_train)
                y_pred = stacking_ensemble.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                results['StackingEnsemble'] = accuracy
                self.models['StackingEnsemble'] = stacking_ensemble
                print(f"Stacking Ensemble: {accuracy*100:.2f}%")
            except Exception as e:
                print(f"Stacking Ensemble error: {e}")
                results['StackingEnsemble'] = 0
        
        return results, X_test, y_test
    
    def train_speaker_models(self, X, y, speakers):
        print("Training speaker-specific models...")
        
        speaker_results = {}
        unique_speakers = [s for s in np.unique(speakers) if s != "Unknown"]
        
        for speaker in unique_speakers:
            print(f"Training model for {speaker}...")
            
            speaker_mask = speakers == speaker
            speaker_X = X[speaker_mask]
            speaker_y = y[speaker_mask]
            
            if len(speaker_X) < 15:
                print(f"Insufficient data for {speaker}: {len(speaker_X)} samples")
                speaker_results[speaker] = 0
                continue
            
            try:
                speaker_models = [
                    ('ExtraTrees', ExtraTreesClassifier(**SPEAKER_MODEL_PARAMS['extra_trees'])),
                    ('RandomForest', RandomForestClassifier(**SPEAKER_MODEL_PARAMS['random_forest'])),
                    ('XGBoost', xgb.XGBClassifier(**SPEAKER_MODEL_PARAMS['xgboost'])),
                    ('SVM', SVC(**SPEAKER_MODEL_PARAMS['svm']))
                ]
                
                best_accuracy = 0
                best_model = None
                best_name = ""
                
                for name, model in speaker_models:
                    try:
                        cv_scores = cross_val_score(
                            model, speaker_X, speaker_y, 
                            cv=min(5, len(np.unique(speaker_y))), 
                            scoring='accuracy'
                        )
                        avg_accuracy = np.mean(cv_scores)
                        
                        if avg_accuracy > best_accuracy:
                            best_accuracy = avg_accuracy
                            best_model = model
                            best_name = name
                            
                    except Exception as e:
                        print(f"Error with {name}: {e}")
                        continue
                
                if best_model is not None:
                    best_model.fit(speaker_X, speaker_y)
                    speaker_results[speaker] = best_accuracy
                    self.speaker_models[speaker] = best_model
                    print(f"{speaker} ({best_name}): {best_accuracy*100:.2f}%")
                else:
                    speaker_results[speaker] = 0
                    print(f"{speaker}: No successful model")
                
            except Exception as e:
                print(f"Error training {speaker} model: {e}")
                speaker_results[speaker] = 0
        
        return speaker_results
    
    def get_models(self):
        return self.models, self.speaker_models
