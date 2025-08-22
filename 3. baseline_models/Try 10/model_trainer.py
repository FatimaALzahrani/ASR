import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import (RandomForestClassifier, VotingClassifier,
                             ExtraTreesClassifier, GradientBoostingClassifier,
                             StackingClassifier)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb


class ModelTrainer:
    def __init__(self):
        self.models = {}
    
    def train_ultimate_models(self, X, y, speakers):
        print("Training models...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        results = {}
        
        print("\n1. Training Random Forest...")
        try:
            rf_optimized = RandomForestClassifier(
                n_estimators=1500,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='sqrt',
                bootstrap=True,
                class_weight='balanced_subsample',
                random_state=42,
                n_jobs=-1
            )
            
            rf_optimized.fit(X_train, y_train)
            y_pred = rf_optimized.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            results['RandomForest_Ultimate'] = accuracy
            self.models['RandomForest_Ultimate'] = rf_optimized
            print(f"Random Forest accuracy: {accuracy:.4f}")
            
        except Exception as e:
            print(f"Error training Random Forest: {e}")
            results['RandomForest_Ultimate'] = 0
        
        print("\n2. Training XGBoost...")
        try:
            class_weights = {}
            unique_classes, class_counts = np.unique(y_train, return_counts=True)
            total_samples = len(y_train)
            
            for cls, count in zip(unique_classes, class_counts):
                class_weights[cls] = total_samples / (len(unique_classes) * count)
            
            sample_weights = np.array([class_weights[cls] for cls in y_train])
            
            xgb_ultimate = xgb.XGBClassifier(
                n_estimators=800,
                max_depth=10,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                gamma=0.1,
                min_child_weight=3,
                random_state=42,
                eval_metric='mlogloss',
                verbosity=0
            )
            
            xgb_ultimate.fit(X_train, y_train, sample_weight=sample_weights)
            y_pred = xgb_ultimate.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            results['XGBoost_Ultimate'] = accuracy
            self.models['XGBoost_Ultimate'] = xgb_ultimate
            print(f"XGBoost accuracy: {accuracy:.4f}")
            
        except Exception as e:
            print(f"Error training XGBoost: {e}")
            results['XGBoost_Ultimate'] = 0
        
        print("\n3. Training MLP...")
        try:
            mlp_ultimate = MLPClassifier(
                hidden_layer_sizes=(1024, 512, 256, 128),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                batch_size=64,
                learning_rate='adaptive',
                learning_rate_init=0.001,
                max_iter=2000,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=30,
                random_state=42
            )
            
            mlp_ultimate.fit(X_train, y_train)
            y_pred = mlp_ultimate.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            results['MLP_Ultimate'] = accuracy
            self.models['MLP_Ultimate'] = mlp_ultimate
            print(f"MLP accuracy: {accuracy:.4f}")
            
        except Exception as e:
            print(f"Error training MLP: {e}")
            results['MLP_Ultimate'] = 0
        
        print("\n4. Training SVM...")
        try:
            svm_ultimate = SVC(
                kernel='rbf',
                C=100,
                gamma='scale',
                class_weight='balanced',
                probability=True,
                random_state=42
            )
            
            svm_ultimate.fit(X_train, y_train)
            y_pred = svm_ultimate.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            results['SVM_Ultimate'] = accuracy
            self.models['SVM_Ultimate'] = svm_ultimate
            print(f"SVM accuracy: {accuracy:.4f}")
            
        except Exception as e:
            print(f"Error training SVM: {e}")
            results['SVM_Ultimate'] = 0
        
        print("\n5. Training Extra Trees...")
        try:
            et_ultimate = ExtraTreesClassifier(
                n_estimators=1200,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='sqrt',
                bootstrap=True,
                class_weight='balanced_subsample',
                random_state=42,
                n_jobs=-1
            )
            
            et_ultimate.fit(X_train, y_train)
            y_pred = et_ultimate.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            results['ExtraTrees_Ultimate'] = accuracy
            self.models['ExtraTrees_Ultimate'] = et_ultimate
            print(f"Extra Trees accuracy: {accuracy:.4f}")
            
        except Exception as e:
            print(f"Error training Extra Trees: {e}")
            results['ExtraTrees_Ultimate'] = 0
        
        print("\n6. Training LightGBM...")
        try:
            lgb_ultimate = lgb.LGBMClassifier(
                n_estimators=800,
                max_depth=12,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                class_weight='balanced',
                random_state=42,
                verbose=-1
            )
            
            lgb_ultimate.fit(X_train, y_train)
            y_pred = lgb_ultimate.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            results['LightGBM_Ultimate'] = accuracy
            self.models['LightGBM_Ultimate'] = lgb_ultimate
            print(f"LightGBM accuracy: {accuracy:.4f}")
            
        except Exception as e:
            print(f"Error training LightGBM: {e}")
            results['LightGBM_Ultimate'] = 0
        
        print("\n7. Training Gradient Boosting...")
        try:
            gb_ultimate = GradientBoostingClassifier(
                n_estimators=500,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                max_features='sqrt',
                random_state=42
            )
            
            gb_ultimate.fit(X_train, y_train)
            y_pred = gb_ultimate.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            results['GradientBoosting_Ultimate'] = accuracy
            self.models['GradientBoosting_Ultimate'] = gb_ultimate
            print(f"Gradient Boosting accuracy: {accuracy:.4f}")
            
        except Exception as e:
            print(f"Error training Gradient Boosting: {e}")
            results['GradientBoosting_Ultimate'] = 0
        
        print("\n8. Creating Ensemble...")
        try:
            valid_models = [(name, model) for name, model in self.models.items() 
                           if results.get(name, 0) > 0.1]
            
            if len(valid_models) >= 3:
                voting_ensemble = VotingClassifier(
                    estimators=valid_models,
                    voting='soft'
                )
                
                voting_ensemble.fit(X_train, y_train)
                y_pred_voting = voting_ensemble.predict(X_test)
                voting_accuracy = accuracy_score(y_test, y_pred_voting)
                
                results['Voting_Ensemble'] = voting_accuracy
                self.models['Voting_Ensemble'] = voting_ensemble
                print(f"Voting Ensemble accuracy: {voting_accuracy:.4f}")
                
                try:
                    stacking_ensemble = StackingClassifier(
                        estimators=valid_models[:5],
                        final_estimator=LogisticRegression(max_iter=1000, class_weight='balanced'),
                        cv=3
                    )
                    
                    stacking_ensemble.fit(X_train, y_train)
                    y_pred_stacking = stacking_ensemble.predict(X_test)
                    stacking_accuracy = accuracy_score(y_test, y_pred_stacking)
                    
                    results['Stacking_Ensemble'] = stacking_accuracy
                    self.models['Stacking_Ensemble'] = stacking_ensemble
                    print(f"Stacking Ensemble accuracy: {stacking_accuracy:.4f}")
                    
                except Exception as e:
                    print(f"Stacking ensemble error: {e}")
                    results['Stacking_Ensemble'] = 0
            else:
                print("Not enough valid models for ensemble")
                results['Voting_Ensemble'] = 0
                results['Stacking_Ensemble'] = 0
                
        except Exception as e:
            print(f"Error creating ensemble: {e}")
            results['Voting_Ensemble'] = 0
            results['Stacking_Ensemble'] = 0
        
        return results