import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score


class ModelTrainer:
    def __init__(self, random_state=42):
        self.random_state = random_state
    
    def prepare_features_for_ml(self, features_df, target_column='word'):
        feature_columns = [col for col in features_df.columns 
                          if col not in ['file_path', 'word', 'speaker', 'filename']]
        
        X = features_df[feature_columns].copy()
        y = features_df[target_column]
        
        X = X.fillna(0.0)
        X = X.replace([np.inf, -np.inf], 0.0)
        
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0.0)
        
        print(f"Prepared {X.shape[1]} features for ML")
        print(f"Feature shape: {X.shape}")
        
        return X, y, feature_columns
    
    def train_traditional_ml_models(self, train_df, test_df):
        print("Training traditional ML models")
        
        X_train, y_train, feature_columns = self.prepare_features_for_ml(train_df)
        X_test, y_test, _ = self.prepare_features_for_ml(test_df)
        
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100, max_depth=10, min_samples_split=5,
                min_samples_leaf=2, random_state=self.random_state, n_jobs=-1
            ),
            'SVM (RBF)': SVC(
                C=1.0, gamma='scale', random_state=self.random_state
            ),
            'Logistic Regression': LogisticRegression(
                C=1.0, max_iter=1000, random_state=self.random_state, n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                random_state=self.random_state
            ),
            'Decision Tree': DecisionTreeClassifier(
                max_depth=8, min_samples_split=5, min_samples_leaf=2,
                random_state=self.random_state
            ),
            'K-Nearest Neighbors': KNeighborsClassifier(
                n_neighbors=5, weights='distance', n_jobs=-1
            )
        }
        
        ml_results = {}
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        for name, model in models.items():
            print(f"Training {name}")
            
            try:
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
                
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                test_accuracy = accuracy_score(y_test, y_pred)
                
                class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                
                ml_results[name] = {
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'test_accuracy': test_accuracy,
                    'overfitting': cv_scores.mean() - test_accuracy,
                    'classification_report': class_report
                }
                
                print(f"{name} - CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}, Test: {test_accuracy:.4f}")
                
            except Exception as e:
                print(f"Error training {name}: {str(e)}")
                ml_results[name] = {'error': str(e)}
        
        return ml_results
    
    def create_summary_table(self, results):
        summary_data = []
        
        if 'traditional_ml' in results:
            for name, results_data in results['traditional_ml'].items():
                if 'test_accuracy' in results_data:
                    summary_data.append({
                        'Model': name,
                        'Type': 'Traditional ML',
                        'Cross-Validation': f"{results_data['cv_mean']:.3f} ± {results_data['cv_std']:.3f}",
                        'Test Accuracy': f"{results_data['test_accuracy']:.3f}",
                        'Overfitting': f"{results_data['overfitting']:+.3f}"
                    })
        
        return pd.DataFrame(summary_data)
