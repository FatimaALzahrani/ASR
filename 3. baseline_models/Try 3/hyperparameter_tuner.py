from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


class HyperparameterTuner:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        
    def tune_hyperparameters(self):
        print("Tuning hyperparameters for promising models...")
        
        rf_param_grid = {
            'n_estimators': [10, 20, 30],
            'max_depth': [2, 3, 4],
            'min_samples_split': [15, 20, 25],
            'min_samples_leaf': [8, 10, 12]
        }
        
        rf_grid = GridSearchCV(
            RandomForestClassifier(random_state=42),
            rf_param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=-1
        )
        
        svm_param_grid = {
            'C': [0.001, 0.01, 0.1],
            'gamma': ['scale', 'auto', 0.001, 0.01]
        }
        
        svm_grid = GridSearchCV(
            SVC(random_state=42),
            svm_param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=-1
        )
        
        tuned_models = {}
        
        try:
            print("Tuning Random Forest...")
            rf_grid.fit(self.X_train, self.y_train)
            tuned_models['Random Forest (Tuned)'] = rf_grid.best_estimator_
            print(f"  Best params: {rf_grid.best_params_}")
            print(f"  Best CV score: {rf_grid.best_score_:.4f}")
        except Exception as e:
            print(f"Error tuning Random Forest: {e}")
        
        try:
            print("Tuning SVM...")
            svm_grid.fit(self.X_train, self.y_train)
            tuned_models['SVM (Tuned)'] = svm_grid.best_estimator_
            print(f"  Best params: {svm_grid.best_params_}")
            print(f"  Best CV score: {svm_grid.best_score_:.4f}")
        except Exception as e:
            print(f"Error tuning SVM: {e}")
        
        return tuned_models