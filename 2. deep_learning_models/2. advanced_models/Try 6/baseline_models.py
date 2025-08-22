import time
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from typing import Dict, Tuple

class BaselineModels:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.results = {}
    
    def create_traditional_models(self) -> Dict:
        return {
            'SVM_RBF': SVC(kernel='rbf', probability=True, random_state=self.random_state),
            'SVM_Linear': SVC(kernel='linear', probability=True, random_state=self.random_state),
            'Random_Forest': RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=-1),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'Naive_Bayes': GaussianNB(),
            'Logistic_Regression': LogisticRegression(random_state=self.random_state, max_iter=1000)
        }
    
    def evaluate_traditional_models(self, X_train: np.ndarray, X_test: np.ndarray, 
                                  y_train: np.ndarray, y_test: np.ndarray) -> Dict:
        results = {}
        models = self.create_traditional_models()
        
        for name, model in models.items():
            try:
                start_time = time.time()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                accuracy = accuracy_score(y_test, y_pred)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_test, y_pred, average='weighted', zero_division=0
                )
                training_time = time.time() - start_time
                
                results[name] = {
                    'accuracy': float(accuracy),
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1_score': float(f1),
                    'training_time': training_time,
                    'model_type': 'traditional'
                }
                
                print(f"Traditional Model {name}: Accuracy = {accuracy:.4f}")
                
            except Exception as e:
                results[name] = {
                    'accuracy': 0.0, 
                    'error': str(e), 
                    'model_type': 'traditional'
                }
                print(f"Error training {name}: {e}")
        
        return results
    
    def get_best_traditional_model(self, results: Dict) -> Tuple[str, float]:
        best_model = None
        best_accuracy = 0.0
        
        for model_name, result in results.items():
            if 'accuracy' in result and result['accuracy'] > best_accuracy:
                best_accuracy = result['accuracy']
                best_model = model_name
        
        return best_model, best_accuracy