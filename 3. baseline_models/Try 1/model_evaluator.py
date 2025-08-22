import numpy as np
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from config import Config

class ModelEvaluator:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def evaluate_random_model(self, labels):
        unique_labels = list(set(labels))
        random_predictions = np.random.choice(unique_labels, len(labels))
        accuracy = accuracy_score(labels, random_predictions)
        print(f"Random model accuracy: {accuracy:.4f}")
        return accuracy
    
    def evaluate_majority_model(self, labels):
        most_common = Counter(labels).most_common(1)[0][0]
        majority_predictions = [most_common] * len(labels)
        accuracy = accuracy_score(labels, majority_predictions)
        print(f"Majority model accuracy: {accuracy:.4f}")
        return accuracy
    
    def evaluate_knn_model(self, features, labels, k=None):
        if k is None:
            k = Config.KNN_NEIGHBORS
            
        features_scaled = self.scaler.fit_transform(features)
        
        knn = KNeighborsClassifier(n_neighbors=k)
        
        scores = cross_val_score(knn, features_scaled, labels, cv=Config.CV_FOLDS)
        accuracy = np.mean(scores)
        
        print(f"k-NN model accuracy: {accuracy:.4f}")
        return accuracy
    
    def evaluate_baseline_models(self, features, labels):
        print("Evaluating baseline models...")
        
        results = {}
        
        results['random'] = self.evaluate_random_model(labels)
        results['majority'] = self.evaluate_majority_model(labels)
        results['knn'] = self.evaluate_knn_model(features, labels)
        
        return results