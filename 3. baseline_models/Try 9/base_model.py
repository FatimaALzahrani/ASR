import os
import pickle
import json
from datetime import datetime
from abc import ABC, abstractmethod


class BaseModel(ABC):
    def __init__(self, features_path="features", results_path="results"):
        self.features_path = features_path
        self.results_path = results_path
        self.model = None
        self.results = {}
        self.is_trained = False
        
        os.makedirs(results_path, exist_ok=True)
        os.makedirs(f"{results_path}/models", exist_ok=True)
        os.makedirs(f"{results_path}/reports", exist_ok=True)
    
    @abstractmethod
    def train(self):
        pass
    
    @abstractmethod
    def predict(self, X):
        pass
    
    @abstractmethod
    def evaluate(self):
        pass
    
    def save_model(self, filename=None):
        if not self.is_trained:
            print("Model not trained yet")
            return None
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.__class__.__name__.lower()}_{timestamp}.pkl"
        
        filepath = os.path.join(self.results_path, "models", filename)
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        
        print(f"Model saved: {filepath}")
        return filepath
    
    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        self.is_trained = True
        print(f"Model loaded: {filepath}")
    
    def save_results(self, filename=None):
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.__class__.__name__.lower()}_results_{timestamp}.json"
        
        filepath = os.path.join(self.results_path, "reports", filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        print(f"Results saved: {filepath}")
        return filepath