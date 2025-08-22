from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import precision_recall_fscore_support


class ModelTrainer:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
    def train_and_evaluate_models(self, models):
        print("Training and evaluating models...")
        
        results = {}
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            try:
                cv_scores = cross_val_score(
                    model, self.X_train, self.y_train, 
                    cv=cv, scoring='accuracy', n_jobs=-1
                )
                
                model.fit(self.X_train, self.y_train)
                
                train_score = model.score(self.X_train, self.y_train)
                test_score = model.score(self.X_test, self.y_test)
                
                y_pred = model.predict(self.X_test)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    self.y_test, y_pred, average='weighted', zero_division=0
                )
                
                results[name] = {
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'train_accuracy': train_score,
                    'test_accuracy': test_score,
                    'overfitting_gap': train_score - test_score,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'cv_scores': cv_scores.tolist()
                }
                
                print(f"Results for {name}:")
                print(f"  - CV Mean: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
                print(f"  - Train Accuracy: {train_score:.4f}")
                print(f"  - Test Accuracy: {test_score:.4f}")
                print(f"  - Overfitting Gap: {(train_score - test_score):.4f}")
                print(f"  - F1-Score: {f1:.4f}")
                
                if train_score - test_score > 0.2:
                    print("  ⚠️ High overfitting")
                elif train_score - test_score > 0.1:
                    print("  ⚠️ Moderate overfitting")
                else:
                    print("  ✅ Low overfitting")
                    
            except Exception as e:
                print(f"Error in {name}: {e}")
                results[name] = None
        
        return results