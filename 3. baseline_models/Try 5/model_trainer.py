import os
import pickle
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif

class ModelTrainer:
    def __init__(self):
        self.models = {
            'rf': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'gb': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'svm': SVC(kernel='rbf', random_state=42, probability=True)
        }

    def train_model(self, X, y, model_name="enhanced_model"):
        print(f"Training enhanced model: {model_name}")

        word_counts = Counter(y)
        min_samples = min(word_counts.values())

        if min_samples >= 2:
            stratify = y
            print("Using stratified split")
        else:
            stratify = None
            print("Using regular split (some words have single sample)")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify
        )

        print(f"Data split:")
        print(f"   Training: {X_train.shape[0]} samples")
        print(f"   Testing: {X_test.shape[0]} samples")

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)

        feature_selector = SelectKBest(score_func=f_classif, k=min(80, X_train_scaled.shape[1]))
        X_train_selected = feature_selector.fit_transform(X_train_scaled, y_train_encoded)
        X_test_selected = feature_selector.transform(X_test_scaled)

        print(f"Selected {X_train_selected.shape[1]} features from {X_train_scaled.shape[1]}")

        results = {}
        trained_models = {}

        for model_type, model in self.models.items():
            print(f"  Training {model_type}...")

            try:
                model.fit(X_train_selected, y_train_encoded)
                y_pred = model.predict(X_test_selected)
                accuracy = accuracy_score(y_test_encoded, y_pred)

                print(f"    {model_type}: {accuracy:.4f} ({accuracy*100:.2f}%)")

                results[model_type] = {
                    'accuracy': accuracy,
                    'predictions': y_pred,
                    'true_labels': y_test_encoded
                }

                trained_models[model_type] = model

            except Exception as e:
                print(f"    Training error for {model_type}: {e}")
                results[model_type] = {'accuracy': 0}

        valid_results = {k: v for k, v in results.items() if v['accuracy'] > 0}

        if valid_results:
            best_model_type = max(valid_results.keys(), key=lambda k: valid_results[k]['accuracy'])
            best_accuracy = valid_results[best_model_type]['accuracy']
            print(f"Best model: {best_model_type} with accuracy {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")

            best_model = trained_models[best_model_type]

            os.makedirs("saved_models", exist_ok=True)
            model_path = os.path.join("saved_models", f"{model_name}_{best_model_type}.pkl")

            with open(model_path, "wb") as f:
                pickle.dump({
                    "model": best_model,
                    "scaler": scaler,
                    "label_encoder": label_encoder,
                    "feature_selector": feature_selector
                }, f)

            print(f"Saved best model and components to: {model_path}")

        else:
            best_accuracy = 0
            print("All models failed to train")

        return best_accuracy, results