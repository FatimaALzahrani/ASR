from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, VotingClassifier
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


class ModelFactory:
    @staticmethod
    def create_regularized_models():
        print("Creating regularized models...")
        
        models = {
            'Random Forest (Regularized)': RandomForestClassifier(
                n_estimators=30,
                max_depth=3,
                min_samples_split=20,
                min_samples_leaf=10,
                max_features='sqrt',
                bootstrap=True,
                oob_score=True,
                random_state=42
            ),
            
            'SVM (High Regularization)': SVC(
                C=0.01,
                kernel='rbf',
                gamma='scale',
                random_state=42
            ),
            
            'Logistic Regression (L2)': LogisticRegression(
                C=0.01,
                penalty='l2',
                max_iter=1000,
                random_state=42
            ),
            
            'Ridge Classifier': RidgeClassifier(
                alpha=10.0,
                random_state=42
            ),
            
            'Gradient Boosting (Regularized)': GradientBoostingClassifier(
                n_estimators=20,
                max_depth=2,
                learning_rate=0.05,
                subsample=0.8,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42
            ),
            
            'KNN (Optimized)': KNeighborsClassifier(
                n_neighbors=7,
                weights='distance',
                metric='manhattan'
            ),
            
            'Naive Bayes': GaussianNB(
                var_smoothing=1e-8
            ),
            
            'Extra Trees (Regularized)': ExtraTreesClassifier(
                n_estimators=20,
                max_depth=3,
                min_samples_split=20,
                min_samples_leaf=10,
                max_features='sqrt',
                bootstrap=True,
                random_state=42
            )
        }
        
        return models
    
    @staticmethod
    def create_ensemble_models(base_models):
        print("Creating ensemble models...")
        
        selected_models = [
            ('rf', base_models['Random Forest (Regularized)']),
            ('svm', base_models['SVM (High Regularization)']),
            ('lr', base_models['Logistic Regression (L2)']),
            ('nb', base_models['Naive Bayes'])
        ]
        
        ensemble_models = {
            'Voting Classifier (Hard)': VotingClassifier(
                estimators=selected_models,
                voting='hard'
            ),
            
            'Voting Classifier (Soft)': VotingClassifier(
                estimators=[
                    ('rf', base_models['Random Forest (Regularized)']),
                    ('lr', base_models['Logistic Regression (L2)']),
                    ('nb', base_models['Naive Bayes'])
                ],
                voting='soft'
            )
        }
        
        return ensemble_models