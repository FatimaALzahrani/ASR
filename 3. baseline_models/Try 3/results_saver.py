import json


class ResultsSaver:
    def __init__(self, train_speakers, test_speakers, X_train, X_test):
        self.train_speakers = train_speakers
        self.test_speakers = test_speakers
        self.X_train = X_train
        self.X_test = X_test
        
    def save_comprehensive_results(self, all_results):
        print("Saving comprehensive results...")
        
        comprehensive_results = {
            'methodology': 'Corrected speaker-based split with strong regularization',
            'train_speakers': self.train_speakers,
            'test_speakers': self.test_speakers,
            'train_samples': len(self.X_train),
            'test_samples': len(self.X_test),
            'features': self.X_train.shape[1],
            'models': all_results,
            'summary': {
                'best_model': None,
                'best_test_accuracy': 0,
                'best_cv_score': 0,
                'models_with_low_overfitting': []
            }
        }
        
        best_test_acc = 0
        best_cv_score = 0
        best_model = None
        low_overfitting_models = []
        
        for name, result in all_results.items():
            if result is not None:
                if result['test_accuracy'] > best_test_acc:
                    best_test_acc = result['test_accuracy']
                    best_model = name
                
                if result['cv_mean'] > best_cv_score:
                    best_cv_score = result['cv_mean']
                
                if result['overfitting_gap'] < 0.1:
                    low_overfitting_models.append(name)
        
        comprehensive_results['summary']['best_model'] = best_model
        comprehensive_results['summary']['best_test_accuracy'] = best_test_acc
        comprehensive_results['summary']['best_cv_score'] = best_cv_score
        comprehensive_results['summary']['models_with_low_overfitting'] = low_overfitting_models
        
        with open('improved_models_comprehensive_results.json', 'w', encoding='utf-8') as f:
            json.dump(comprehensive_results, f, ensure_ascii=False, indent=2)
        
        print("Results saved to: improved_models_comprehensive_results.json")
        
        return comprehensive_results