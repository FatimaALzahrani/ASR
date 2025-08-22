import numpy as np

class OverfittingDetector:
    
    @staticmethod
    def detect_overfitting(cv_scores, test_score, threshold=0.05):
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        
        indicators = {
            'cv_test_gap': cv_mean - test_score,
            'high_cv_mean': cv_mean > 0.95,
            'zero_std': cv_std < 0.001,
            'unrealistic_performance': cv_mean > 0.90 and cv_std < 0.02
        }
        
        is_overfitted = (
            abs(indicators['cv_test_gap']) > threshold or
            indicators['high_cv_mean'] or
            indicators['zero_std'] or
            indicators['unrealistic_performance']
        )
        
        severity = 'none'
        if is_overfitted:
            if indicators['zero_std'] or indicators['unrealistic_performance']:
                severity = 'severe'
            elif abs(indicators['cv_test_gap']) > 0.1:
                severity = 'moderate'
            else:
                severity = 'mild'
        
        return {
            'is_overfitted': is_overfitted,
            'severity': severity,
            'indicators': indicators,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'test_score': test_score
        }
    
    @staticmethod
    def recommend_fixes(detection_result):
        recommendations = []
        
        if detection_result['severity'] == 'severe':
            recommendations.extend([
                "Reduce features significantly (< 20)",
                "Use simpler models (max_depth=3-5)",
                "Increase min_samples_leaf (5-10)",
                "Reduce n_estimators (< 50)"
            ])
        elif detection_result['severity'] == 'moderate':
            recommendations.extend([
                "Reduce number of features (< 30)",
                "Increase regularization",
                "Use stricter cross-validation"
            ])
        elif detection_result['severity'] == 'mild':
            recommendations.extend([
                "Monitor performance on new data",
                "Apply light feature selection"
            ])
        
        return recommendations