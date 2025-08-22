class QualityMetricsCalculator:
    def calculate_quality_score(self, quality_metrics):
        score = 0.5
        
        if quality_metrics['snr'] > 20:
            score += 0.2
        elif quality_metrics['snr'] > 10:
            score += 0.1
        elif quality_metrics['snr'] < 5:
            score -= 0.2
        
        if 0.05 < quality_metrics['rms_energy'] < 0.2:
            score += 0.1
        elif quality_metrics['rms_energy'] < 0.01:
            score -= 0.2
        
        if quality_metrics['clipping_ratio'] < 0.01:
            score += 0.1
        elif quality_metrics['clipping_ratio'] > 0.05:
            score -= 0.2
        
        if quality_metrics['silence_ratio'] < 0.3:
            score += 0.1
        elif quality_metrics['silence_ratio'] > 0.6:
            score -= 0.1
        
        return max(0.0, min(1.0, score))