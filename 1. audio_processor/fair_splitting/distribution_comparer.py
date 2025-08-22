import pandas as pd


class DistributionComparer:
    def compare_distributions(self):
        print("\nQuick distribution comparison:")
        
        try:
            old_test = pd.read_csv("data/processed/backup_unfair/test.csv", encoding='utf-8')
            print("Old distribution (unfair):")
            old_dist = old_test['speaker'].value_counts()
            for speaker, count in old_dist.items():
                print(f"  {speaker}: {count} samples")
            
            new_test = pd.read_csv("data/processed/test.csv", encoding='utf-8')
            print("\nNew distribution (fair):")
            new_dist = new_test['speaker'].value_counts()
            for speaker, count in new_dist.items():
                print(f"  {speaker}: {count} samples")
                
        except Exception as e:
            print(f"Cannot compare: {e}")
    
    def calculate_fairness_metrics(self, test_df):
        if test_df.empty:
            return {}
        
        speaker_counts = test_df['speaker'].value_counts()
        total_samples = len(test_df)
        
        count_values = list(speaker_counts.values) if hasattr(speaker_counts, 'values') else list(speaker_counts)
        speaker_names = list(speaker_counts.index) if hasattr(speaker_counts, 'index') else list(range(len(count_values)))
        
        percentages = [(count / total_samples) * 100 for count in count_values]
        mean_percentage = sum(percentages) / len(percentages) if percentages else 0
        
        if mean_percentage > 0:
            variance = sum((p - mean_percentage) ** 2 for p in percentages) / len(percentages)
            std_dev = variance ** 0.5
            cv = std_dev / mean_percentage
        else:
            cv = 0
        
        min_count = min(count_values) if count_values else 0
        max_count = max(count_values) if count_values else 0
        min_max_ratio = min_count / max_count if max_count > 0 else 0
        
        return {
            'coefficient_of_variation': cv,
            'min_max_ratio': min_max_ratio,
            'speaker_percentages': dict(zip(speaker_names, percentages))
        }