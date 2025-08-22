from pathlib import Path
from data_analyzer import DataAnalyzer
from fair_splitter import FairSplitter
from data_saver_fair import DataSaverFair
from distribution_comparer import DistributionComparer


class FairDataProcessor:
    def __init__(self, data_path="data/processed"):
        self.data_path = Path(data_path)
        self.analyzer = DataAnalyzer(data_path)
        self.splitter = FairSplitter(test_size=0.2, val_size=0.1)
        self.saver = DataSaverFair(data_path)
        self.comparer = DistributionComparer()
    
    def run_fair_splitting(self):
        print("Fair Data Splitting for Down Syndrome Speech Recognition")
        print("=" * 70)
        
        # Load and analyze current data
        if not self.analyzer.load_current_data():
            return None
        
        self.analyzer.analyze_current_distribution()
        
        # Create fair splits
        train_df, val_df, test_df = self.splitter.create_fair_splits(self.analyzer.all_data)
        
        # Analyze new distribution
        self.analyzer.analyze_new_distribution(train_df, val_df, test_df)
        
        # Save new splits
        self.saver.save_fair_splits(
            train_df, val_df, test_df,
            self.analyzer.train_df, self.analyzer.val_df, self.analyzer.test_df
        )
        
        print(f"\nFair splitting completed successfully!")
        print(f"All speakers are now fairly represented in all datasets")
        print(f"You can now retrain the models for fair results")
        
        return train_df, val_df, test_df
    
    def evaluate_fairness(self):
        """Evaluate the fairness of current data splits"""
        if not self.analyzer.load_current_data():
            return
        
        print("\nFairness Evaluation:")
        print("=" * 30)
        
        # Calculate fairness metrics for test set
        fairness_metrics = self.comparer.calculate_fairness_metrics(self.analyzer.test_df)
        
        if fairness_metrics:
            print(f"Coefficient of Variation: {fairness_metrics['coefficient_of_variation']:.3f}")
            print(f"Min/Max Ratio: {fairness_metrics['min_max_ratio']:.3f}")
            print("Speaker percentages in test set:")
            for speaker, percentage in fairness_metrics['speaker_percentages'].items():
                print(f"  {speaker}: {percentage:.1f}%")
            
            # Interpretation
            if fairness_metrics['coefficient_of_variation'] < 0.3:
                print("Assessment: Fair distribution")
            elif fairness_metrics['coefficient_of_variation'] < 0.5:
                print("Assessment: Moderately fair distribution")
            else:
                print("Assessment: Unfair distribution - consider running fair splitting")
        
        return fairness_metrics