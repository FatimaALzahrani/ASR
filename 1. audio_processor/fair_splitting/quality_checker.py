import pandas as pd
import numpy as np
from pathlib import Path
import json


class QualityChecker:
    def __init__(self, data_path="data/processed"):
        self.data_path = Path(data_path)
    
    def load_splits(self):
        try:
            self.train_df = pd.read_csv(self.data_path / "train.csv", encoding='utf-8')
            self.val_df = pd.read_csv(self.data_path / "validation.csv", encoding='utf-8')
            self.test_df = pd.read_csv(self.data_path / "test.csv", encoding='utf-8')
            return True
        except Exception as e:
            print(f"Error loading splits: {e}")
            return False
    
    def check_data_integrity(self):
        print("Data Integrity Check:")
        print("=" * 30)
        
        issues = []
        
        # Check for missing values
        for name, df in [("train", self.train_df), ("validation", self.val_df), ("test", self.test_df)]:
            if df.empty:
                continue
                
            missing_speakers = df['speaker'].isna().sum()
            missing_words = df['word'].isna().sum()
            
            if missing_speakers > 0:
                issues.append(f"{name}: {missing_speakers} missing speaker values")
            if missing_words > 0:
                issues.append(f"{name}: {missing_words} missing word values")
        
        # Check for data leakage
        train_files = set(self.train_df['filename']) if not self.train_df.empty else set()
        val_files = set(self.val_df['filename']) if not self.val_df.empty else set()
        test_files = set(self.test_df['filename']) if not self.test_df.empty else set()
        
        train_val_overlap = train_files.intersection(val_files)
        train_test_overlap = train_files.intersection(test_files)
        val_test_overlap = val_files.intersection(test_files)
        
        if train_val_overlap:
            issues.append(f"Data leakage: {len(train_val_overlap)} files in both train and validation")
        if train_test_overlap:
            issues.append(f"Data leakage: {len(train_test_overlap)} files in both train and test")
        if val_test_overlap:
            issues.append(f"Data leakage: {len(val_test_overlap)} files in both validation and test")
        
        if issues:
            print("Issues found:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("No integrity issues found")
        
        return len(issues) == 0
    
    def check_speaker_balance(self):
        """Check speaker balance across splits"""
        print("\nSpeaker Balance Check:")
        print("=" * 30)
        
        all_data = pd.concat([self.train_df, self.val_df, self.test_df], ignore_index=True)
        speakers = all_data['speaker'].unique()
        
        balance_report = {}
        
        for speaker in speakers:
            train_count = len(self.train_df[self.train_df['speaker'] == speaker])
            val_count = len(self.val_df[self.val_df['speaker'] == speaker])
            test_count = len(self.test_df[self.test_df['speaker'] == speaker])
            total = train_count + val_count + test_count
            
            balance_report[speaker] = {
                'train': train_count,
                'train_pct': (train_count / total * 100) if total > 0 else 0,
                'validation': val_count,
                'val_pct': (val_count / total * 100) if total > 0 else 0,
                'test': test_count,
                'test_pct': (test_count / total * 100) if total > 0 else 0,
                'total': total
            }
        
        print("Speaker distribution (percentages):")
        print("Speaker    | Train | Val  | Test | Total")
        print("-" * 45)
        
        for speaker, data in balance_report.items():
            print(f"{speaker:10} | {data['train_pct']:5.1f} | {data['val_pct']:4.1f} | {data['test_pct']:4.1f} | {data['total']:5}")
        
        return balance_report
    
    def check_word_coverage(self):
        """Check word coverage across splits"""
        print("\nWord Coverage Check:")
        print("=" * 30)
        
        train_words = set(self.train_df['word'].unique()) if not self.train_df.empty else set()
        val_words = set(self.val_df['word'].unique()) if not self.val_df.empty else set()
        test_words = set(self.test_df['word'].unique()) if not self.test_df.empty else set()
        
        all_words = train_words.union(val_words).union(test_words)
        
        print(f"Total unique words: {len(all_words)}")
        print(f"Words in training: {len(train_words)} ({len(train_words)/len(all_words)*100:.1f}%)")
        print(f"Words in validation: {len(val_words)} ({len(val_words)/len(all_words)*100:.1f}%)")
        print(f"Words in test: {len(test_words)} ({len(test_words)/len(all_words)*100:.1f}%)")
        
        # Check for words only in test/validation (potential issues)
        test_only = test_words - train_words
        val_only = val_words - train_words
        
        if test_only:
            print(f"Warning: {len(test_only)} words only in test set")
        if val_only:
            print(f"Warning: {len(val_only)} words only in validation set")
        
        return {
            'total_words': len(all_words),
            'train_words': len(train_words),
            'val_words': len(val_words),
            'test_words': len(test_words),
            'test_only': list(test_only),
            'val_only': list(val_only)
        }
    
    def calculate_fairness_score(self):
        """Calculate overall fairness score"""
        print("\nFairness Score Calculation:")
        print("=" * 30)
        
        if self.test_df.empty:
            print("Cannot calculate fairness score: empty test set")
            return 0
        
        # Calculate coefficient of variation for test set
        speaker_counts = self.test_df['speaker'].value_counts()
        mean_count = speaker_counts.mean()
        std_count = speaker_counts.std()
        cv = std_count / mean_count if mean_count > 0 else float('inf')
        
        # Convert CV to fairness score (0-100, higher is better)
        # CV of 0 = 100% fair, CV of 1 = 0% fair
        fairness_score = max(0, min(100, (1 - cv) * 100))
        
        print(f"Coefficient of Variation: {cv:.3f}")
        print(f"Fairness Score: {fairness_score:.1f}/100")
        
        if fairness_score >= 80:
            print("Assessment: Excellent fairness")
        elif fairness_score >= 60:
            print("Assessment: Good fairness")
        elif fairness_score >= 40:
            print("Assessment: Fair")
        else:
            print("Assessment: Poor fairness - consider rebalancing")
        
        return fairness_score
    
    def generate_quality_report(self):
        """Generate comprehensive quality report"""
        print("Generating Quality Report...")
        
        if not self.load_splits():
            return None
        
        # Run all checks
        integrity_ok = self.check_data_integrity()
        speaker_balance = self.check_speaker_balance()
        word_coverage = self.check_word_coverage()
        fairness_score = self.calculate_fairness_score()
        
        # Generate report
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'data_integrity': integrity_ok,
            'speaker_balance': speaker_balance,
            'word_coverage': word_coverage,
            'fairness_score': fairness_score,
            'dataset_sizes': {
                'train': len(self.train_df),
                'validation': len(self.val_df),
                'test': len(self.test_df)
            }
        }
        
        # Save report
        report_path = self.data_path / 'quality_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"\nQuality report saved: {report_path}")
        return report


def main():
    """Main quality checking function"""
    print("Data Quality Checker for Fair Splits")
    print("=" * 40)
    
    checker = QualityChecker()
    report = checker.generate_quality_report()
    
    if report:
        print(f"\nQuality Check Summary:")
        print(f"Data Integrity: {'PASS' if report['data_integrity'] else 'FAIL'}")
        print(f"Fairness Score: {report['fairness_score']:.1f}/100")
        print(f"Total Samples: {sum(report['dataset_sizes'].values())}")


if __name__ == "__main__":
    main()