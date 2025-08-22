import pandas as pd
from typing import Dict, List, Tuple
from collections import Counter
import numpy as np


class DataValidator:
    
    def __init__(self):
        self.issues = []
        self.warnings = []
        self.recommendations = []
    
    def validate_dataset(self, df: pd.DataFrame) -> Dict:
        print("Running dataset validation...")
        
        self.issues = []
        self.warnings = []
        self.recommendations = []
        
        # Basic validation
        self._validate_basic_structure(df)
        
        # Distribution validation
        self._validate_word_distribution(df)
        
        # Speaker validation
        self._validate_speaker_distribution(df)
        
        # Quality validation
        self._validate_audio_quality(df)
        
        # Splitting feasibility
        self._validate_splitting_feasibility(df)
        
        return self._generate_report()
    
    def _validate_basic_structure(self, df: pd.DataFrame):
        required_columns = ['filename', 'word', 'speaker', 'duration', 'quality_score']
        
        for col in required_columns:
            if col not in df.columns:
                self.issues.append(f"Missing required column: {col}")
        
        if len(df) == 0:
            self.issues.append("Dataset is empty")
            return
        
        # Check for missing values
        for col in required_columns:
            if col in df.columns and df[col].isnull().any():
                null_count = df[col].isnull().sum()
                self.warnings.append(f"Column '{col}' has {null_count} missing values")
        
        # Check for duplicates
        if df.duplicated().any():
            dup_count = df.duplicated().sum()
            self.warnings.append(f"Dataset contains {dup_count} duplicate rows")
    
    def _validate_word_distribution(self, df: pd.DataFrame):
        if 'word' not in df.columns:
            return
        
        word_counts = df['word'].value_counts()
        
        # Single sample words
        single_words = word_counts[word_counts == 1]
        if len(single_words) > 0:
            self.warnings.append(f"{len(single_words)} words have only 1 sample each")
            self.recommendations.append("Consider collecting more samples for single-occurrence words")
        
        # Very few samples
        few_sample_words = word_counts[(word_counts >= 2) & (word_counts <= 3)]
        if len(few_sample_words) > 0:
            self.warnings.append(f"{len(few_sample_words)} words have only 2-3 samples each")
            self.recommendations.append("Words with 2-3 samples may not split well for validation/testing")
        
        # Check imbalance
        max_samples = word_counts.max()
        min_samples = word_counts.min()
        imbalance_ratio = max_samples / min_samples if min_samples > 0 else float('inf')
        
        if imbalance_ratio > 20:
            self.warnings.append(f"High class imbalance: max={max_samples}, min={min_samples} samples")
            self.recommendations.append("Consider data augmentation for underrepresented words")
        
        # Overall statistics
        total_words = len(word_counts)
        avg_samples = word_counts.mean()
        
        if avg_samples < 5:
            self.warnings.append(f"Low average samples per word: {avg_samples:.1f}")
            self.recommendations.append("Consider collecting more data or reducing vocabulary size")
    
    def _validate_speaker_distribution(self, df: pd.DataFrame):
        if 'speaker' not in df.columns:
            return
        
        speaker_counts = df['speaker'].value_counts()
        
        # Check for speakers with too few samples
        low_sample_speakers = speaker_counts[speaker_counts < 10]
        if len(low_sample_speakers) > 0:
            self.warnings.append(f"{len(low_sample_speakers)} speakers have fewer than 10 samples")
            self.recommendations.append("Speakers with few samples may not be suitable for personalized models")
        
        # Check speaker-word distribution
        speaker_word_matrix = pd.crosstab(df['speaker'], df['word'])
        
        # Find speakers missing many words
        words_per_speaker = (speaker_word_matrix > 0).sum(axis=1)
        total_words = len(df['word'].unique())
        
        for speaker, word_count in words_per_speaker.items():
            coverage = word_count / total_words
            if coverage < 0.5:
                self.warnings.append(f"Speaker '{speaker}' covers only {coverage:.1%} of vocabulary")
        
        # Check for words spoken by only one speaker
        speakers_per_word = (speaker_word_matrix > 0).sum(axis=0)
        single_speaker_words = speakers_per_word[speakers_per_word == 1]
        
        if len(single_speaker_words) > 0:
            self.warnings.append(f"{len(single_speaker_words)} words are spoken by only one speaker")
            self.recommendations.append("Single-speaker words may cause overfitting in general models")
    
    def _validate_audio_quality(self, df: pd.DataFrame):
        quality_columns = ['quality_score', 'snr_db', 'rms_energy', 'clipping_ratio', 'silence_ratio']
        
        for col in quality_columns:
            if col not in df.columns:
                continue
            
            if col == 'quality_score':
                low_quality = df[df[col] < 0.3]
                if len(low_quality) > 0:
                    pct = len(low_quality) / len(df) * 100
                    self.warnings.append(f"{len(low_quality)} samples ({pct:.1f}%) have low quality scores < 0.3")
                    
                    if pct > 20:
                        self.recommendations.append("Consider audio enhancement or quality filtering")
            
            elif col == 'snr_db':
                low_snr = df[df[col] < 10]
                if len(low_snr) > 0:
                    pct = len(low_snr) / len(df) * 100
                    self.warnings.append(f"{len(low_snr)} samples ({pct:.1f}%) have low SNR < 10dB")
            
            elif col == 'clipping_ratio':
                high_clipping = df[df[col] > 0.05]
                if len(high_clipping) > 0:
                    pct = len(high_clipping) / len(df) * 100
                    self.warnings.append(f"{len(high_clipping)} samples ({pct:.1f}%) have high clipping > 5%")
            
            elif col == 'silence_ratio':
                high_silence = df[df[col] > 0.7]
                if len(high_silence) > 0:
                    pct = len(high_silence) / len(df) * 100
                    self.warnings.append(f"{len(high_silence)} samples ({pct:.1f}%) have high silence > 70%")
    
    def _validate_splitting_feasibility(self, df: pd.DataFrame):
        if 'word' not in df.columns or 'speaker' not in df.columns:
            return
        
        word_counts = df['word'].value_counts()
        
        # Words that cannot be split
        unsplittable_words = word_counts[word_counts <= 2]
        if len(unsplittable_words) > 0:
            pct = len(unsplittable_words) / len(word_counts) * 100
            self.warnings.append(f"{len(unsplittable_words)} words ({pct:.1f}%) cannot be split for train/test")
        
        # Words that can only have train/test (no validation)
        limited_split_words = word_counts[(word_counts > 2) & (word_counts <= 5)]
        if len(limited_split_words) > 0:
            pct = len(limited_split_words) / len(word_counts) * 100
            self.warnings.append(f"{len(limited_split_words)} words ({pct:.1f}%) can only be split for train/test")
        
        # Check stratification feasibility
        speaker_word_matrix = pd.crosstab(df['speaker'], df['word'])
        
        stratifiable_words = 0
        for word in df['word'].unique():
            word_data = df[df['word'] == word]
            speaker_counts = word_data['speaker'].value_counts()
            
            # Need at least 2 speakers with 2+ samples each for stratification
            viable_speakers = (speaker_counts >= 2).sum()
            if viable_speakers >= 2:
                stratifiable_words += 1
        
        if stratifiable_words < len(df['word'].unique()) * 0.5:
            pct = stratifiable_words / len(df['word'].unique()) * 100
            self.warnings.append(f"Only {stratifiable_words} words ({pct:.1f}%) can use stratified splitting")
            self.recommendations.append("Consider collecting more balanced data across speakers")
    
    def _generate_report(self) -> Dict:
        severity_score = len(self.issues) * 3 + len(self.warnings)
        
        if severity_score == 0:
            status = "EXCELLENT"
        elif severity_score <= 3:
            status = "GOOD"
        elif severity_score <= 8:
            status = "FAIR"
        else:
            status = "POOR"
        
        return {
            'status': status,
            'severity_score': severity_score,
            'issues': self.issues,
            'warnings': self.warnings,
            'recommendations': self.recommendations,
            'summary': {
                'total_issues': len(self.issues),
                'total_warnings': len(self.warnings),
                'total_recommendations': len(self.recommendations)
            }
        }
    
    def print_report(self, report: Dict):
        print(f"\n{'='*60}")
        print(f"DATASET VALIDATION REPORT - Status: {report['status']}")
        print(f"{'='*60}")
        
        if report['issues']:
            print(f"\n🔴 CRITICAL ISSUES ({len(report['issues'])}):")
            for i, issue in enumerate(report['issues'], 1):
                print(f"  {i}. {issue}")
        
        if report['warnings']:
            print(f"\n🟡 WARNINGS ({len(report['warnings'])}):")
            for i, warning in enumerate(report['warnings'], 1):
                print(f"  {i}. {warning}")
        
        if report['recommendations']:
            print(f"\n💡 RECOMMENDATIONS ({len(report['recommendations'])}):")
            for i, rec in enumerate(report['recommendations'], 1):
                print(f"  {i}. {rec}")
        
        if not report['issues'] and not report['warnings']:
            print("\n✅ Dataset passed all validation checks!")
        
        print(f"\n{'='*60}")
    
    def get_splitting_strategy_advice(self, df: pd.DataFrame) -> Dict:
        if 'word' not in df.columns:
            return {}
        
        word_counts = df['word'].value_counts()
        
        strategies = {
            'single_sample_words': len(word_counts[word_counts == 1]),
            'small_words_2_4': len(word_counts[(word_counts >= 2) & (word_counts <= 4)]),
            'medium_words_5_15': len(word_counts[(word_counts >= 5) & (word_counts <= 15)]),
            'large_words_15+': len(word_counts[word_counts > 15]),
        }
        
        advice = []
        
        if strategies['single_sample_words'] > 0:
            advice.append(f"Put {strategies['single_sample_words']} single-sample words in training only")
        
        if strategies['small_words_2_4'] > 0:
            advice.append(f"Use simple split for {strategies['small_words_2_4']} small words (2-4 samples)")
        
        if strategies['medium_words_5_15'] > 0:
            advice.append(f"Use train/test split for {strategies['medium_words_5_15']} medium words (5-15 samples)")
        
        if strategies['large_words_15+'] > 0:
            advice.append(f"Use full train/val/test split for {strategies['large_words_15+']} large words (15+ samples)")
        
        return {
            'strategies': strategies,
            'advice': advice,
            'recommended_approach': 'adaptive_splitting_by_word_size'
        }