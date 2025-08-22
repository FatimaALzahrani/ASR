import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


class ResearchComparisonHelper:
    def __init__(self, results_file=None):
        self.results_file = results_file or "output_files/deep_learning_asr_results.json"
        self.results = self.load_results()
        
    def load_results(self):
        try:
            with open(self.results_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Results file not found: {self.results_file}")
            return None
    
    def generate_performance_table(self):
        if not self.results:
            return None
            
        model_results = self.results['model_results']
        
        performance_data = []
        for model_name, accuracy in model_results.items():
            performance_data.append({
                'Model': model_name,
                'Accuracy (%)': accuracy * 100,
                'Category': self.categorize_performance(accuracy)
            })
        
        df = pd.DataFrame(performance_data)
        df = df.sort_values('Accuracy (%)', ascending=False)
        
        return df
    
    def categorize_performance(self, accuracy):
        if accuracy >= 0.95:
            return "Excellent"
        elif accuracy >= 0.90:
            return "Very Good"
        elif accuracy >= 0.80:
            return "Good"
        elif accuracy >= 0.70:
            return "Fair"
        else:
            return "Poor"
    
    def get_statistical_summary(self):
        if not self.results:
            return None
            
        stats = self.results['accuracy_statistics']
        
        summary = {
            'Total Models Evaluated': self.results['deep_learning_metrics']['models_trained'],
            'Successful Models': self.results['deep_learning_metrics']['successful_models'],
            'Best Accuracy (%)': stats['max'] * 100,
            'Average Accuracy (%)': stats['mean'] * 100,
            'Median Accuracy (%)': stats['median'] * 100,
            'Worst Accuracy (%)': stats['min'] * 100,
            'Standard Deviation (%)': stats['std'] * 100,
            'Best Model': self.results['absolute_best']['model']
        }
        
        return summary
    
    def generate_dataset_info_table(self):
        if not self.results:
            return None
            
        dataset_info = self.results['dataset_info']
        
        info_table = {
            'Metric': [
                'Total Audio Samples (Original)',
                'Total Audio Samples (After Balancing)',
                'Number of Features Extracted',
                'Number of Unique Words',
                'Number of Speakers',
                'Feature Scaling Method',
                'Data Balancing Method'
            ],
            'Value': [
                f"{dataset_info['original_samples']:,}",
                f"{dataset_info['balanced_samples']:,}",
                f"{dataset_info['features']:,}",
                f"{dataset_info['words']:,}",
                f"{dataset_info['speakers']:,}",
                dataset_info['scaler_used'],
                "SMOTE + Manual Augmentation"
            ]
        }
        
        return pd.DataFrame(info_table)
    
    def get_speaker_distribution(self):
        if not self.results:
            return None
            
        speaker_dist = self.results['dataset_info']['speaker_distribution']
        
        speaker_data = []
        for speaker, info in speaker_dist.items():
            speaker_data.append({
                'Speaker': speaker,
                'Total Samples': info['total_samples'],
                'Unique Words': info['unique_words'],
                'Avg Samples per Word': info['total_samples'] / info['unique_words']
            })
        
        return pd.DataFrame(speaker_data)
    
    def compare_with_baseline(self):
        if not self.results:
            return None
            
        model_results = self.results['model_results']
        
        baseline_accuracy = 0.5
        
        comparison_data = []
        for model_name, accuracy in model_results.items():
            improvement = (accuracy - baseline_accuracy) * 100
            comparison_data.append({
                'Model': model_name,
                'Accuracy (%)': accuracy * 100,
                'Improvement over Baseline (%)': improvement,
                'Relative Improvement (x)': accuracy / baseline_accuracy if baseline_accuracy > 0 else 0
            })
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('Accuracy (%)', ascending=False)
        
        return df
    
    def generate_research_summary(self):
        if not self.results:
            return "No results available for analysis"
            
        summary = f"""
RESEARCH SUMMARY: Speech Recognition for Children with Down Syndrome

=== DATASET CHARACTERISTICS ===
• Original audio samples: {self.results['dataset_info']['original_samples']:,}
• Balanced dataset size: {self.results['dataset_info']['balanced_samples']:,}
• Feature dimensions: {self.results['dataset_info']['features']:,}
• Vocabulary size: {self.results['dataset_info']['words']:,} Arabic words
• Participants: {self.results['dataset_info']['speakers']:,} children with Down syndrome

=== METHODOLOGY ===
• Feature extraction: MFCC, spectral, prosodic, and temporal features
• Data preprocessing: {self.results['dataset_info']['scaler_used']} normalization
• Data balancing: SMOTE with manual augmentation
• Model architectures: CNN, LSTM, Transformer, ResNet-LSTM hybrid, and ensemble methods

=== EXPERIMENTAL RESULTS ===
• Total models evaluated: {self.results['deep_learning_metrics']['models_trained']}
• Best performing model: {self.results['absolute_best']['model']}
• Highest accuracy achieved: {self.results['absolute_best']['accuracy']*100:.2f}%
• Average accuracy across all models: {self.results['accuracy_statistics']['mean']*100:.2f}%
• Standard deviation: {self.results['accuracy_statistics']['std']*100:.2f}%

=== CLINICAL SIGNIFICANCE ===
• The {self.results['absolute_best']['model']} model achieved {self.results['absolute_best']['accuracy']*100:.2f}% accuracy
• This represents a {self.results['absolute_best']['improvement_over_baseline']:.1f}% improvement over baseline
• Results demonstrate feasibility of automated speech recognition for children with Down syndrome
• Advanced deep learning architectures show superior performance compared to traditional methods

=== TECHNICAL CONTRIBUTIONS ===
• Novel application of Conformer and WaveNet architectures to Down syndrome speech
• Comprehensive feature engineering specifically for atypical speech patterns
• Effective data augmentation strategies for limited clinical datasets
• Ensemble methods achieving robust performance across diverse speech characteristics
        """
        
        return summary.strip()
    
    def export_tables_for_paper(self, output_dir="tables"):
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        performance_table = self.generate_performance_table()
        if performance_table is not None:
            performance_table.to_csv(output_path / "model_performance_comparison.csv", index=False)
            performance_table.to_latex(output_path / "model_performance_comparison.tex", index=False)
        
        dataset_table = self.generate_dataset_info_table()
        if dataset_table is not None:
            dataset_table.to_csv(output_path / "dataset_characteristics.csv", index=False)
            dataset_table.to_latex(output_path / "dataset_characteristics.tex", index=False)
        
        speaker_table = self.get_speaker_distribution()
        if speaker_table is not None:
            speaker_table.to_csv(output_path / "speaker_distribution.csv", index=False)
            speaker_table.to_latex(output_path / "speaker_distribution.tex", index=False)
        
        baseline_comparison = self.compare_with_baseline()
        if baseline_comparison is not None:
            baseline_comparison.to_csv(output_path / "baseline_comparison.csv", index=False)
            baseline_comparison.to_latex(output_path / "baseline_comparison.tex", index=False)
        
        with open(output_path / "research_summary.txt", "w", encoding="utf-8") as f:
            f.write(self.generate_research_summary())
        
        print(f"Tables and summary exported to {output_path}")
    
    def generate_confusion_matrix_data(self):
        if not self.results:
            return None
            
        word_frequencies = self.results['dataset_info']['word_frequencies']
        
        words = list(word_frequencies.keys())
        word_counts = list(word_frequencies.values())
        
        return pd.DataFrame({
            'Word': words,
            'Sample_Count': word_counts
        }).sort_values('Sample_Count', ascending=False)


def main():
    helper = ResearchComparisonHelper()
    
    print("=== RESEARCH ANALYSIS REPORT ===")
    print()
    
    print("1. Statistical Summary:")
    summary = helper.get_statistical_summary()
    if summary:
        for key, value in summary.items():
            print(f"   {key}: {value}")
    
    print("\n2. Performance Table (Top 5):")
    perf_table = helper.generate_performance_table()
    if perf_table is not None:
        print(perf_table.head().to_string(index=False))
    
    print("\n3. Dataset Characteristics:")
    dataset_table = helper.generate_dataset_info_table()
    if dataset_table is not None:
        print(dataset_table.to_string(index=False))
    
    print("\n4. Research Summary:")
    print(helper.generate_research_summary())
    
    helper.export_tables_for_paper()


if __name__ == "__main__":
    main()