import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import Dict


class DataVisualizer:
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def create_analysis_plots(self, df: pd.DataFrame, balance_info: Dict):
        print("Creating data visualization plots...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Down Syndrome Speech Data Analysis', fontsize=16, fontweight='bold')
        
        # 1. Word distribution (top 20)
        self._plot_word_distribution(df, balance_info, axes[0, 0])
        
        # 2. Speaker distribution
        self._plot_speaker_distribution(balance_info, axes[0, 1])
        
        # 3. Speech quality distribution
        self._plot_quality_distribution(balance_info, axes[0, 2])
        
        # 4. Duration distribution
        self._plot_duration_distribution(df, axes[1, 0])
        
        # 5. Audio quality by speaker
        self._plot_quality_by_speaker(df, axes[1, 1])
        
        # 6. SNR vs Quality correlation
        self._plot_snr_quality_correlation(df, axes[1, 2])
        
        plt.tight_layout()
        output_path = self.output_dir / 'data_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Analysis plots saved to: {output_path}")
    
    def _plot_word_distribution(self, df: pd.DataFrame, balance_info: Dict, ax):
        word_counts = pd.Series(balance_info['word_distribution']).head(20)
        word_counts.plot(kind='bar', ax=ax, color='skyblue')
        ax.set_title('Top 20 Most Frequent Words')
        ax.set_xlabel('Words')
        ax.set_ylabel('Frequency')
        ax.tick_params(axis='x', rotation=45)
    
    def _plot_speaker_distribution(self, balance_info: Dict, ax):
        speaker_counts = pd.Series(balance_info['speaker_distribution'])
        speaker_counts.plot(kind='bar', ax=ax, color='lightgreen')
        ax.set_title('Sample Distribution by Speaker')
        ax.set_xlabel('Speaker')
        ax.set_ylabel('Number of Samples')
        ax.tick_params(axis='x', rotation=45)
    
    def _plot_quality_distribution(self, balance_info: Dict, ax):
        quality_counts = pd.Series(balance_info['quality_distribution'])
        quality_counts.plot(kind='pie', ax=ax, autopct='%1.1f%%')
        ax.set_title('Speech Quality Distribution')
    
    def _plot_duration_distribution(self, df: pd.DataFrame, ax):
        df['duration'].hist(bins=30, ax=ax, color='orange', alpha=0.7)
        ax.set_title('Recording Duration Distribution')
        ax.set_xlabel('Duration (seconds)')
        ax.set_ylabel('Frequency')
    
    def _plot_quality_by_speaker(self, df: pd.DataFrame, ax):
        df.boxplot(column='quality_score', by='speaker', ax=ax)
        ax.set_title('Audio Quality Score by Speaker')
        ax.set_xlabel('Speaker')
        ax.set_ylabel('Quality Score')
    
    def _plot_snr_quality_correlation(self, df: pd.DataFrame, ax):
        for quality in df['speech_quality'].unique():
            quality_data = df[df['speech_quality'] == quality]
            ax.scatter(quality_data['snr_db'], quality_data['quality_score'], 
                      label=quality, alpha=0.6)
        
        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel('Quality Score')
        ax.set_title('SNR vs Audio Quality Correlation')
        ax.legend()
    
    def create_balance_plots(self, balance_info: Dict):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Data Balance Analysis', fontsize=14, fontweight='bold')
        
        # Word count distribution
        word_counts = pd.Series(balance_info['word_distribution'])
        word_counts.hist(bins=20, ax=axes[0, 0], color='lightblue', alpha=0.7)
        axes[0, 0].set_title('Distribution of Samples per Word')
        axes[0, 0].set_xlabel('Number of Samples')
        axes[0, 0].set_ylabel('Number of Words')
        
        # Rare vs common words
        rare_count = len(balance_info['rare_words'])
        common_count = len(balance_info['common_words'])
        normal_count = balance_info['total_words'] - rare_count - common_count
        
        categories = ['Rare\n(< 3 samples)', 'Normal\n(3-20 samples)', 'Common\n(> 20 samples)']
        counts = [rare_count, normal_count, common_count]
        
        axes[0, 1].bar(categories, counts, color=['red', 'yellow', 'green'], alpha=0.7)
        axes[0, 1].set_title('Word Categories by Sample Count')
        axes[0, 1].set_ylabel('Number of Words')
        
        # Speaker balance
        speaker_counts = pd.Series(balance_info['speaker_distribution'])
        speaker_counts.plot(kind='bar', ax=axes[1, 0], color='lightcoral')
        axes[1, 0].set_title('Speaker Sample Distribution')
        axes[1, 0].set_ylabel('Number of Samples')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Quality distribution
        quality_counts = pd.Series(balance_info['quality_distribution'])
        quality_counts.plot(kind='bar', ax=axes[1, 1], color='lightsteelblue')
        axes[1, 1].set_title('Speech Quality Distribution')
        axes[1, 1].set_ylabel('Number of Samples')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        output_path = self.output_dir / 'balance_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Balance analysis plots saved to: {output_path}")
    
    def create_quality_analysis_plots(self, df: pd.DataFrame):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Audio Quality Analysis', fontsize=14, fontweight='bold')
        
        # SNR distribution
        df['snr_db'].hist(bins=30, ax=axes[0, 0], color='skyblue', alpha=0.7)
        axes[0, 0].set_title('SNR Distribution')
        axes[0, 0].set_xlabel('SNR (dB)')
        axes[0, 0].set_ylabel('Frequency')
        
        # RMS Energy vs Quality Score
        axes[0, 1].scatter(df['rms_energy'], df['quality_score'], alpha=0.6)
        axes[0, 1].set_title('RMS Energy vs Quality Score')
        axes[0, 1].set_xlabel('RMS Energy')
        axes[0, 1].set_ylabel('Quality Score')
        
        # Clipping ratio distribution
        df['clipping_ratio'].hist(bins=30, ax=axes[1, 0], color='orange', alpha=0.7)
        axes[1, 0].set_title('Clipping Ratio Distribution')
        axes[1, 0].set_xlabel('Clipping Ratio')
        axes[1, 0].set_ylabel('Frequency')
        
        # Overall quality score distribution
        df['quality_score'].hist(bins=30, ax=axes[1, 1], color='lightgreen', alpha=0.7)
        axes[1, 1].set_title('Overall Quality Score Distribution')
        axes[1, 1].set_xlabel('Quality Score')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        output_path = self.output_dir / 'quality_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Quality analysis plots saved to: {output_path}")