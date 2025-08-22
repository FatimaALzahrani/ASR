import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Dict


class DataVisualizerFixed:
    def __init__(self, reports_dir: Path):
        self.reports_dir = reports_dir
    
    def create_visualizations(self, df: pd.DataFrame, balance_info: Dict):
        print("\nCreating visualization charts...")
        
        try:
            plt.style.use('default')
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Down Syndrome Children Speech Data Analysis', fontsize=16, fontweight='bold')
            
            # 1. Most frequent words
            top_words = pd.Series(balance_info['word_distribution']).head(15)
            top_words.plot(kind='bar', ax=axes[0,0], color='skyblue')
            axes[0,0].set_title('Top 15 Most Frequent Words')
            axes[0,0].tick_params(axis='x', rotation=45)
            
            # 2. Speaker distribution
            speaker_counts = pd.Series(balance_info['speaker_distribution'])
            speaker_counts.plot(kind='bar', ax=axes[0,1], color='lightgreen')
            axes[0,1].set_title('Sample Distribution by Speaker')
            
            # 3. Speech quality distribution
            quality_counts = pd.Series(balance_info['quality_distribution'])
            axes[0,2].pie(quality_counts.values, labels=quality_counts.index, autopct='%1.1f%%')
            axes[0,2].set_title('Speech Quality Distribution')
            
            # 4. Recording duration
            df['duration'].hist(bins=30, ax=axes[1,0], color='orange', alpha=0.7)
            axes[1,0].set_title('Recording Duration Distribution (seconds)')
            axes[1,0].set_xlabel('Duration')
            axes[1,0].set_ylabel('Frequency')
            
            # 5. Audio quality
            df['quality_score'].hist(bins=20, ax=axes[1,1], color='purple', alpha=0.7)
            axes[1,1].set_title('Audio Quality Score Distribution')
            axes[1,1].set_xlabel('Quality Score')
            axes[1,1].set_ylabel('Frequency')
            
            # 6. Sample distribution by categories
            categories = ['Single', 'Few (2-4)', 'Moderate (5-14)', 'Many (15+)']
            counts = [
                len(balance_info['single_sample_words']),
                len(balance_info['few_sample_words']),
                len(balance_info['moderate_sample_words']),
                len(balance_info['many_sample_words'])
            ]
            axes[1,2].bar(categories, counts, color=['red', 'orange', 'yellow', 'green'])
            axes[1,2].set_title('Word Classification by Sample Count')
            axes[1,2].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            output_path = self.reports_dir / 'data_analysis_fixed.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Charts saved to: {output_path}")
            
        except Exception as e:
            print(f"Warning: Error creating charts: {e}")