import matplotlib.pyplot as plt
import pandas as pd


class DataVisualizer:
    def __init__(self, dataframe):
        self.df = dataframe
        plt.rcParams['font.family'] = ['DejaVu Sans']
    
    def create_visualizations(self):
        print("Creating visualizations...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Audio Data Analysis for Children with Down Syndrome', fontsize=16, y=0.98)
        
        speaker_counts = self.df['speaker'].value_counts()
        axes[0, 0].bar(speaker_counts.index, speaker_counts.values, color='skyblue')
        axes[0, 0].set_title('Recordings per Speaker')
        axes[0, 0].set_ylabel('Number of Recordings')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        axes[0, 1].hist(self.df['duration'], bins=20, color='lightgreen', alpha=0.7)
        axes[0, 1].set_title('Recording Duration Distribution')
        axes[0, 1].set_xlabel('Duration (seconds)')
        axes[0, 1].set_ylabel('Frequency')
        
        quality_bins = pd.cut(self.df['quality_score'], bins=[0, 0.3, 0.6, 1.0], 
                             labels=['Poor', 'Average', 'Good'])
        quality_counts = quality_bins.value_counts()
        axes[0, 2].pie(quality_counts.values, labels=quality_counts.index, autopct='%1.1f%%')
        axes[0, 2].set_title('Quality Distribution')
        
        speaker_quality = self.df.groupby('speaker')['quality_score'].mean().sort_values(ascending=False)
        axes[1, 0].bar(speaker_quality.index, speaker_quality.values, color='orange')
        axes[1, 0].set_title('Average Quality per Speaker')
        axes[1, 0].set_ylabel('Quality Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        top_words = self.df['word'].value_counts().head(10)
        axes[1, 1].barh(range(len(top_words)), top_words.values, color='purple', alpha=0.7)
        axes[1, 1].set_yticks(range(len(top_words)))
        axes[1, 1].set_yticklabels(top_words.index)
        axes[1, 1].set_title('Top 10 Most Recorded Words')
        axes[1, 1].set_xlabel('Number of Recordings')
        
        axes[1, 2].scatter(self.df['duration'], self.df['quality_score'], alpha=0.6, color='red')
        axes[1, 2].set_title('Duration vs Quality Relationship')
        axes[1, 2].set_xlabel('Duration (seconds)')
        axes[1, 2].set_ylabel('Quality Score')
        
        plt.tight_layout()
        plt.savefig('comprehensive_data_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Visualizations saved successfully")