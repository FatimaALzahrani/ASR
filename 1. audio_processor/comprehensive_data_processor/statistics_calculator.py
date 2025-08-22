import pandas as pd


class StatisticsCalculator:
    def __init__(self, audio_files):
        self.df = pd.DataFrame(audio_files)
        self.statistics = {}
        self.speaker_analysis = {}
        self.word_analysis = {}
        self.quality_analysis = {}
    
    def calculate_all_statistics(self):
        print("Calculating statistics...")
        
        self.statistics['total_files'] = len(self.df)
        self.statistics['total_words'] = self.df['word'].nunique()
        self.statistics['total_speakers'] = self.df['speaker'].nunique()
        self.statistics['total_duration'] = self.df['duration'].sum()
        self.statistics['avg_duration'] = self.df['duration'].mean()
        self.statistics['avg_file_size'] = self.df['file_size'].mean()
        
        speaker_stats = self.df.groupby('speaker').agg({
            'file_path': 'count',
            'duration': ['sum', 'mean'],
            'quality_score': 'mean',
            'word': 'nunique'
        }).round(3)
        
        self.speaker_analysis = speaker_stats.to_dict()
        
        word_stats = self.df.groupby('word').agg({
            'file_path': 'count',
            'duration': ['sum', 'mean'],
            'quality_score': 'mean',
            'speaker': 'nunique'
        }).round(3)
        
        self.word_analysis = word_stats.to_dict()
        
        quality_bins = pd.cut(self.df['quality_score'], bins=[0, 0.3, 0.6, 1.0], 
                             labels=['Poor', 'Average', 'Good'])
        self.quality_analysis = quality_bins.value_counts().to_dict()
        
        print("Statistics calculation completed")