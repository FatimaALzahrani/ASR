import json
import pandas as pd
from datetime import datetime


class ReportGenerator:
    def __init__(self, dataframe, speaker_manager, statistics_calc):
        self.df = dataframe
        self.speaker_manager = speaker_manager
        self.stats_calc = statistics_calc
    
    def generate_recommendations(self):
        recommendations = []
        
        speaker_quality = self.df.groupby('speaker')['quality_score'].mean()
        worst_speaker = speaker_quality.idxmin()
        best_speaker = speaker_quality.idxmax()
        
        recommendations.append(f"Speaker {best_speaker} achieves best speech quality ({speaker_quality[best_speaker]:.3f})")
        recommendations.append(f"Speaker {worst_speaker} needs improvement in speech quality ({speaker_quality[worst_speaker]:.3f})")
        
        word_quality = self.df.groupby('word')['quality_score'].mean()
        difficult_words = word_quality.nsmallest(5)
        easy_words = word_quality.nlargest(5)
        
        recommendations.append(f"Most difficult words: {', '.join(difficult_words.index[:3])}")
        recommendations.append(f"Easiest words: {', '.join(easy_words.index[:3])}")
        
        avg_duration = self.df['duration'].mean()
        if avg_duration < 1.5:
            recommendations.append("Recordings are relatively short, may need longer recordings")
        elif avg_duration > 4.0:
            recommendations.append("Recordings are relatively long, can be shortened")
        
        word_counts = self.df['word'].value_counts()
        if word_counts.std() > word_counts.mean():
            recommendations.append("Unbalanced word distribution, some words need additional recordings")
        
        return recommendations
    
    def generate_detailed_report(self):
        print("Generating detailed report...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'dataset_overview': {
                'total_files': len(self.df),
                'total_words': self.df['word'].nunique(),
                'total_speakers': self.df['speaker'].nunique(),
                'total_duration_hours': round(self.df['duration'].sum() / 3600, 2),
                'average_file_duration': round(self.df['duration'].mean(), 2),
                'total_size_mb': round(self.df['file_size'].sum() / (1024*1024), 2)
            },
            'speaker_analysis': {},
            'word_analysis': {},
            'quality_distribution': {},
            'recommendations': []
        }
        
        for speaker in self.df['speaker'].unique():
            speaker_data = self.df[self.df['speaker'] == speaker]
            report['speaker_analysis'][speaker] = {
                'total_recordings': len(speaker_data),
                'unique_words': speaker_data['word'].nunique(),
                'total_duration': round(speaker_data['duration'].sum(), 2),
                'average_duration': round(speaker_data['duration'].mean(), 2),
                'average_quality': round(speaker_data['quality_score'].mean(), 3),
                'quality_std': round(speaker_data['quality_score'].std(), 3),
                'most_recorded_word': speaker_data['word'].value_counts().index[0] if len(speaker_data) > 0 else None,
                'speaker_info': self.speaker_manager.speaker_info.get(speaker, {})
            }
        
        for word in self.df['word'].unique():
            word_data = self.df[self.df['word'] == word]
            report['word_analysis'][word] = {
                'total_recordings': len(word_data),
                'unique_speakers': word_data['speaker'].nunique(),
                'average_duration': round(word_data['duration'].mean(), 2),
                'average_quality': round(word_data['quality_score'].mean(), 3),
                'quality_std': round(word_data['quality_score'].std(), 3),
                'best_speaker': word_data.groupby('speaker')['quality_score'].mean().idxmax(),
                'worst_speaker': word_data.groupby('speaker')['quality_score'].mean().idxmin()
            }
        
        quality_bins = pd.cut(self.df['quality_score'], bins=[0, 0.3, 0.6, 1.0], 
                             labels=['Poor', 'Average', 'Good'])
        quality_counts = quality_bins.value_counts()
        report['quality_distribution'] = {
            'poor': int(quality_counts.get('Poor', 0)),
            'average': int(quality_counts.get('Average', 0)),
            'good': int(quality_counts.get('Good', 0))
        }
        
        report['recommendations'] = self.generate_recommendations()
        
        with open('comprehensive_analysis_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print("Detailed report saved successfully")
        return report