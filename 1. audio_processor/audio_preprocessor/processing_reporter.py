import json
import numpy as np
import pandas as pd


class ProcessingReporter:
    def __init__(self, quality_metrics_calculator):
        self.quality_metrics_calculator = quality_metrics_calculator
    
    def print_processing_summary(self, processing_stats):
        print("\n" + "=" * 60)
        print("Processing Summary:")
        print("=" * 60)
        
        print(f"Total files: {processing_stats['total_files']}")
        print(f"Successfully processed: {processing_stats['processed_files']}")
        print(f"Failed files: {processing_stats['failed_files']}")
        print(f"Noise reduced files: {processing_stats['noise_reduced_files']}")
        print(f"Volume normalized files: {processing_stats['volume_normalized_files']}")
        print(f"Duration adjusted files: {processing_stats['duration_adjusted_files']}")
        
        if processing_stats['total_files'] > 0:
            success_rate = (processing_stats['processed_files'] / processing_stats['total_files']) * 100
            print(f"Success rate: {success_rate:.1f}%")
    
    def analyze_quality_improvements(self, quality_metrics):
        print("\nAnalyzing quality improvements...")
        
        if not quality_metrics:
            print("Warning: No quality data for analysis")
            return
        
        improvements = []
        for metric in quality_metrics:
            if metric['improvement']:
                improvements.append(metric['improvement'])
        
        if improvements:
            avg_snr_improvement = np.mean([imp['snr_improvement'] for imp in improvements])
            avg_energy_change = np.mean([imp['energy_change'] for imp in improvements])
            avg_clipping_reduction = np.mean([imp['clipping_reduction'] for imp in improvements])
            
            print(f"Average SNR improvement: {avg_snr_improvement:.2f} dB")
            print(f"Average energy change: {avg_energy_change:.2f}x")
            print(f"Average clipping reduction: {avg_clipping_reduction:.4f}")
            
            speaker_improvements = {}
            for metric in quality_metrics:
                speaker = metric['speaker']
                if speaker not in speaker_improvements:
                    speaker_improvements[speaker] = []
                if metric['improvement']:
                    speaker_improvements[speaker].append(metric['improvement']['snr_improvement'])
            
            print("\nImprovements by speaker:")
            for speaker, improvements in speaker_improvements.items():
                if improvements:
                    avg_improvement = np.mean(improvements)
                    print(f"  {speaker}: {avg_improvement:.2f} dB")
    
    def create_processed_dataset_csv(self, quality_metrics):
        print("\nCreating processed dataset CSV...")
        
        processed_data = []
        for metric in quality_metrics:
            if metric['final_quality']:
                processed_data.append({
                    'file_path': metric['output_path'],
                    'original_path': metric['file_path'],
                    'word': metric['word'],
                    'speaker': metric['speaker'],
                    'duration': metric['final_quality']['duration'],
                    'rms_energy': metric['final_quality']['rms_energy'],
                    'snr': metric['final_quality']['snr'],
                    'quality_score': self.quality_metrics_calculator.calculate_quality_score(metric['final_quality'])
                })
        
        df = pd.DataFrame(processed_data)
        df.to_csv('processed_dataset.csv', index=False, encoding='utf-8')
        
        print(f"Saved {len(df)} processed samples to: processed_dataset.csv")
        return df
    
    def save_processing_report(self, processing_stats, quality_metrics, settings):
        print("\nSaving processing report...")
        
        report = {
            'processing_stats': processing_stats,
            'quality_metrics': quality_metrics,
            'settings': settings
        }
        
        with open('audio_processing_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print("Report saved to: audio_processing_report.json")