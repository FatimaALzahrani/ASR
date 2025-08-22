import json
import numpy as np
from pathlib import Path


class EnhancementReportGenerator:
    def __init__(self, reports_dir):
        self.reports_dir = Path(reports_dir)
    
    def generate_enhancement_report(self, enhancement_stats, down_syndrome_params):
        print("\nGenerating enhancement report...")
        
        stats = enhancement_stats
        
        if stats['quality_improvements']:
            improvements = stats['quality_improvements']
            avg_snr_improvement = np.mean([imp['snr_improvement'] for imp in improvements])
            avg_rms_improvement = np.mean([imp['rms_improvement'] for imp in improvements])
            avg_clipping_reduction = np.mean([imp['clipping_reduction'] for imp in improvements])
        else:
            avg_snr_improvement = avg_rms_improvement = avg_clipping_reduction = 0.0
        
        report = {
            'processing_summary': {
                'total_files_processed': stats['total_processed'],
                'microphone_recordings': stats['mic_recordings_processed'],
                'computer_recordings': stats['computer_recordings_processed'],
                'success_rate': f"{(stats['total_processed'] / max(1, stats['total_processed'])) * 100:.1f}%"
            },
            'enhancement_techniques_applied': {
                'noise_reduction': stats['noise_reduced'],
                'volume_enhancement': stats['volume_enhanced'],
                'articulation_improvement': stats['articulation_improved']
            },
            'quality_improvements': {
                'average_snr_improvement_db': round(avg_snr_improvement, 2),
                'average_volume_improvement': round(avg_rms_improvement, 2),
                'average_clipping_reduction': round(avg_clipping_reduction, 4)
            },
            'down_syndrome_specific_enhancements': {
                'low_frequency_emphasis': down_syndrome_params['low_freq_emphasis'],
                'gentle_normalization': down_syndrome_params['gentle_normalization'],
                'articulation_enhancement': down_syndrome_params['articulation_enhancement'],
                'breathing_noise_reduction': down_syndrome_params['breathing_noise_reduction']
            }
        }
        
        report_path = self.reports_dir / 'audio_enhancement_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print("=" * 60)
        print("Enhancement Summary:")
        print("=" * 60)
        print(f"Files processed: {stats['total_processed']}")
        print(f"Microphone recordings: {stats['mic_recordings_processed']}")
        print(f"Computer recordings: {stats['computer_recordings_processed']}")
        print(f"Noise improvement: {avg_snr_improvement:.1f} dB")
        print(f"Volume improvement: {avg_rms_improvement:.2f}x")
        print(f"Clipping reduction: {avg_clipping_reduction:.4f}")
        print(f"Detailed report: {report_path}")
        
        return report