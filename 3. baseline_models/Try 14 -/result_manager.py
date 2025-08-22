import json
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime

class ResultManager:
    
    def __init__(self, output_path="results"):
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)
    
    def save_results(self, results, speaker_models, speaker_scalers, speaker_profiles):
        print(f"Saving results...")
        
        try:
            with open(self.output_path / 'results.json', 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            
            models_data = {
                'speaker_models': speaker_models,
                'speaker_scalers': speaker_scalers,
                'speaker_profiles': speaker_profiles,
                'timestamp': datetime.now().isoformat(),
                'version': 'DS_ASR_v1.0'
            }
            
            with open(self.output_path / 'models.pkl', 'wb') as f:
                pickle.dump(models_data, f)
            
            print(f"Results saved to: {self.output_path}")
            
        except Exception as e:
            print(f"Failed to save results: {e}")
    
    def print_report(self, results):
        print(f"\n" + "="*60)
        print(f"FINAL RESULTS REPORT")
        print(f"="*60)
        
        if not results:
            print("No results available!")
            return
        
        sorted_results = sorted(results.items(), key=lambda x: x[1]['best_accuracy'], reverse=True)
        
        print(f"\nSpeaker Results (sorted by accuracy):")
        print("-" * 80)
        print(f"{'Speaker':<12} {'Accuracy':<8} {'Model':<20} {'Samples':<8} {'Words':<8}")
        print("-" * 80)
        
        total_weighted_accuracy = 0
        total_samples = 0
        
        for speaker, result in sorted_results:
            accuracy = result['best_accuracy']
            samples = result['samples']
            words = result['words']
            model = result['best_model']
            
            total_weighted_accuracy += accuracy * samples
            total_samples += samples
            
            performance_icon = "🟢" if accuracy >= 0.75 else "🟡" if accuracy >= 0.60 else "🔴"
            
            print(f"{speaker:<12} {accuracy*100:>6.2f}% {model:<20} {samples:>8d} {words:>8d} {performance_icon}")
        
        overall_accuracy = total_weighted_accuracy / total_samples if total_samples > 0 else 0
        
        print("-" * 80)
        print(f"Overall Weighted Average: {overall_accuracy*100:.2f}%")
        print(f"Total Samples: {total_samples}")
        print(f"Trained Speakers: {len(results)}")
        
        self._print_quality_analysis(results)
        self._print_model_analysis(results)
        self._print_performance_summary(sorted_results)
        self._print_recommendations(sorted_results)
        
        print(f"\n" + "="*60)
        print(f"Analysis completed successfully!")
        print(f"="*60)
    
    def _print_quality_analysis(self, results):
        print(f"\nPerformance by Speaker Quality:")
        
        quality_analysis = {}
        for speaker, result in results.items():
            quality = result['quality']
            if quality not in quality_analysis:
                quality_analysis[quality] = []
            quality_analysis[quality].append(result['best_accuracy'])
        
        for quality, accuracies in quality_analysis.items():
            avg_acc = np.mean(accuracies)
            count = len(accuracies)
            std_acc = np.std(accuracies)
            print(f"   {quality:15}: {avg_acc*100:6.2f}% ± {std_acc*100:4.1f}% ({count} speakers)")
    
    def _print_model_analysis(self, results):
        print(f"\nModel Usage Analysis:")
        
        model_analysis = {}
        for speaker, result in results.items():
            model = result['best_model']
            if model not in model_analysis:
                model_analysis[model] = []
            model_analysis[model].append(result['best_accuracy'])
        
        for model, accuracies in model_analysis.items():
            avg_acc = np.mean(accuracies)
            count = len(accuracies)
            print(f"   {model:20}: {avg_acc*100:6.2f}% average ({count} uses)")
    
    def _print_performance_summary(self, sorted_results):
        if sorted_results:
            best_speaker, best_result = sorted_results[0]
            worst_speaker, worst_result = sorted_results[-1]
            
            print(f"\nBest Performance:")
            print(f"   Speaker: {best_speaker}")
            print(f"   Accuracy: {best_result['best_accuracy']*100:.2f}%")
            print(f"   Model: {best_result['best_model']}")
            print(f"   Samples: {best_result['samples']}")
            
            print(f"\nLowest Performance:")
            print(f"   Speaker: {worst_speaker}")
            print(f"   Accuracy: {worst_result['best_accuracy']*100:.2f}%")
            print(f"   Model: {worst_result['best_model']}")
            print(f"   Samples: {worst_result['samples']}")
            
            performance_gap = (best_result['best_accuracy'] - worst_result['best_accuracy']) * 100
            print(f"\nPerformance Gap: {performance_gap:.2f}%")
    
    def _print_recommendations(self, sorted_results):
        print(f"\nRecommendations:")
        
        performance_ranges = {
            'Excellent (≥80%)': [r for r in [res[1] for res in sorted_results] if r['best_accuracy'] >= 0.80],
            'Good (65-79%)': [r for r in [res[1] for res in sorted_results] if 0.65 <= r['best_accuracy'] < 0.80],
            'Fair (50-64%)': [r for r in [res[1] for res in sorted_results] if 0.50 <= r['best_accuracy'] < 0.65],
            'Needs Improvement (<50%)': [r for r in [res[1] for res in sorted_results] if r['best_accuracy'] < 0.50]
        }
        
        for range_name, performers in performance_ranges.items():
            count = len(performers)
            percentage = (count / len(sorted_results)) * 100 if sorted_results else 0
            print(f"   {range_name}: {count} speakers ({percentage:.1f}%)")
        
        need_improvement = [speaker for speaker, result in sorted_results if result['best_accuracy'] < 0.65]
        if need_improvement:
            print(f"\nSpeakers needing improvement:")
            for speaker in need_improvement[:3]:
                print(f"   - {speaker}: Increase data samples and improve recording quality")
