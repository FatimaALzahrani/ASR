import json
from config import Config

class ReportGenerator:
    def __init__(self):
        pass
        
    def generate_insights(self, baseline_results, speaker_analysis):
        insights = []
        
        try:
            if 'knn' in baseline_results and 'random' in baseline_results and baseline_results['random'] > 0:
                knn_vs_random = baseline_results['knn'] / baseline_results['random']
                insights.append(f"k-NN model is {knn_vs_random:.2f}x better than random")
            
            if 'knn' in baseline_results and 'majority' in baseline_results and baseline_results['majority'] > 0:
                knn_vs_majority = baseline_results['knn'] / baseline_results['majority']
                insights.append(f"k-NN model is {knn_vs_majority:.2f}x better than majority")
            
            if speaker_analysis:
                speaker_scores = [(speaker, data['simple_accuracy']) 
                                 for speaker, data in speaker_analysis.items() 
                                 if 'simple_accuracy' in data]
                
                if speaker_scores:
                    speaker_scores.sort(key=lambda x: x[1], reverse=True)
                    
                    best_speaker = speaker_scores[0]
                    worst_speaker = speaker_scores[-1]
                    
                    insights.append(f"Best speaker: {best_speaker[0]} ({best_speaker[1]:.4f})")
                    insights.append(f"Worst speaker: {worst_speaker[0]} ({worst_speaker[1]:.4f})")
                    
                    total_samples = sum(data.get('total_samples', 0) for data in speaker_analysis.values())
                    if len(speaker_analysis) > 0:
                        avg_samples = total_samples / len(speaker_analysis)
                        insights.append(f"Average samples per speaker: {avg_samples:.1f}")
        
        except Exception as e:
            print(f"Warning: Error generating insights: {e}")
            insights.append("Could not generate detailed insights due to data processing error")
        
        return insights
    
    def generate_comprehensive_report(self, dataset_info, baseline_results, speaker_analysis, word_analysis):
        print("Generating comprehensive report...")
        
        try:
            report = {
                'dataset_info': dataset_info,
                'baseline_results': baseline_results,
                'speaker_analysis': speaker_analysis,
                'word_analysis': word_analysis,
                'insights': self.generate_insights(baseline_results, speaker_analysis)
            }
            
            with open(Config.OUTPUT_REPORT, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            print(f"Report saved to: {Config.OUTPUT_REPORT}")
            return report
            
        except Exception as e:
            print(f"Error generating report: {e}")
            
            basic_report = {
                'dataset_info': dataset_info,
                'baseline_results': baseline_results,
                'error': str(e)
            }
            
            try:
                with open(Config.OUTPUT_REPORT, 'w', encoding='utf-8') as f:
                    json.dump(basic_report, f, ensure_ascii=False, indent=2)
                print(f"Basic report saved to: {Config.OUTPUT_REPORT}")
            except:
                print("Could not save report to file")
            
            return basic_report
    
    def print_final_results(self, baseline_results, speaker_analysis):
        try:
            print("\n" + "=" * 60)
            print("Evaluation completed successfully!")
            
            if 'random' in baseline_results:
                print(f"Random accuracy: {baseline_results['random']:.4f}")
            if 'majority' in baseline_results:
                print(f"Majority accuracy: {baseline_results['majority']:.4f}")
            if 'knn' in baseline_results:
                print(f"k-NN accuracy: {baseline_results['knn']:.4f}")
            
            if speaker_analysis:
                print("\nTop speakers by performance:")
                speaker_scores = [(speaker, data.get('simple_accuracy', 0)) 
                                 for speaker, data in speaker_analysis.items()]
                speaker_scores.sort(key=lambda x: x[1], reverse=True)
                
                for i, (speaker, score) in enumerate(speaker_scores, 1):
                    print(f"{i}. {speaker}: {score:.4f}")
                    
            print("\nGenerated files:")
            print(f"• {Config.OUTPUT_REPORT} - Comprehensive evaluation report")
            
        except Exception as e:
            print(f"Error printing final results: {e}")
            print("Evaluation completed but there was an issue displaying results.")