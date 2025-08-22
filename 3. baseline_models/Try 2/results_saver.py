import json
from pathlib import Path
from typing import Dict
from config import Config

class ResultsSaver:
    def __init__(self):
        self.results_dir = Path(Config.RESULTS_DIR)
        self.results_dir.mkdir(exist_ok=True)
        
    def save_results(self, results: Dict):
        with open(self.results_dir / Config.RESULTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        summary = self._create_summary(results)
        
        with open(self.results_dir / Config.SUMMARY_FILE, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"Results saved to: {self.results_dir}")
        
        self._print_summary(summary)
    
    def _create_summary(self, results: Dict) -> Dict:
        summary = {
            "overall_performance": {
                "test_accuracy": results["test_evaluation"]["accuracy"],
                "test_wer": results["test_evaluation"]["wer"],
                "sample_accuracy": results["sample_evaluation"]["accuracy"],
                "sample_wer": results["sample_evaluation"]["wer"]
            },
            "best_speaker": None,
            "worst_speaker": None,
            "best_word": None,
            "worst_word": None
        }
        
        if "speaker_analysis" in results:
            speaker_accuracies = {
                speaker: data["accuracy"] 
                for speaker, data in results["speaker_analysis"].items()
            }
            if speaker_accuracies:
                summary["best_speaker"] = max(speaker_accuracies, key=speaker_accuracies.get)
                summary["worst_speaker"] = min(speaker_accuracies, key=speaker_accuracies.get)
        
        if "word_analysis" in results:
            word_accuracies = {
                word: data["accuracy"] 
                for word, data in results["word_analysis"].items()
            }
            if word_accuracies:
                summary["best_word"] = max(word_accuracies, key=word_accuracies.get)
                summary["worst_word"] = min(word_accuracies, key=word_accuracies.get)
        
        return summary
    
    def _print_summary(self, summary: Dict):
        print("\nResults Summary:")
        print(f"   Test accuracy: {summary['overall_performance']['test_accuracy']*100:.1f}%")
        print(f"   Word Error Rate: {summary['overall_performance']['test_wer']:.3f}")
        if summary["best_speaker"]:
            print(f"   Best speaker: {summary['best_speaker']}")
        if summary["best_word"]:
            print(f"   Best word: {summary['best_word']}")