from config import *

class Evaluator:
    def __init__(self):
        self.results = {}
    
    def evaluate_all_models(self, speaker_models=None, global_models=None):
        print("Evaluating all models...")
        
        all_results = {}
        
        if speaker_models:
            print("Speaker-specific model results:")
            for speaker, models in speaker_models.items():
                print(f"{speaker}:")
                speaker_results = {}
                for model_name, model_data in models.items():
                    accuracy = model_data['accuracy']
                    speaker_results[model_name] = accuracy
                    print(f"  {model_name}: {accuracy:.4f} ({accuracy*100:.2f}%)")
                all_results[f"Speaker_{speaker}"] = speaker_results
        
        if global_models:
            print("Global model results:")
            global_results = {}
            for model_name, model_data in global_models.items():
                accuracy = model_data['accuracy']
                global_results[model_name] = accuracy
                print(f"  {model_name}: {accuracy:.4f} ({accuracy*100:.2f}%)")
            all_results["Global_Models"] = global_results
        
        self.results = all_results
        return all_results
    
    def save_results(self, output_dir='output'):
        os.makedirs(output_dir, exist_ok=True)
        
        results_file = os.path.join(output_dir, 'final_speaker_specific_results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        summary_data = []
        
        for category, models in self.results.items():
            for model_name, accuracy in models.items():
                summary_data.append({
                    'Category': category,
                    'Model': model_name,
                    'Accuracy': f"{accuracy:.4f}",
                    'Accuracy_Percent': f"{accuracy*100:.2f}%"
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = os.path.join(output_dir, 'final_speaker_specific_summary.csv')
        summary_df.to_csv(summary_file, index=False, encoding='utf-8')
        
        print(f"Results saved to: {output_dir}")
        print(f"Detailed results: {results_file}")
        print(f"Summary table: {summary_file}")
        
        return results_file, summary_file