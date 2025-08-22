import json
import torch
import warnings
warnings.filterwarnings('ignore')

from data_loader import DataLoader
from trainer import ModelTrainer
from strategies import TrainingStrategies


class MaxDataUsageTrainer:
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_loader = DataLoader()
        self.results = {}
        
        self.data_loader.load_all_data()
        self.data_loader.setup_encoders()
        
        all_data = self.data_loader.get_data()
        label_encoder, word_to_id = self.data_loader.get_encoders()
        
        self.trainer = ModelTrainer(self.device, label_encoder, word_to_id)
        self.trainer.set_all_data(all_data)
        self.strategies = TrainingStrategies(all_data, self.trainer)
    
    def run_all_strategies(self):
        print("Maximum Data Usage Strategies Comparison")
        print("=" * 80)
        
        results = {}
        
        try:
            acc1 = self.strategies.strategy_1_simple_split()
            results['Simple_90_10'] = acc1
            
            mean_acc2, std_acc2 = self.strategies.strategy_2_cross_validation()
            results['Cross_Validation'] = {'mean': mean_acc2, 'std': std_acc2}
            
            mean_acc3, speaker_acc3 = self.strategies.strategy_3_leave_one_speaker_out()
            results['LOSO'] = {'mean': mean_acc3, 'speaker_results': speaker_acc3}
            
            acc4 = self.strategies.strategy_4_full_data_with_regularization()
            results['Full_Data'] = acc4
            
        except Exception as e:
            print(f"Strategy error: {e}")
        
        print("\n" + "=" * 80)
        print("All Strategies Comparison:")
        print("=" * 80)
        
        print(f"1. Simple 90/10 Split: {results.get('Simple_90_10', 0):.4f}")
        
        if 'Cross_Validation' in results:
            cv_result = results['Cross_Validation']
            print(f"2. Cross Validation: {cv_result['mean']:.4f} ± {cv_result['std']:.4f}")
        
        if 'LOSO' in results:
            loso_result = results['LOSO']
            print(f"3. Leave-One-Speaker-Out: {loso_result['mean']:.4f}")
        
        print(f"4. Full Data + Regularization: {results.get('Full_Data', 0):.4f}")
        
        with open('max_data_usage_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"\nResults saved to: max_data_usage_results.json")
        
        best_strategy = max([
            ('Simple_90_10', results.get('Simple_90_10', 0)),
            ('Cross_Validation', results.get('Cross_Validation', {}).get('mean', 0)),
            ('LOSO', results.get('LOSO', {}).get('mean', 0)),
            ('Full_Data', results.get('Full_Data', 0))
        ], key=lambda x: x[1])
        
        print(f"\nBest strategy: {best_strategy[0]} ({best_strategy[1]:.4f})")
        
        return results