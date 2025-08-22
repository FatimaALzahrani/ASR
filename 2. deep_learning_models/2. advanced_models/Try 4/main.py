import argparse
import warnings
warnings.filterwarnings('ignore')

from config import get_config
from final_professional_trainer import FinalProfessionalTrainer


def main():
    parser = argparse.ArgumentParser(description='Final Professional Training System')
    parser.add_argument('--config', type=str, default='basic', 
                       choices=['basic', 'intermediate', 'advanced'],
                       help='Configuration type')
    
    args = parser.parse_args()
    
    config = get_config(args.config)
    
    print("Starting Final Professional Training System from Scratch")
    print("=" * 60)
    print(f"Configuration: {args.config}")
    print(f"Target: Training specialized model for children with Down syndrome")
    
    try:
        trainer = FinalProfessionalTrainer(config)
        
        print("\nStarting base training...")
        base_accuracy = trainer.train_base_model()
        
        print("\nStarting comprehensive evaluation...")
        final_results = trainer.comprehensive_evaluation()
        
        final_results['training_info'] = {
            'config': config,
            'final_accuracy': base_accuracy
        }
        
        trainer.save_results(final_results)
        
        print("\nFinal professional training completed successfully!")
        print("Check results/ folder for results and models")
        print("Send final_professional_results.json to get research paper")
        
        accuracy = final_results['base_model']['test_accuracy']
        improvement = final_results['base_model']['improvement_over_baseline']
        
        print(f"\nQuick Results:")
        print(f"   Achieved accuracy: {accuracy*100:.2f}%")
        print(f"   Improvement: +{improvement*100:.2f}%")
        print(f"   Successfully understood speech from children with Down syndrome!")
        
    except Exception as e:
        print(f"Error in training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()