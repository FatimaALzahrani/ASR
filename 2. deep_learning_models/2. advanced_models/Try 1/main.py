import torch
import warnings
from comprehensive_trainer import ComprehensiveTrainer

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def main():
    print("Complete Fixed Training System for Down Syndrome Children Speech Recognition")
    print("=" * 70)
    
    try:
        trainer = ComprehensiveTrainer()
        results = trainer.run_comprehensive_training()
        
        print("\nTraining completed successfully!")
        
    except Exception as e:
        print(f"\nTraining error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()