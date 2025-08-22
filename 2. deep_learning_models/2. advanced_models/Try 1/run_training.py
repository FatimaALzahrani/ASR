import sys
import time
from pathlib import Path
from utils import check_data_files, setup_directories, format_time
from comprehensive_trainer import ComprehensiveTrainer
import config


def main():
    print("Fixed Training System for Down Syndrome Speech Recognition")
    print("=" * 60)
    print(f"Using device: {config.DEVICE}")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        if not check_data_files():
            print("Please ensure all required data files are present")
            sys.exit(1)
        
        setup_directories()
        
        print("Initializing comprehensive trainer...")
        trainer = ComprehensiveTrainer()
        
        print("Starting training pipeline...")
        results = trainer.run_comprehensive_training()
        
        end_time = time.time()
        training_time = end_time - start_time
        
        print(f"\nTraining completed successfully!")
        print(f"Total training time: {format_time(training_time)}")
        
        print(f"\nFinal model rankings:")
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        for i, (model_name, accuracy) in enumerate(sorted_results, 1):
            print(f"  {i}. {model_name}: {accuracy:.4f}")
        
        print(f"\nOutput files generated:")
        for model_name, filepath in config.OUTPUT_CONFIG['model_files'].items():
            if Path(filepath).exists():
                print(f"  {filepath}")
        
        if Path(config.OUTPUT_CONFIG['results_file']).exists():
            print(f"  {config.OUTPUT_CONFIG['results_file']}")
        
        return True
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return False
        
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)