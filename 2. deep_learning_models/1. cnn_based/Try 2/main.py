import os
import sys
from pathlib import Path


def check_requirements():
    required_dirs = [
        'C:/Users/فاطمة الزهراني/Desktop/ابحاث/الداون/Data/clean',
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print("Error: Missing required directories:")
        for dir_path in missing_dirs:
            print(f"  - {dir_path}")
        return False
    
    return True


def run_data_processing():
    print("=" * 60)
    print("STEP 1: DATA PROCESSING")
    print("=" * 60)
    
    try:
        from main_data_processing import main as process_main
        process_main()
        return True
    except Exception as e:
        print(f"Error in data processing: {e}")
        return False


def run_training():
    print("\n" + "=" * 60)
    print("STEP 2: MODEL TRAINING")
    print("=" * 60)
    
    try:
        from main_training import main as training_main
        training_main()
        return True
    except Exception as e:
        print(f"Error in training: {e}")
        return False


def main():
    print("Complete Speech Recognition Pipeline for Down Syndrome Children")
    print("=" * 70)
    
    # Check requirements
    if not check_requirements():
        print("\nPlease ensure the required directories exist and try again.")
        return
    
    # Create output directories
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Run data processing
    if not run_data_processing():
        print("Data processing failed. Stopping pipeline.")
        return
    
    # Run training
    if not run_training():
        print("Training failed. Check the errors above.")
        return
    
    print("\n🎉 Finishing!")


if __name__ == "__main__":
    main()