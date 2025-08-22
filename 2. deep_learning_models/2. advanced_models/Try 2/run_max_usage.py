"""
Fixed version of maximum data usage trainer
Handles import issues and provides robust error handling
"""

import json
import torch
import warnings
import sys
import os
from pathlib import Path

warnings.filterwarnings('ignore')

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Check if required files exist
required_files = ['data_loader.py', 'trainer.py', 'strategies.py', 'max_data_usage.py']
missing_files = [f for f in required_files if not Path(f).exists()]

if missing_files:
    print(f"Missing required files: {missing_files}")
    print("Please ensure all files are in the same directory")
    sys.exit(1)

# Check if data files exist
data_files = [
    'data/processed/train.csv',
    'data/processed/validation.csv', 
    'data/processed/test.csv'
]
missing_data = [f for f in data_files if not Path(f).exists()]

if missing_data:
    print(f"Missing data files: {missing_data}")
    print("Please run the data preprocessing first")
    sys.exit(1)

def check_requirements():
    """Check if required packages are installed"""
    required_packages = ['pandas', 'numpy', 'torch', 'sklearn', 'librosa']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing packages: {missing_packages}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    return True

def main():
    print("Maximum Data Usage for Small Dataset - Fixed Version")
    print("=" * 70)
    
    # Check requirements
    if not check_requirements():
        return
    
    try:
        # Import after checking requirements
        from max_data_usage import MaxDataUsageTrainer
        
        trainer = MaxDataUsageTrainer()
        results = trainer.run_all_strategies()
        
        print("\nAll strategies completed successfully!")
        print("Use the best strategy for your final project")
        
        # Print summary
        if results:
            best_result = 0
            best_strategy = "None"
            
            for strategy, result in results.items():
                if isinstance(result, dict) and 'mean' in result:
                    accuracy = result['mean']
                elif isinstance(result, (int, float)):
                    accuracy = result
                else:
                    continue
                    
                if accuracy > best_result:
                    best_result = accuracy
                    best_strategy = strategy
            
            print(f"\nSummary:")
            print(f"Best performing strategy: {best_strategy}")
            print(f"Best accuracy: {best_result:.4f}")
            
            if best_result > 0:
                print("SUCCESS: Training completed with valid results")
            else:
                print("WARNING: All strategies returned 0 accuracy - check data and model")
        
    except Exception as e:
        print(f"Error during training: {e}")
        print("\nTroubleshooting tips:")
        print("1. Ensure all Python files are in the same directory")
        print("2. Check that data files exist in data/processed/")
        print("3. Verify audio files are accessible")
        print("4. Make sure all required packages are installed")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()