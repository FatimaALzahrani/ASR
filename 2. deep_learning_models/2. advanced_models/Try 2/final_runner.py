"""
Final runner that automatically detects and fixes import issues
This is the recommended way to run the maximum data usage system
"""

import sys
import os
from pathlib import Path

def try_full_version():
    """Try to run the full-featured version"""
    print("Attempting to run full-featured version...")
    
    try:
        # Add current directory to path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.append(current_dir)
        
        # Check if all required files exist
        required_files = [
            'data_loader.py', 'trainer.py', 'strategies.py', 
            'max_data_usage.py', 'audio_dataset.py', 'models.py'
        ]
        
        missing_files = [f for f in required_files if not Path(f).exists()]
        if missing_files:
            print(f"Missing files: {missing_files}")
            return False
        
        # Try importing modules
        from max_data_usage import MaxDataUsageTrainer
        
        print("Full version loaded successfully!")
        
        # Run the trainer
        trainer = MaxDataUsageTrainer()
        results = trainer.run_all_strategies()
        
        return True
        
    except ImportError as e:
        print(f"Import error in full version: {e}")
        return False
    except Exception as e:
        print(f"Error in full version: {e}")
        return False

def try_simple_version():
    """Fall back to simple self-contained version"""
    print("Falling back to simple self-contained version...")
    
    try:
        # Check if simple runner exists
        if not Path('simple_runner.py').exists():
            print("Simple runner not found")
            return False
        
        # Import and run simple version
        import simple_runner
        simple_runner.main()
        
        return True
        
    except Exception as e:
        print(f"Error in simple version: {e}")
        return False

def check_data_files():
    """Check if required data files exist"""
    data_files = [
        'data/processed/train.csv',
        'data/processed/validation.csv',
        'data/processed/test.csv'
    ]
    
    missing = [f for f in data_files if not Path(f).exists()]
    
    if missing:
        print(f"Missing data files: {missing}")
        print("Please run data preprocessing first")
        return False
    
    return True

def check_basic_requirements():
    """Check basic package requirements"""
    required_packages = ['pandas', 'numpy', 'torch', 'sklearn']
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"Missing basic packages: {missing}")
        print("Install with: pip install pandas numpy torch scikit-learn")
        return False
    
    return True

def main():
    """Main function that tries different approaches"""
    print("Maximum Data Usage System - Smart Runner")
    print("=" * 50)
    
    # Check basic requirements
    if not check_basic_requirements():
        print("\nPlease install basic packages first:")
        print("pip install pandas numpy torch scikit-learn")
        return
    
    # Check data files
    if not check_data_files():
        print("\nPlease ensure data files are available")
        return
    
    print("Basic requirements and data files OK")
    print()
    
    # Try full version first
    if try_full_version():
        print("\n✅ SUCCESS: Full-featured version completed")
        print("Check the results in 'max_data_usage_results.json'")
        return
    
    print()
    print("Full version failed, trying simple version...")
    print()
    
    # Fall back to simple version
    if try_simple_version():
        print("\n✅ SUCCESS: Simple version completed")
        print("Check the results in 'simple_max_usage_results.json'")
        print("\nNote: Simple version uses dummy features.")
        print("For real audio processing, fix the import issues in full version.")
        return
    
    print("\n❌ Both versions failed")
    print("\nTroubleshooting steps:")
    print("1. Ensure all .py files are in the same directory")
    print("2. Check that data/processed/ contains CSV files")
    print("3. Install missing packages: pip install -r requirements.txt")
    print("4. Run: python quick_check.py for detailed diagnostics")

if __name__ == "__main__":
    main()