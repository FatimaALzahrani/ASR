import sys
from pathlib import Path
from fair_data_processor import FairDataProcessor
from quality_checker import QualityChecker
from distribution_comparer import DistributionComparer


class ToolRunner:
    def __init__(self):
        self.processor = FairDataProcessor()
        self.checker = QualityChecker()
        self.comparer = DistributionComparer()
    
    def check_prerequisites(self):
        required_files = [
            "data/processed/train.csv",
            "data/processed/validation.csv", 
            "data/processed/test.csv"
        ]
        
        missing_files = []
        for file_path in required_files:
            if not Path(file_path).exists():
                missing_files.append(file_path)
        
        if missing_files:
            print("Error: Missing required files:")
            for file_path in missing_files:
                print(f"  - {file_path}")
            print("\nPlease run data processing first.")
            return False
        
        return True
    
    def run_quality_assessment(self):
        print("Step 1: Initial Quality Assessment")
        print("=" * 40)
        
        # Generate quality report
        report = self.checker.generate_quality_report()
        
        if report and report['fairness_score'] >= 70:
            print(f"\nCurrent data splits are already fair (score: {report['fairness_score']:.1f}/100)")
            return True
        
        print(f"\nData splits need improvement (score: {report['fairness_score']:.1f}/100)")
        return False
    
    def run_fair_splitting(self):
        print("\nStep 2: Fair Data Splitting")
        print("=" * 40)
        
        result = self.processor.run_fair_splitting()
        return result is not None
    
    def run_final_verification(self):
        print("\nStep 3: Final Quality Verification")
        print("=" * 40)
        
        # Check quality after splitting
        report = self.checker.generate_quality_report()
        
        if report:
            print(f"\nFinal Results:")
            print(f"Fairness Score: {report['fairness_score']:.1f}/100")
            print(f"Data Integrity: {'PASS' if report['data_integrity'] else 'FAIL'}")
            
            # Compare distributions
            print("\nDistribution Comparison:")
            self.comparer.compare_distributions()
            
            return report['fairness_score'] >= 70
        
        return False
    
    def run_complete_pipeline(self):
        print("Complete Fair Data Splitting Pipeline")
        print("=" * 50)
        
        # Check prerequisites
        if not self.check_prerequisites():
            return False
        
        # Initial assessment
        is_already_fair = self.run_quality_assessment()
        
        if is_already_fair:
            response = input("\nData appears fair. Continue with splitting anyway? (y/N): ")
            if response.lower() != 'y':
                print("Pipeline cancelled by user.")
                return True
        
        # Run fair splitting
        if not self.run_fair_splitting():
            print("Error: Fair splitting failed")
            return False
        
        # Final verification
        success = self.run_final_verification()
        
        if success:
            print("\n" + "=" * 50)
            print("SUCCESS: Fair splitting completed!")
            print("Your data is now ready for unbiased training.")
            print("Recommendation: Retrain your models with the new fair splits.")
        else:
            print("\n" + "=" * 50)
            print("WARNING: Fair splitting completed but quality issues remain.")
            print("Please review the quality report for details.")
        
        return success


def main():
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "check":
            # Only run quality check
            checker = QualityChecker()
            checker.generate_quality_report()
            
        elif command == "split":
            # Only run fair splitting
            processor = FairDataProcessor()
            processor.run_fair_splitting()
            
        elif command == "compare":
            # Only run comparison
            comparer = DistributionComparer()
            comparer.compare_distributions()
            
        elif command == "help":
            print("Usage: python run_all_tools.py [command]")
            print("Commands:")
            print("  check   - Run quality check only")
            print("  split   - Run fair splitting only")
            print("  compare - Compare distributions only")
            print("  help    - Show this help message")
            print("  (no command) - Run complete pipeline")
            
        else:
            print(f"Unknown command: {command}")
            print("Use 'help' for available commands")
    
    else:
        # Run complete pipeline
        runner = ToolRunner()
        runner.run_complete_pipeline()


if __name__ == "__main__":
    main()