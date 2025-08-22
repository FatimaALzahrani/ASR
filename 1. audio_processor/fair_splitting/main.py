from fair_data_processor import FairDataProcessor
from distribution_comparer import DistributionComparer


def main():
    print("Fair Data Splitting Tool")
    print("=" * 30)
    
    processor = FairDataProcessor()
    comparer = DistributionComparer()
    
    # Evaluate current fairness
    print("Step 1: Evaluating current data distribution")
    processor.evaluate_fairness()
    
    # Run fair splitting
    print("\nStep 2: Creating fair data splits")
    result = processor.run_fair_splitting()
    
    if result is None:
        print("Error: Could not complete fair splitting")
        return
    
    # Compare distributions
    print("\nStep 3: Comparing old vs new distributions")
    comparer.compare_distributions()
    
    print(f"\nRecommendation:")
    print(f"Retrain your models now to get fair and accurate results!")


if __name__ == "__main__":
    main()