from max_data_usage import MaxDataUsageTrainer


def main():
    print("Maximum Data Usage for Small Dataset")
    print("=" * 70)
    
    trainer = MaxDataUsageTrainer()
    results = trainer.run_all_strategies()
    
    print("\nAll strategies completed!")
    print("Use the best strategy for your final project")


if __name__ == "__main__":
    main()