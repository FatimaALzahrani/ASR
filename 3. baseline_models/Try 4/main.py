import warnings
warnings.filterwarnings('ignore')

from rapid_trainer import RapidTrainer


def main():
    print("=" * 60)
    
    trainer = RapidTrainer()
    results = trainer.run_rapid_training()
    
    print("\nTraining completed successfully!")
    print("Generated files:")
    print("• best_basic_model.pth - Best basic model")
    print("• best_personalized_model.pth - Best personalized model")
    print("• detailed_results_basic.json - Basic model results")
    print("• detailed_results_personalized.json - Personalized model results")
    print("• rapid_training_summary.json - Training summary")


if __name__ == "__main__":
    main()