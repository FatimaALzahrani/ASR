import warnings
import argparse
from pipeline import TrainingPipeline

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train speech recognition models'
    )
    
    parser.add_argument(
        '--general-epochs', 
        type=int, 
        default=60, 
        help='Number of epochs for general model training (default: 60)'
    )
    
    parser.add_argument(
        '--personalized-epochs', 
        type=int, 
        default=60, 
        help='Number of epochs for personalized models training (default: 60)'
    )
    
    parser.add_argument(
        '--general-only', 
        action='store_true', 
        help='Train only the general model'
    )
    
    parser.add_argument(
        '--personalized-only', 
        action='store_true', 
        help='Train only personalized models'
    )
    
    return parser.parse_args()


def main():
    """Main function to run the training pipeline"""
    args = parse_arguments()
    
    # Initialize pipeline
    pipeline = TrainingPipeline()
    
    print("Speech Recognition Training Pipeline")
    print("=" * 50)
    print(f"General model epochs: {args.general_epochs}")
    print(f"Personalized model epochs: {args.personalized_epochs}")
    print("=" * 50)
    
    try:
        if args.general_only:
            # Train only general model
            pipeline.train_general_model(args.general_epochs)
            
        elif args.personalized_only:
            # Train only personalized models
            pipeline.train_personalized_models(args.personalized_epochs)
            
        else:
            # Train all models (default)
            pipeline.run_full_training(
                args.general_epochs, 
                args.personalized_epochs
            )
        
        # Show final results
        pipeline.print_results_summary()
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise
    
    finally:
        print("Training session ended.")


if __name__ == "__main__":
    main()