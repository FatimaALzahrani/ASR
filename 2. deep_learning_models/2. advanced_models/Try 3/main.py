import warnings
warnings.filterwarnings('ignore')

from enhanced_models_trainer import EnhancedModelsTrainer


def main():
    print("Enhanced Speech Recognition Models Trainer")
    print("=" * 80)
    
    trainer = EnhancedModelsTrainer()
    results = trainer.run_comprehensive_comparison()
    
    print("\nEnhanced comparison completed successfully!")
    print("Generated files:")
    print("• best_hmm_dnn_enhanced.pth - Enhanced HMM-DNN model")
    print("• best_rnn_cnn_enhanced.pth - Enhanced RNN-CNN model") 
    print("• best_end_to_end_enhanced.pth - Enhanced End-to-End model")
    print("• enhanced_models_comparison.json - Comprehensive models comparison")
    print("• *_training_curves.png - Training curves for each model")


if __name__ == "__main__":
    main()