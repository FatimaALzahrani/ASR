import warnings
warnings.filterwarnings('ignore')

from improved_models_trainer import ImprovedModelsTrainer


def main():
    trainer = ImprovedModelsTrainer()
    trainer.run_complete_pipeline()


if __name__ == "__main__":
    main()