import os
from asr_system import UltimateASRSystem
from config import Config


def main():
    print("ASR System for Down Syndrome")
    print("=" * 50)
    
    data_path = input("Enter data path (or press Enter for default): ").strip()
    
    if not data_path:
        for path in Config.DEFAULT_DATA_PATHS:
            if os.path.exists(path):
                data_path = path
                break
        
        if not data_path:
            print("No valid data path found!")
            data_path = input("Enter correct data path: ").strip()
    
    if not os.path.exists(data_path):
        print(f"Path not found: {data_path}")
        return
    
    print(f"Using data path: {data_path}")
    
    asr_system = UltimateASRSystem(data_path)
    
    try:
        output_path = asr_system.run_complete_pipeline()
        
        if output_path:
            print(f"\nSystem completed successfully!")
            print(f"Output saved to: {output_path}")
            
            test_choice = input("\nTest prediction on audio file? (y/n): ")
            if test_choice.lower() == 'y':
                test_file = input("Enter audio file path: ").strip()
                if os.path.exists(test_file):
                    speaker = input("Enter speaker name (optional): ").strip() or None
                    
                    result = asr_system.predict_audio(test_file, speaker)
                    if result:
                        print(f"\nPrediction result:")
                        print(f"Predicted word: {result['predicted_word']}")
                        print(f"Confidence: {result['confidence']:.2%}")
                        print(f"Model used: {result['model_used']}")
                        print(f"Top 3 predictions:")
                        for i, (word, prob) in enumerate(result['top_3'], 1):
                            print(f"  {i}. {word}: {prob:.2%}")
                else:
                    print(f"File not found: {test_file}")
        
    except Exception as e:
        print(f"Error running system: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()