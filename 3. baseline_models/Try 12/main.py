import os
from asr_system import RealisticASRSystem

def main():
    print("Realistic ASR System for Down Syndrome (Anti-Overfitting)")
    print("=" * 70)
    
    data_path = input("Enter data folder path: ").strip()
    
    if not data_path:
        data_path = "C:/Users/فاطمة الزهراني/Desktop/ابحاث/الداون/Data/clean"
    
    if not os.path.exists(data_path):
        print(f"Path does not exist: {data_path}")
        return
    
    print(f"Using path: {data_path}")
    
    realistic_asr = RealisticASRSystem(data_path)
    
    try:
        output_path = realistic_asr.run_realistic_pipeline()
        
        if output_path:
            print(f"\nRealistic system completed successfully!")
            print(f"Results saved to: {output_path}")
            
            test_choice = input("\nDo you want to test prediction? (y/n): ")
            if test_choice.lower() == 'y':
                test_file = input("Audio file path: ").strip()
                if os.path.exists(test_file):
                    speaker = input("Speaker name (optional): ").strip() or None
                    
                    result = realistic_asr.predict_with_confidence(test_file, speaker)
                    if result:
                        print(f"\nRealistic prediction result:")
                        print(f"   Word: {result['predicted_word']}")
                        print(f"   Confidence: {result['confidence']:.2%}")
                        print(f"   Reliability: {result['prediction_reliability']}")
                        print(f"   Model: {result['model_used']}")
                        if result['warning']:
                            print(f"   {result['warning']}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()