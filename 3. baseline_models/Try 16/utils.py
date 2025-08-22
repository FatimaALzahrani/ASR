import warnings
warnings.filterwarnings('ignore')

from config import SPEAKER_MAPPING

def get_speaker_from_filename(filename):
    try:
        file_num = int(filename.split('.')[0])
        for num_range, speaker in SPEAKER_MAPPING.items():
            if file_num in num_range:
                return speaker
    except:
        pass
    return "Unknown"

def print_results_summary(final_results):
    dataset_info = final_results['dataset_info']
    general_results = final_results['general_models']
    speaker_results = final_results['speaker_models']
    
    print("Dataset Information:")
    print(f"Samples: {dataset_info['samples']}")
    print(f"Features: {dataset_info['features']}")
    print(f"Words: {dataset_info['words']}")
    print(f"Speakers: {dataset_info['speakers']}")
    
    print("\nGeneral Models Results:")
    for model, acc in sorted(general_results.items(), key=lambda x: x[1], reverse=True):
        print(f"{model}: {acc*100:.2f}%")
    
    print("\nSpeaker-Specific Models Results:")
    for speaker, acc in sorted(speaker_results.items(), key=lambda x: x[1], reverse=True):
        print(f"{speaker}: {acc*100:.2f}%")
    
    best_general = max(general_results.values()) if general_results else 0
    best_speaker = max(speaker_results.values()) if speaker_results else 0
    ultimate_best = max(best_general, best_speaker)
    
    print(f"\nFinal Results:")
    print(f"Best General Model: {best_general*100:.2f}%")
    print(f"Best Speaker Model: {best_speaker*100:.2f}%")
    print(f"Highest Accuracy: {ultimate_best*100:.2f}%")
    
    if ultimate_best >= 0.80:
        print("TARGET ACHIEVED: 80%+ ACCURACY!")
    elif ultimate_best >= 0.75:
        print("EXCELLENT: 75%+ ACCURACY")
    elif ultimate_best >= 0.70:
        print("GREAT: 70%+ ACCURACY")
    elif ultimate_best >= 0.65:
        print("VERY GOOD: 65%+ ACCURACY")
    else:
        print(f"GOOD PROGRESS: {ultimate_best*100:.2f}% ACCURACY")
