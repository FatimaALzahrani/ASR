import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score

class AudioAnalyzer:
    def __init__(self):
        pass
        
    def analyze_by_speaker(self, features, labels, speakers):
        print("Analyzing performance by speaker...")
        
        speaker_analysis = {}
        
        for speaker in set(speakers):
            speaker_mask = [s == speaker for s in speakers]
            speaker_features = features[speaker_mask]
            speaker_labels = [labels[i] for i, mask in enumerate(speaker_mask) if mask]
            
            if len(speaker_labels) > 0:
                unique_words = len(set(speaker_labels))
                total_samples = len(speaker_labels)
                
                most_common_word = Counter(speaker_labels).most_common(1)[0][0]
                simple_predictions = [most_common_word] * len(speaker_labels)
                simple_accuracy = accuracy_score(speaker_labels, simple_predictions)
                
                speaker_analysis[speaker] = {
                    'total_samples': total_samples,
                    'unique_words': unique_words,
                    'simple_accuracy': simple_accuracy,
                    'most_common_word': most_common_word
                }
                
                print(f"  {speaker}: {total_samples} samples, {unique_words} words, accuracy: {simple_accuracy:.4f}")
        
        return speaker_analysis
    
    def analyze_by_word(self, labels):
        print("Analyzing performance by word...")
        
        word_analysis = {}
        word_counts = Counter(labels)
        
        for word, count in word_counts.most_common(10):
            word_analysis[word] = {
                'total_samples': count,
                'frequency': count / len(labels)
            }
            
            print(f"  {word}: {count} samples ({count/len(labels)*100:.1f}%)")
        
        return word_analysis