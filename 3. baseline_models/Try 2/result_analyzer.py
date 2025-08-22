import pandas as pd
from typing import Dict
from whisper_evaluator import WhisperEvaluator
from config import Config

class ResultAnalyzer:
    def __init__(self, evaluator: WhisperEvaluator):
        self.evaluator = evaluator
        
    def analyze_by_speaker(self, data: pd.DataFrame) -> Dict:
        print("Analyzing performance by speaker...")
        
        speaker_results = {}
        
        for speaker in data['speaker'].unique():
            speaker_data = data[data['speaker'] == speaker]
            
            if len(speaker_data) == 0:
                continue
            
            print(f"\nEvaluating speaker: {speaker}")
            results = self.evaluator.evaluate_sample(speaker_data, f"Speaker {speaker}")
            
            results["speaker"] = speaker
            results["quality"] = speaker_data['quality'].iloc[0] if 'quality' in speaker_data.columns else 'Unknown'
            results["num_words"] = len(speaker_data['text'].unique())
            
            speaker_results[speaker] = results
        
        return speaker_results
    
    def analyze_by_word(self, data: pd.DataFrame) -> Dict:
        print("Analyzing performance by word...")
        
        word_results = {}
        
        top_words = data['text'].value_counts().head(Config.TOP_WORDS_COUNT)
        
        for word in top_words.index:
            word_data = data[data['text'] == word]
            
            if len(word_data) < Config.MIN_WORD_SAMPLES:
                continue
            
            print(f"\nEvaluating word: {word}")
            results = self.evaluator.evaluate_sample(word_data, f"Word {word}")
            
            results["word"] = word
            results["num_speakers"] = len(word_data['speaker'].unique())
            
            word_results[word] = results
        
        return word_results