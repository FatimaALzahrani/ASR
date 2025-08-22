import numpy as np
from collections import Counter
from difflib import SequenceMatcher, get_close_matches
from typing import List, Dict, Optional

class AdvancedLanguageCorrector:
    def __init__(self, vocabulary: Optional[List[str]] = None):
        self.vocabulary = set(vocabulary) if vocabulary else set()
        self.word_frequencies = Counter()
        
        self.phonetic_patterns = {
            'down_syndrome_patterns': {
                'substitution': {
                    'ر': 'ل', 'ث': 'س', 'ذ': 'ز', 'ظ': 'ز', 'ض': 'د', 'ق': 'ك',
                    'غ': 'ك', 'خ': 'ح', 'ع': 'ا', 'ط': 'ت', 'ص': 'س'
                },
                'deletion': ['ة', 'ت', 'ن', 'ه', 'ء'],
                'repetition': ['ل', 'ر', 'م', 'ن', 'ب']
            }
        }
        
        self.common_errors = {}
        self.correction_statistics = {
            'total_corrections': 0,
            'successful_corrections': 0,
            'confidence_scores': []
        }
    
    def train_on_vocabulary(self, words: List[str]):
        self.vocabulary = set(words)
        self.word_frequencies = Counter(words)
        self.generate_common_errors()
    
    def generate_common_errors(self):
        self.common_errors = {}
        
        for word in self.vocabulary:
            word_errors = set()
            
            for original, replacement in self.phonetic_patterns['down_syndrome_patterns']['substitution'].items():
                if original in word:
                    error_word = word.replace(original, replacement)
                    word_errors.add(error_word)
            
            for char in self.phonetic_patterns['down_syndrome_patterns']['deletion']:
                if word.endswith(char) and len(word) > 2:
                    error_word = word[:-1]
                    word_errors.add(error_word)
            
            for char in self.phonetic_patterns['down_syndrome_patterns']['repetition']:
                if char in word:
                    idx = word.find(char)
                    if idx != -1:
                        error_word = word[:idx+1] + char + word[idx+1:]
                        word_errors.add(error_word)
            
            if word_errors:
                self.common_errors[word] = list(word_errors)
    
    def correct_word(self, word: str, top_n: int = 3) -> Dict:
        self.correction_statistics['total_corrections'] += 1
        
        if word in self.vocabulary:
            return {
                'original': word,
                'corrected': word,
                'confidence': 1.0,
                'method': 'exact_match'
            }
        
        candidates = []
        
        for correct_word, error_list in self.common_errors.items():
            if word in error_list:
                candidates.append({
                    'word': correct_word,
                    'similarity': 0.95,
                    'frequency': self.word_frequencies.get(correct_word, 0)
                })
        
        close_matches = get_close_matches(word, self.vocabulary, n=top_n*2, cutoff=0.5)
        
        for match in close_matches:
            similarity = SequenceMatcher(None, word, match).ratio()
            candidates.append({
                'word': match,
                'similarity': similarity,
                'frequency': self.word_frequencies.get(match, 0)
            })
        
        if candidates:
            best_candidate = max(candidates, key=lambda x: (x['similarity'], x['frequency']))
            self.correction_statistics['confidence_scores'].append(best_candidate['similarity'])
            
            if best_candidate['similarity'] > 0.7:
                self.correction_statistics['successful_corrections'] += 1
            
            return {
                'original': word,
                'corrected': best_candidate['word'],
                'confidence': best_candidate['similarity'],
                'method': 'similarity_match'
            }
        
        return {
            'original': word,
            'corrected': word,
            'confidence': 0.0,
            'method': 'no_correction'
        }
    
    def simulate_realistic_errors(self, predictions: List[str], error_rate: float = 0.15) -> List[str]:
        simulated_errors = []
        
        for word in predictions:
            if np.random.random() < error_rate:
                if word in self.common_errors and self.common_errors[word]:
                    error_word = np.random.choice(self.common_errors[word])
                    simulated_errors.append(error_word)
                else:
                    simulated_errors.append(word)
            else:
                simulated_errors.append(word)
        
        return simulated_errors