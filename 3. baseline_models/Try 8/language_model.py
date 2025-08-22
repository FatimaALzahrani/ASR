import re
import pickle
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class LanguageModel:
    def __init__(self, n_gram: int = 3):
        self.n_gram = n_gram
        self.word_counts = Counter()
        self.ngram_counts = defaultdict(Counter)
        self.vocabulary = set()
        self.word_probabilities = {}
        
        logger.info(f"Language model initialized with {n_gram}-gram")
    
    def train(self, texts: List[str]):
        logger.info("Training language model")
        
        for text in texts:
            words = self.tokenize_arabic(text)
            
            for word in words:
                self.vocabulary.add(word)
                self.word_counts[word] += 1
            
            for i in range(len(words) - self.n_gram + 1):
                context = tuple(words[i:i + self.n_gram - 1])
                next_word = words[i + self.n_gram - 1]
                self.ngram_counts[context][next_word] += 1
        
        total_words = sum(self.word_counts.values())
        for word, count in self.word_counts.items():
            self.word_probabilities[word] = count / total_words
        
        logger.info(f"Language model trained on {len(self.vocabulary)} unique words")
    
    def tokenize_arabic(self, text: str) -> List[str]:
        words = re.findall(r'[\u0600-\u06FF]+', text)
        return [word.strip() for word in words if word.strip()]
    
    def get_word_probability(self, word: str) -> float:
        return self.word_probabilities.get(word, 1e-6)
    
    def get_next_word_probabilities(self, context: List[str]) -> Dict[str, float]:
        if len(context) >= self.n_gram - 1:
            context_tuple = tuple(context[-(self.n_gram - 1):])
        else:
            context_tuple = tuple(context)
        
        if context_tuple in self.ngram_counts:
            total_count = sum(self.ngram_counts[context_tuple].values())
            return {word: count / total_count 
                   for word, count in self.ngram_counts[context_tuple].items()}
        else:
            return self.word_probabilities
    
    def suggest_corrections(self, word: str, candidates: List[str]) -> List[Tuple[str, float]]:
        scored_candidates = []
        
        for candidate in candidates:
            probability = self.get_word_probability(candidate)
            edit_distance = self.calculate_edit_distance(word, candidate)
            
            score = probability * (1.0 / (1.0 + edit_distance))
            scored_candidates.append((candidate, score))
        
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        return scored_candidates
    
    def calculate_edit_distance(self, word1: str, word2: str) -> int:
        m, n = len(word1), len(word2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        
        return dp[m][n]
    
    def save_model(self, model_path: str):
        model_data = {
            'n_gram': self.n_gram,
            'word_counts': dict(self.word_counts),
            'ngram_counts': {k: dict(v) for k, v in self.ngram_counts.items()},
            'vocabulary': list(self.vocabulary),
            'word_probabilities': self.word_probabilities
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Language model saved to: {model_path}")
    
    def load_model(self, model_path: str):
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.n_gram = model_data['n_gram']
        self.word_counts = Counter(model_data['word_counts'])
        self.ngram_counts = defaultdict(Counter)
        for k, v in model_data['ngram_counts'].items():
            self.ngram_counts[k] = Counter(v)
        self.vocabulary = set(model_data['vocabulary'])
        self.word_probabilities = model_data['word_probabilities']
        
        logger.info(f"Language model loaded from: {model_path}")
