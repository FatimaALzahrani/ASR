import pickle
import math
from datetime import datetime
from collections import Counter, defaultdict
from base_model import BaseModel
from data_processor import DataProcessor


class LanguageModel(BaseModel):
    def __init__(self, features_path="features", results_path="advanced_results"):
        super().__init__(features_path, results_path)
        self.data_processor = DataProcessor()
        
        self.vocabulary = set()
        self.word_frequencies = Counter()
        self.bigram_frequencies = Counter()
        self.trigram_frequencies = Counter()
        self.word_probabilities = {}
        self.bigram_probabilities = {}
        self.trigram_probabilities = {}
        self.speaker_vocabularies = {}
        
        self.context_window = 3
        
        print("Language Model initialized")
    
    def load_vocabulary_data(self):
        print("Loading vocabulary data...")
        
        self.df = self.data_processor.load_features(self.features_path)
        
        words = self.df['word'].dropna().astype(str).tolist()
        speakers = self.df['speaker'].dropna().astype(str).tolist()
        
        print(f"Loaded {len(words)} words from {len(set(speakers))} speakers")
        
        self.vocabulary = set(words)
        self.word_frequencies = Counter(words)
        
        print(f"Vocabulary size: {len(self.vocabulary)}")
        print(f"Total words: {len(words)}")
        
        speaker_word_counts = defaultdict(Counter)
        for word, speaker in zip(words, speakers):
            speaker_word_counts[speaker][word] += 1
        
        self.speaker_vocabularies = dict(speaker_word_counts)
        
        print("Speaker distributions:")
        for speaker, word_count in speaker_word_counts.items():
            print(f"  {speaker}: {len(word_count)} unique words, {sum(word_count.values())} total")
        
        return words, speakers
    
    def build_language_model(self, words):
        print("Building language model...")
        
        total_words = len(words)
        self.word_probabilities = {
            word: count / total_words 
            for word, count in self.word_frequencies.items()
        }
        
        print(f"Calculated {len(self.word_probabilities)} unigram probabilities")
        
        bigrams = []
        for i in range(len(words) - 1):
            bigram = (words[i], words[i + 1])
            bigrams.append(bigram)
            self.bigram_frequencies[bigram] += 1
        
        for bigram, count in self.bigram_frequencies.items():
            first_word = bigram[0]
            first_word_count = self.word_frequencies[first_word]
            self.bigram_probabilities[bigram] = count / first_word_count
        
        print(f"Built bigram model with {len(self.bigram_frequencies)} pairs")
        
        trigrams = []
        for i in range(len(words) - 2):
            trigram = (words[i], words[i + 1], words[i + 2])
            trigrams.append(trigram)
            self.trigram_frequencies[trigram] += 1
        
        for trigram, count in self.trigram_frequencies.items():
            bigram_prefix = trigram[:2]
            bigram_count = self.bigram_frequencies[bigram_prefix]
            if bigram_count > 0:
                self.trigram_probabilities[trigram] = count / bigram_count
        
        print(f"Built trigram model with {len(self.trigram_frequencies)} triplets")
        
        language_model_stats = {
            'vocabulary_size': len(self.vocabulary),
            'total_words': total_words,
            'unique_bigrams': len(self.bigram_frequencies),
            'unique_trigrams': len(self.trigram_frequencies),
            'most_common_words': dict(self.word_frequencies.most_common(10)),
            'most_common_bigrams': {f"{k[0]}_{k[1]}": v for k, v in self.bigram_frequencies.most_common(10)},
            'most_common_trigrams': {f"{k[0]}_{k[1]}_{k[2]}": v for k, v in self.trigram_frequencies.most_common(10)}
        }
        
        self.results['language_model_stats'] = language_model_stats
    
    def calculate_word_probability(self, word, context=None):
        unigram_prob = self.word_probabilities.get(word, 1e-6)
        
        if context is None or len(context) == 0:
            return unigram_prob
        
        score = math.log(unigram_prob)
        
        if len(context) >= 1:
            prev_word = context[-1]
            bigram = (prev_word, word)
            bigram_prob = self.bigram_probabilities.get(bigram, 1e-6)
            score += math.log(bigram_prob)
        
        if len(context) >= 2:
            prev_prev_word = context[-2]
            prev_word = context[-1]
            trigram = (prev_prev_word, prev_word, word)
            trigram_prob = self.trigram_probabilities.get(trigram, 1e-6)
            score += math.log(trigram_prob)
        
        return score
    
    def get_word_suggestions(self, context, n=5):
        if not context:
            return list(self.word_frequencies.most_common(n))
        
        candidates = []
        for word in self.vocabulary:
            prob = self.calculate_word_probability(word, context)
            candidates.append((word, prob))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:n]
    
    def apply_language_model_scoring(self, candidates, context):
        scored_candidates = []
        
        for candidate in candidates:
            word = candidate['word']
            base_score = candidate['similarity']
            
            context_score = self.calculate_word_probability(word, context)
            combined_score = (base_score * 0.7) + (context_score * 0.3)
            
            scored_candidates.append({
                **candidate,
                'similarity': combined_score,
                'context_score': context_score
            })
        
        scored_candidates.sort(key=lambda x: x['similarity'], reverse=True)
        return scored_candidates
    
    def apply_speaker_specific_scoring(self, candidates, speaker):
        if speaker not in self.speaker_vocabularies:
            return candidates
        
        speaker_vocab = self.speaker_vocabularies[speaker]
        max_freq = max(speaker_vocab.values()) if speaker_vocab else 1
        
        scored_candidates = []
        
        for candidate in candidates:
            word = candidate['word']
            base_score = candidate['similarity']
            
            speaker_boost = 0.0
            if word in speaker_vocab:
                word_freq = speaker_vocab[word]
                speaker_boost = (word_freq / max_freq) * 0.2
            
            final_score = base_score + speaker_boost
            
            scored_candidates.append({
                **candidate,
                'similarity': final_score,
                'speaker_boost': speaker_boost
            })
        
        scored_candidates.sort(key=lambda x: x['similarity'], reverse=True)
        return scored_candidates
    
    def train(self):
        print("Starting language model training...")
        
        try:
            words, speakers = self.load_vocabulary_data()
            self.build_language_model(words)
            
            self.model = {
                'vocabulary': list(self.vocabulary),
                'word_frequencies': dict(self.word_frequencies),
                'word_probabilities': self.word_probabilities,
                'bigram_probabilities': self.bigram_probabilities,
                'trigram_probabilities': self.trigram_probabilities,
                'speaker_vocabularies': self.speaker_vocabularies
            }
            
            self.is_trained = True
            print("Language model training completed")
            return True
            
        except Exception as e:
            print(f"Training error: {str(e)}")
            return False
    
    def predict(self, context):
        return self.get_word_suggestions(context)
    
    def evaluate(self):
        if 'language_model_stats' in self.results:
            return self.results['language_model_stats']
        return None
    
    def save_model(self, filename=None):
        if not self.is_trained:
            print("Model not trained yet")
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if filename is None:
            filename = f"language_model_{timestamp}.pkl"
        
        filepath = f"{self.results_path}/language_models/{filename}"
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        
        print(f"Language model saved: {filepath}")
        return filepath