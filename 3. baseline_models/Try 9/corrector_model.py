import pickle
from datetime import datetime
from collections import defaultdict, Counter
from difflib import SequenceMatcher, get_close_matches
from itertools import combinations
from base_model import BaseModel
from data_processor import DataProcessor


class AutoCorrector(BaseModel):
    def __init__(self, features_path="features", results_path="advanced_results"):
        super().__init__(features_path, results_path)
        self.data_processor = DataProcessor()
        
        self.vocabulary = set()
        self.phonetic_patterns = {}
        self.common_errors = {}
        self.speaker_specific_patterns = {}
        self.confusion_matrix = defaultdict(Counter)
        
        self.max_edit_distance = 2
        self.similarity_threshold = 0.6
        self.context_window = 3
        
        print("AutoCorrector initialized")
    
    def analyze_phonetic_patterns(self):
        print("Analyzing phonetic patterns...")
        
        arabic_phonetic_groups = {
            'similar_sounds': {
                't_t': ['ت', 'ط'],
                'd_d': ['د', 'ض'],
                's_s': ['س', 'ص'],
                'z_z': ['ز', 'ظ'],
                'k_q': ['ك', 'ق'],
                'b_m': ['ب', 'م'],
                'f_th': ['ف', 'ث'],
                'j_sh': ['ج', 'ش'],
                'h_kh': ['ح', 'خ'],
                'a_gh': ['ع', 'غ']
            },
            'vowels': ['ا', 'و', 'ي', 'ة', 'ى'],
            'difficult_sounds': ['ث', 'ذ', 'ظ', 'غ', 'ض', 'ق']
        }
        
        down_syndrome_patterns = {
            'substitutions': {
                'ث': 'س',  
                'ذ': 'د',  
                'ظ': 'ز',  
                'ض': 'د',  
                'ق': 'ك',  
                'غ': 'ع',  
            },
            'deletions': {
                'word_end': ['ة', 'ت', 'ن'],
                'word_middle': ['ء', 'ه']
            },
            'insertions': {
                'repeated_chars': ['ل', 'ر', 'م', 'ن']
            }
        }
        
        self.phonetic_patterns = {
            'arabic_groups': arabic_phonetic_groups,
            'down_syndrome_patterns': down_syndrome_patterns
        }
        
        print("Phonetic patterns analyzed")
    
    def generate_common_errors(self):
        print("Generating common errors...")
        
        common_errors = {}
        
        for word in self.vocabulary:
            word_errors = set()
            
            for original, replacement in self.phonetic_patterns['down_syndrome_patterns']['substitutions'].items():
                if original in word:
                    error_word = word.replace(original, replacement)
                    word_errors.add(error_word)
            
            for char_to_remove in self.phonetic_patterns['down_syndrome_patterns']['deletions']['word_end']:
                if word.endswith(char_to_remove):
                    error_word = word[:-1]
                    if len(error_word) > 1:
                        word_errors.add(error_word)
            
            for char_to_repeat in self.phonetic_patterns['down_syndrome_patterns']['insertions']['repeated_chars']:
                if char_to_repeat in word:
                    error_word = word.replace(char_to_repeat, char_to_repeat + char_to_repeat, 1)
                    word_errors.add(error_word)
            
            if word_errors:
                common_errors[word] = list(word_errors)
        
        self.common_errors = common_errors
        print(f"Generated errors for {len(common_errors)} words")
    
    def build_confusion_matrix(self):
        print("Building confusion matrix...")
        
        for original, replacement in self.phonetic_patterns['down_syndrome_patterns']['substitutions'].items():
            self.confusion_matrix[original][replacement] = 0.7
            self.confusion_matrix[original][original] = 0.3
            
            self.confusion_matrix[replacement][original] = 0.2
            self.confusion_matrix[replacement][replacement] = 0.8
        
        for group_name, chars in self.phonetic_patterns['arabic_groups']['similar_sounds'].items():
            for char1, char2 in combinations(chars, 2):
                self.confusion_matrix[char1][char2] = 0.3
                self.confusion_matrix[char2][char1] = 0.3
        
        print(f"Built confusion matrix for {len(self.confusion_matrix)} characters")
    
    def build_speaker_specific_correctors(self):
        print("Building speaker-specific correctors...")
        
        speaker_patterns = {}
        
        for speaker, word_counts in self.speaker_vocabularies.items():
            speaker_errors = {}
            
            most_common = word_counts.most_common(10)
            
            for word, count in most_common:
                if word in self.common_errors:
                    speaker_errors[word] = {
                        'common_errors': self.common_errors[word],
                        'frequency': count,
                        'confidence': min(count / max(word_counts.values()), 1.0)
                    }
            
            speaker_patterns[speaker] = {
                'vocabulary_size': len(word_counts),
                'total_words': sum(word_counts.values()),
                'most_common_words': dict(most_common),
                'error_patterns': speaker_errors
            }
        
        self.speaker_specific_patterns = speaker_patterns
        print(f"Built correctors for {len(speaker_patterns)} speakers")
    
    def calculate_word_similarity(self, word1, word2):
        text_similarity = SequenceMatcher(None, word1, word2).ratio()
        phonetic_similarity = self.calculate_phonetic_similarity(word1, word2)
        combined_similarity = (text_similarity * 0.6) + (phonetic_similarity * 0.4)
        return combined_similarity
    
    def calculate_phonetic_similarity(self, word1, word2):
        if len(word1) == 0 or len(word2) == 0:
            return 0.0
        
        phonetic1 = self.to_phonetic_representation(word1)
        phonetic2 = self.to_phonetic_representation(word2)
        
        return SequenceMatcher(None, phonetic1, phonetic2).ratio()
    
    def to_phonetic_representation(self, word):
        phonetic = word
        
        for original, replacement in self.phonetic_patterns['down_syndrome_patterns']['substitutions'].items():
            phonetic = phonetic.replace(original, replacement)
        
        for group_chars in self.phonetic_patterns['arabic_groups']['similar_sounds'].values():
            if len(group_chars) >= 2:
                representative = group_chars[0]
                for char in group_chars[1:]:
                    phonetic = phonetic.replace(char, representative)
        
        return phonetic
    
    def correct_word(self, word, context=None, speaker=None):
        if word in self.vocabulary:
            return {
                'original': word,
                'corrected': word,
                'confidence': 1.0,
                'method': 'exact_match',
                'alternatives': []
            }
        
        candidates = []
        
        close_matches = get_close_matches(
            word, self.vocabulary, 
            n=5, cutoff=self.similarity_threshold
        )
        
        for match in close_matches:
            similarity = self.calculate_word_similarity(word, match)
            candidates.append({
                'word': match,
                'similarity': similarity,
                'method': 'text_similarity'
            })
        
        for vocab_word in self.vocabulary:
            if vocab_word not in close_matches:
                phonetic_sim = self.calculate_phonetic_similarity(word, vocab_word)
                if phonetic_sim >= self.similarity_threshold:
                    candidates.append({
                        'word': vocab_word,
                        'similarity': phonetic_sim,
                        'method': 'phonetic_similarity'
                    })
        
        candidates.sort(key=lambda x: x['similarity'], reverse=True)
        
        if candidates:
            best_candidate = candidates[0]
            return {
                'original': word,
                'corrected': best_candidate['word'],
                'confidence': best_candidate['similarity'],
                'method': best_candidate['method'],
                'alternatives': [c['word'] for c in candidates[1:5]]
            }
        else:
            return {
                'original': word,
                'corrected': word,
                'confidence': 0.0,
                'method': 'no_correction',
                'alternatives': []
            }
    
    def correct_sequence(self, words, speaker=None):
        corrected_words = []
        context = []
        
        for word in words:
            current_context = context[-self.context_window:] if context else []
            correction_result = self.correct_word(word, current_context, speaker)
            corrected_word = correction_result['corrected']
            
            corrected_words.append(correction_result)
            context.append(corrected_word)
        
        return corrected_words
    
    def evaluate_correction_performance(self):
        print("Evaluating correction performance...")
        
        test_cases = []
        
        for correct_word, error_variants in list(self.common_errors.items())[:50]:
            for error_word in error_variants[:3]:
                test_cases.append({
                    'correct': correct_word,
                    'error': error_word
                })
        
        correct_corrections = 0
        total_tests = len(test_cases)
        
        results = []
        
        for test_case in test_cases:
            correction_result = self.correct_word(test_case['error'])
            corrected = correction_result['corrected']
            
            is_correct = (corrected == test_case['correct'])
            if is_correct:
                correct_corrections += 1
            
            results.append({
                'original_correct': test_case['correct'],
                'error_input': test_case['error'],
                'corrected_output': corrected,
                'is_correct': is_correct,
                'confidence': correction_result['confidence']
            })
        
        accuracy = correct_corrections / total_tests if total_tests > 0 else 0
        print(f"Correction accuracy: {accuracy:.4f} ({correct_corrections}/{total_tests})")
        
        self.results['correction_accuracy'] = accuracy
        self.results['test_results'] = results[:10]
        
        return accuracy
    
    def train(self):
        print("Starting corrector training...")
        
        try:
            self.df = self.data_processor.load_features(self.features_path)
            
            words = self.df['word'].dropna().astype(str).tolist()
            speakers = self.df['speaker'].dropna().astype(str).tolist()
            
            self.vocabulary = set(words)
            
            speaker_word_counts = defaultdict(Counter)
            for word, speaker in zip(words, speakers):
                speaker_word_counts[speaker][word] += 1
            
            self.speaker_vocabularies = dict(speaker_word_counts)
            
            self.analyze_phonetic_patterns()
            self.generate_common_errors()
            self.build_confusion_matrix()
            self.build_speaker_specific_correctors()
            
            accuracy = self.evaluate_correction_performance()
            
            self.model = {
                'vocabulary': list(self.vocabulary),
                'phonetic_patterns': self.phonetic_patterns,
                'common_errors': self.common_errors,
                'confusion_matrix': dict(self.confusion_matrix),
                'speaker_specific_patterns': self.speaker_specific_patterns,
                'similarity_threshold': self.similarity_threshold,
                'max_edit_distance': self.max_edit_distance
            }
            
            self.is_trained = True
            print("Corrector training completed")
            return True
            
        except Exception as e:
            print(f"Training error: {str(e)}")
            return False
    
    def predict(self, word, context=None, speaker=None):
        return self.correct_word(word, context, speaker)
    
    def evaluate(self):
        if 'correction_accuracy' in self.results:
            return {'accuracy': self.results['correction_accuracy']}
        return None
    
    def save_model(self, filename=None):
        if not self.is_trained:
            print("Model not trained yet")
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if filename is None:
            filename = f"corrector_{timestamp}.pkl"
        
        filepath = f"{self.results_path}/correction_models/{filename}"
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        
        print(f"Corrector model saved: {filepath}")
        return filepath