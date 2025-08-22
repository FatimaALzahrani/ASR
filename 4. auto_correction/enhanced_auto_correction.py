import numpy as np
from collections import defaultdict, Counter
from difflib import SequenceMatcher

class EnhancedAutoCorrection:
    
    def __init__(self):
        self.error_patterns = {}
        self.phonetic_map = {}
        self.word_frequencies = {}
        self.context_model = {}
        self.confidence_threshold = 0.6
        self.correction_rules = {}
        
        self.phonetic_similarity = {
            'ت': ['ط', 'د', 'ث'],
            'د': ['ت', 'ط', 'ذ'],
            'ط': ['ت', 'د'],
            'ث': ['ت', 'س', 'ص'],
            'ذ': ['د', 'ز'],
            'س': ['ص', 'ث', 'ش'],
            'ص': ['س', 'ض'],
            'ش': ['س', 'ج'],
            'ز': ['ذ', 'ظ'],
            'ظ': ['ز', 'ض'],
            'ض': ['ظ', 'ص'],
            'ك': ['ق', 'ج'],
            'ق': ['ك', 'غ'],
            'غ': ['ق', 'خ'],
            'خ': ['غ', 'ح'],
            'ح': ['خ', 'ه'],
            'ه': ['ح', 'ء'],
            'ع': ['غ', 'أ'],
            'أ': ['ا', 'ع'],
            'إ': ['ا', 'أ'],
            'آ': ['ا', 'أ'],
            'ا': ['أ', 'إ', 'آ'],
            'ي': ['ى', 'ج'],
            'ى': ['ي'],
            'و': ['ؤ'],
            'ؤ': ['و'],
            'ئ': ['ي', 'ء'],
            'ء': ['ئ', 'ه']
        }
        
        self.common_corrections = {
            'كتب': ['كتاب', 'كتابة'],
            'بيت': ['بيوت', 'منزل'],
            'سيارة': ['سيارات', 'عربة'],
            'مدرسة': ['مدارس', 'تعليم'],
            'طفل': ['أطفال', 'ولد'],
            'بنت': ['بنات', 'فتاة'],
            'لعبة': ['لعب', 'ألعاب'],
            'قلم': ['أقلام', 'كتابة'],
            'كرة': ['كرات', 'لعب'],
            'شمس': ['نور', 'ضوء']
        }
    
    def analyze_error_patterns(self, predictions, true_labels):
        print("Analyzing error patterns...")
        
        error_stats = {
            'total_errors': 0,
            'substitution_errors': 0,
            'phonetic_errors': 0,
            'length_errors': 0,
            'common_confusions': defaultdict(int),
            'accuracy': 0
        }
        
        correct_predictions = 0
        
        for pred, true in zip(predictions, true_labels):
            if pred == true:
                correct_predictions += 1
            else:
                error_stats['total_errors'] += 1
                
                if len(pred) == len(true):
                    error_stats['substitution_errors'] += 1
                    
                    if self._is_phonetic_error(pred, true):
                        error_stats['phonetic_errors'] += 1
                else:
                    error_stats['length_errors'] += 1
                
                error_stats['common_confusions'][(true, pred)] += 1
        
        error_stats['accuracy'] = correct_predictions / len(predictions)
        self.error_patterns = dict(error_stats['common_confusions'])
        
        return error_stats
    
    def _is_phonetic_error(self, word1, word2):
        if len(word1) != len(word2):
            return False
            
        phonetic_errors = 0
        total_differences = 0
        
        for c1, c2 in zip(word1, word2):
            if c1 != c2:
                total_differences += 1
                if c2 in self.phonetic_similarity.get(c1, []):
                    phonetic_errors += 1
        
        return total_differences > 0 and (phonetic_errors / total_differences) >= 0.5
    
    def build_language_model(self, word_list):
        print("Building language model...")
        
        self.word_frequencies = Counter(word_list)
        
        common_words = [
            'كتاب', 'قلم', 'مدرسة', 'بيت', 'سيارة', 'كرة', 'لعبة',
            'ماء', 'طعام', 'حليب', 'خبز', 'تفاح', 'موز', 'برتقال',
            'أب', 'أم', 'أخ', 'أخت', 'جد', 'جدة', 'عم', 'خال',
            'صديق', 'معلم', 'طبيب', 'شرطي', 'نار', 'ماء', 'هواء'
        ]
        
        for word in common_words:
            if word not in self.word_frequencies:
                self.word_frequencies[word] = 1
    
    def calculate_similarity_score(self, word1, word2):
        text_similarity = SequenceMatcher(None, word1, word2).ratio()
        
        phonetic_similarity = 0
        if len(word1) == len(word2):
            phonetic_matches = 0
            for c1, c2 in zip(word1, word2):
                if c1 == c2:
                    phonetic_matches += 1
                elif c2 in self.phonetic_similarity.get(c1, []):
                    phonetic_matches += 0.7
            phonetic_similarity = phonetic_matches / len(word1)
        
        length_similarity = 1 - abs(len(word1) - len(word2)) / max(len(word1), len(word2))
        
        total_score = (text_similarity * 0.4 + 
                      phonetic_similarity * 0.4 + 
                      length_similarity * 0.2)
        
        return total_score
    
    def get_correction_candidates(self, word, vocabulary, max_candidates=3):
        candidates = []
        
        for (correct, wrong), freq in self.error_patterns.items():
            if wrong == word and correct in vocabulary:
                candidates.append((correct, freq * 0.5, 'pattern'))
        
        for correct_word, related_words in self.common_corrections.items():
            if word in related_words and correct_word in vocabulary:
                candidates.append((correct_word, 0.3, 'common'))
        
        for vocab_word in vocabulary:
            if vocab_word != word:
                similarity = self.calculate_similarity_score(word, vocab_word)
                if similarity > 0.6:
                    freq_score = self.word_frequencies.get(vocab_word, 1) / 100
                    total_score = similarity * 0.7 + freq_score * 0.3
                    candidates.append((vocab_word, total_score, 'similarity'))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        return candidates[:max_candidates]
    
    def apply_auto_correction(self, predictions, vocabulary, confidence_scores=None):
        print("Applying auto correction...")
        
        corrected_predictions = []
        correction_stats = {
            'total_corrections': 0,
            'pattern_corrections': 0,
            'similarity_corrections': 0,
            'common_corrections': 0,
            'low_confidence_corrections': 0
        }
        
        for i, pred in enumerate(predictions):
            corrected_word = pred
            should_correct = False
            
            if confidence_scores is not None:
                if confidence_scores[i] < self.confidence_threshold:
                    should_correct = True
                    correction_stats['low_confidence_corrections'] += 1
            else:
                if pred not in vocabulary:
                    should_correct = True
            
            if should_correct:
                candidates = self.get_correction_candidates(pred, vocabulary)
                
                if candidates:
                    best_candidate, score, method = candidates[0]
                    
                    if score > 0.3:
                        corrected_word = best_candidate
                        correction_stats['total_corrections'] += 1
                        
                        if method == 'pattern':
                            correction_stats['pattern_corrections'] += 1
                        elif method == 'similarity':
                            correction_stats['similarity_corrections'] += 1
                        elif method == 'common':
                            correction_stats['common_corrections'] += 1
            
            corrected_predictions.append(corrected_word)
        
        return corrected_predictions, correction_stats