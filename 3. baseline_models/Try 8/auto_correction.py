from typing import Dict, List, Tuple
import logging

from language_model import LanguageModel

logger = logging.getLogger(__name__)

class AutoCorrection:
    def __init__(self, language_model: LanguageModel):
        self.language_model = language_model
        self.arabic_corrections = self._build_arabic_corrections()
        
        logger.info("Auto-correction system initialized")
    
    def _build_arabic_corrections(self) -> Dict[str, str]:
        corrections = {
            'ض': 'ظ', 'ظ': 'ض',
            'ذ': 'ز', 'ز': 'ذ',
            'س': 'ص', 'ص': 'س',
            'ت': 'ط', 'ط': 'ت',
            'ه': 'ح', 'ح': 'ه',
            'ق': 'ك', 'ك': 'ق',
            'أ': 'ا', 'إ': 'ا',
            'ة': 'ه', 'ه': 'ة',
            'ي': 'ى', 'ى': 'ي',
            'و': 'ؤ', 'ؤ': 'و'
        }
        return corrections
    
    def generate_candidates(self, word: str, max_candidates: int = 10) -> List[str]:
        candidates = set()
        
        candidates.add(word)
        
        for i, char in enumerate(word):
            for replacement in self.arabic_corrections.get(char, [char]):
                candidate = word[:i] + replacement + word[i+1:]
                candidates.add(candidate)
        
        for i in range(len(word)):
            candidate = word[:i] + word[i+1:]
            if candidate:
                candidates.add(candidate)
        
        arabic_chars = 'ابتثجحخدذرزسشصضطظعغفقكلمنهوي'
        for i in range(len(word) + 1):
            for char in arabic_chars:
                candidate = word[:i] + char + word[i:]
                candidates.add(candidate)
        
        for i in range(len(word) - 1):
            candidate = word[:i] + word[i+1] + word[i] + word[i+2:]
            candidates.add(candidate)
        
        valid_candidates = [c for c in candidates if c in self.language_model.vocabulary]
        
        if not valid_candidates:
            valid_candidates = list(candidates)
        
        return valid_candidates[:max_candidates]
    
    def correct_word(self, word: str, context: List[str] = None) -> Tuple[str, float]:
        candidates = self.generate_candidates(word)
        
        scored_candidates = self.language_model.suggest_corrections(word, candidates)
        
        if scored_candidates:
            best_candidate, confidence = scored_candidates[0]
            return best_candidate, confidence
        else:
            return word, 0.0
    
    def correct_sentence(self, words: List[str]) -> List[Tuple[str, float]]:
        corrected_words = []
        
        for i, word in enumerate(words):
            context = words[:i] if i > 0 else []
            
            corrected_word, confidence = self.correct_word(word, context)
            corrected_words.append((corrected_word, confidence))
        
        return corrected_words
