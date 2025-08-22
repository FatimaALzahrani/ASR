import pandas as pd
from typing import Dict


class MappingCreator:
    def create_mappings(self, data: pd.DataFrame) -> Dict:
        words = sorted(data['word'].unique())
        speakers = sorted(data['speaker'].unique())
        qualities = sorted(data['quality'].unique())
        
        word_to_id = {word: idx for idx, word in enumerate(words)}
        id_to_word = {idx: word for word, idx in word_to_id.items()}
        
        speaker_to_id = {speaker: idx for idx, speaker in enumerate(speakers)}
        id_to_speaker = {idx: speaker for speaker, idx in speaker_to_id.items()}
        
        quality_to_id = {quality: idx for idx, quality in enumerate(qualities)}
        id_to_quality = {idx: quality for quality, idx in quality_to_id.items()}
        
        mappings = {
            'word_to_id': word_to_id,
            'id_to_word': id_to_word,
            'speaker_to_id': speaker_to_id,
            'id_to_speaker': id_to_speaker,
            'quality_to_id': quality_to_id,
            'id_to_quality': id_to_quality,
            'num_words': len(words),
            'num_speakers': len(speakers),
            'num_qualities': len(qualities)
        }
        
        return mappings