import pandas as pd
from typing import Dict
from sklearn.model_selection import train_test_split
import config


class DataSplitter:
    
    def __init__(self, test_size: float = config.TEST_SIZE, 
                 val_size: float = config.VALIDATION_SIZE):
        self.test_size = test_size
        self.val_size = val_size
    
    def create_balanced_splits(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        print("Creating balanced data splits...")
        
        # Analyze word distribution
        word_counts = df['word'].value_counts()
        
        # Separate words by sample count for different strategies
        single_words = word_counts[word_counts == 1].index.tolist()
        small_words = word_counts[(word_counts >= 2) & (word_counts <= 4)].index.tolist()
        medium_words = word_counts[(word_counts >= 5) & (word_counts <= 15)].index.tolist()
        large_words = word_counts[word_counts > 15].index.tolist()
        
        print(f"Word distribution: {len(single_words)} single, {len(small_words)} small, "
              f"{len(medium_words)} medium, {len(large_words)} large")
        
        # Initialize splits
        splits = {'train': [], 'validation': [], 'test': []}
        
        # Strategy 1: Single sample words -> training only
        if single_words:
            single_data = df[df['word'].isin(single_words)]
            splits['train'].append(single_data)
            print(f"Added {len(single_data)} single-sample words to training")
        
        # Strategy 2: Small words (2-4 samples) -> mostly training, one to test
        for word in small_words:
            word_data = df[df['word'] == word]
            if len(word_data) == 2:
                # 2 samples: both to training
                splits['train'].append(word_data)
            else:
                # 3-4 samples: 1 to test, rest to training
                test_data = word_data.sample(n=1, random_state=42)
                train_data = word_data.drop(test_data.index)
                splits['train'].append(train_data)
                splits['test'].append(test_data)
        
        # Strategy 3: Medium words (5-15 samples) -> simple split
        for word in medium_words:
            word_data = df[df['word'] == word]
            word_splits = self._split_medium_word_data(word_data)
            splits['train'].append(word_splits['train'])
            if not word_splits['validation'].empty:
                splits['validation'].append(word_splits['validation'])
            if not word_splits['test'].empty:
                splits['test'].append(word_splits['test'])
        
        # Strategy 4: Large words (>15 samples) -> full stratified split
        for word in large_words:
            word_data = df[df['word'] == word]
            try:
                word_splits = self._split_large_word_data(word_data)
                splits['train'].append(word_splits['train'])
                splits['validation'].append(word_splits['validation'])
                splits['test'].append(word_splits['test'])
            except Exception as e:
                print(f"Warning: Failed to split large word {word}: {e}")
                # Fallback to simple split
                word_splits = self._split_medium_word_data(word_data)
                splits['train'].append(word_splits['train'])
                if not word_splits['validation'].empty:
                    splits['validation'].append(word_splits['validation'])
                if not word_splits['test'].empty:
                    splits['test'].append(word_splits['test'])
        
        # Combine data
        final_splits = {}
        for split_name, split_data in splits.items():
            if split_data:
                final_splits[split_name] = pd.concat(split_data, ignore_index=True)
            else:
                final_splits[split_name] = pd.DataFrame()
        
        self._print_split_statistics(final_splits)
        
        return final_splits
    
    def _split_medium_word_data(self, word_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        n_samples = len(word_data)
        
        # Calculate split sizes
        test_size = max(1, int(n_samples * 0.2))  # At least 1 for test
        val_size = max(1, int(n_samples * 0.1)) if n_samples >= 10 else 0  # Validation only if enough samples
        
        # Adjust if total split size exceeds available samples
        if test_size + val_size >= n_samples:
            val_size = 0  # Skip validation if not enough samples
            test_size = min(test_size, n_samples - 1)  # Leave at least 1 for training
        
        # Random sampling without stratification
        test_data = word_data.sample(n=test_size, random_state=42)
        remaining_data = word_data.drop(test_data.index)
        
        if val_size > 0 and len(remaining_data) > val_size:
            val_data = remaining_data.sample(n=val_size, random_state=42)
            train_data = remaining_data.drop(val_data.index)
        else:
            val_data = pd.DataFrame(columns=word_data.columns)
            train_data = remaining_data
        
        return {
            'train': train_data,
            'validation': val_data,
            'test': test_data
        }
    
    def _split_large_word_data(self, word_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        # Try stratified split if multiple speakers
        unique_speakers = word_data['speaker'].unique()
        
        if len(unique_speakers) > 1:
            # Check if each speaker has enough samples for stratification
            speaker_counts = word_data['speaker'].value_counts()
            min_speaker_count = speaker_counts.min()
            
            if min_speaker_count >= 2:
                # Full stratified split
                train_data, temp_data = train_test_split(
                    word_data, 
                    test_size=self.test_size + self.val_size, 
                    random_state=42, 
                    stratify=word_data['speaker']
                )
                
                if len(temp_data) >= 2:
                    val_data, test_data = train_test_split(
                        temp_data, 
                        test_size=self.test_size/(self.test_size + self.val_size),
                        random_state=42
                    )
                else:
                    # Not enough for further split
                    val_data = pd.DataFrame(columns=word_data.columns)
                    test_data = temp_data
            else:
                # Fallback to simple random split
                return self._split_medium_word_data(word_data)
        else:
            # Single speaker - simple random split
            return self._split_medium_word_data(word_data)
        
        return {
            'train': train_data,
            'validation': val_data,
            'test': test_data
        }
    
    def _print_split_statistics(self, splits: Dict[str, pd.DataFrame]):
        print("Data split summary:")
        
        total_samples = 0
        all_words = set()
        all_speakers = set()
        
        for split_name, split_df in splits.items():
            if not split_df.empty:
                unique_words = len(split_df['word'].unique())
                unique_speakers = len(split_df['speaker'].unique())
                samples = len(split_df)
                
                total_samples += samples
                all_words.update(split_df['word'].unique())
                all_speakers.update(split_df['speaker'].unique())
                
                print(f"  {split_name}: {samples} samples, "
                      f"{unique_words} words, {unique_speakers} speakers")
            else:
                print(f"  {split_name}: empty")
        
        print(f"Total: {total_samples} samples, {len(all_words)} unique words, {len(all_speakers)} speakers")
        
        # Check for reasonable split ratios
        if splits['train'].empty:
            print("WARNING: Training set is empty!")
        elif total_samples > 0:
            train_ratio = len(splits['train']) / total_samples
            val_ratio = len(splits['validation']) / total_samples if not splits['validation'].empty else 0
            test_ratio = len(splits['test']) / total_samples if not splits['test'].empty else 0
            
            print(f"Split ratios - Train: {train_ratio:.1%}, Val: {val_ratio:.1%}, Test: {test_ratio:.1%}")
    
    def create_speaker_specific_splits(self, df: pd.DataFrame) -> Dict[str, Dict[str, pd.DataFrame]]:
        print("Creating speaker-specific splits...")
        
        speaker_splits = {}
        
        for speaker in df['speaker'].unique():
            speaker_data = df[df['speaker'] == speaker]
            
            print(f"Processing speaker {speaker}: {len(speaker_data)} samples")
            
            if len(speaker_data) < 3:
                print(f"Warning: Speaker {speaker} has only {len(speaker_data)} samples - skipping")
                continue
            
            try:
                # Check word distribution for this speaker
                word_counts = speaker_data['word'].value_counts()
                
                # If most words have only 1 sample, use simple split
                single_word_ratio = (word_counts == 1).sum() / len(word_counts)
                
                if single_word_ratio > 0.8:  # More than 80% are single-sample words
                    # Simple random split without stratification
                    test_size = min(0.3, max(0.1, 1.0 / len(speaker_data)))  # Adaptive test size
                    
                    train_data, test_data = train_test_split(
                        speaker_data, 
                        test_size=test_size, 
                        random_state=42
                    )
                    
                    speaker_splits[speaker] = {
                        'train': train_data,
                        'test': test_data
                    }
                    
                else:
                    # Try stratified split
                    # Only use words that have multiple samples for stratification
                    multi_sample_words = word_counts[word_counts > 1].index
                    
                    if len(multi_sample_words) > 0:
                        # Create stratification labels only for multi-sample words
                        stratify_data = speaker_data[speaker_data['word'].isin(multi_sample_words)]
                        single_sample_data = speaker_data[~speaker_data['word'].isin(multi_sample_words)]
                        
                        if len(stratify_data) >= 4:  # Need at least 4 samples for 30% test split
                            # Stratified split for multi-sample words
                            strat_train, strat_test = train_test_split(
                                stratify_data,
                                test_size=0.3,
                                random_state=42,
                                stratify=stratify_data['word']
                            )
                            
                            # Combine with single-sample words (all go to training)
                            final_train = pd.concat([strat_train, single_sample_data], ignore_index=True)
                            final_test = strat_test
                        else:
                            # Not enough for stratification, simple split
                            final_train, final_test = train_test_split(
                                speaker_data,
                                test_size=0.3,
                                random_state=42
                            )
                    else:
                        # All words are single-sample, simple split
                        final_train, final_test = train_test_split(
                            speaker_data,
                            test_size=0.3,
                            random_state=42
                        )
                    
                    speaker_splits[speaker] = {
                        'train': final_train,
                        'test': final_test
                    }
                
                train_size = len(speaker_splits[speaker]['train'])
                test_size = len(speaker_splits[speaker]['test'])
                print(f"Speaker {speaker}: {train_size} train, {test_size} test")
                
            except Exception as e:
                print(f"Warning: Failed to split speaker {speaker}: {e}")
                # Fallback: simple random split
                if len(speaker_data) >= 2:
                    try:
                        train_data, test_data = train_test_split(
                            speaker_data,
                            test_size=1,  # Just one sample for test
                            random_state=42
                        )
                        speaker_splits[speaker] = {
                            'train': train_data,
                            'test': test_data
                        }
                        print(f"Speaker {speaker}: fallback split successful")
                    except:
                        print(f"Speaker {speaker}: all fallback methods failed")
        
        return speaker_splits