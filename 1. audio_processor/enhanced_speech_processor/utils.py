import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Any, Dict, List, Union


def convert_to_json_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif hasattr(obj, 'item'):  # numpy scalar
        return obj.item()
    else:
        return obj


def safe_json_save(data: Dict, filepath: Union[str, Path], encoding: str = 'utf-8') -> bool:
    try:
        # Convert data to JSON serializable format
        json_data = convert_to_json_serializable(data)
        
        # Save to file
        with open(filepath, 'w', encoding=encoding) as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        return True
        
    except Exception as e:
        print(f"Error saving JSON to {filepath}: {e}")
        return False


def validate_file_structure(data_root: Path) -> Dict[str, Any]:
    results = {
        'valid': True,
        'issues': [],
        'warnings': [],
        'stats': {}
    }
    
    # Check main directories
    clean_dir = data_root / "clean"
    if not clean_dir.exists():
        results['valid'] = False
        results['issues'].append(f"Clean audio directory not found: {clean_dir}")
        return results
    
    # Check word folders
    word_folders = [d for d in clean_dir.iterdir() if d.is_dir()]
    if not word_folders:
        results['valid'] = False
        results['issues'].append("No word folders found in clean directory")
        return results
    
    # Count audio files
    total_files = 0
    word_stats = {}
    
    for word_folder in word_folders:
        audio_files = list(word_folder.glob("*.wav"))
        file_count = len(audio_files)
        total_files += file_count
        word_stats[word_folder.name] = file_count
        
        if file_count == 0:
            results['warnings'].append(f"No audio files in folder: {word_folder.name}")
        elif file_count == 1:
            results['warnings'].append(f"Only 1 audio file in folder: {word_folder.name}")
    
    results['stats'] = {
        'total_word_folders': len(word_folders),
        'total_audio_files': total_files,
        'word_distribution': word_stats,
        'avg_files_per_word': total_files / len(word_folders) if word_folders else 0
    }
    
    if total_files == 0:
        results['valid'] = False
        results['issues'].append("No audio files found in any word folder")
    elif total_files < 100:
        results['warnings'].append(f"Very few audio files found: {total_files}")
    
    return results


def print_processing_summary(stats: Dict[str, Any], title: str = "Processing Summary"):
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    
    if 'total_samples' in stats:
        print(f"📊 Data Statistics:")
        print(f"  Total samples: {stats.get('total_samples', 'N/A')}")
        print(f"  Total words: {stats.get('total_words', 'N/A')}")
        print(f"  Total speakers: {stats.get('total_speakers', 'N/A')}")
        
        if 'avg_duration' in stats:
            print(f"  Average duration: {stats['avg_duration']:.2f}s")
        if 'avg_quality_score' in stats:
            print(f"  Average quality: {stats['avg_quality_score']:.2f}")
    
    if 'processing_stats' in stats:
        proc_stats = stats['processing_stats']
        print(f"\n🔄 Processing Statistics:")
        print(f"  Files processed: {proc_stats.get('processed_files', 'N/A')}")
        print(f"  Files excluded: {proc_stats.get('excluded_files', 'N/A')}")
        print(f"  Files failed: {proc_stats.get('failed_files', 'N/A')}")
    
    print(f"{'='*60}")


def create_simple_mappings(words: List[str], speakers: List[str]) -> Dict[str, Any]:
    return {
        'words': sorted(words),
        'speakers': sorted(speakers),
        'word_to_id': {word: idx for idx, word in enumerate(sorted(words))},
        'speaker_to_id': {speaker: idx for idx, speaker in enumerate(sorted(speakers))},
        'num_words': len(words),
        'num_speakers': len(speakers)
    }


def check_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {'status': 'EMPTY', 'issues': ['Dataset is empty']}
    
    issues = []
    warnings = []
    
    # Check for required columns
    required_cols = ['word', 'speaker', 'duration', 'quality_score']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")
    
    # Check for missing values
    for col in required_cols:
        if col in df.columns and df[col].isnull().any():
            null_count = df[col].isnull().sum()
            warnings.append(f"Column '{col}' has {null_count} missing values")
    
    # Check data balance
    if 'word' in df.columns:
        word_counts = df['word'].value_counts()
        single_words = (word_counts == 1).sum()
        if single_words > len(word_counts) * 0.5:
            warnings.append(f"Many words ({single_words}) have only 1 sample")
    
    # Determine status
    if issues:
        status = 'POOR'
    elif len(warnings) > 5:
        status = 'FAIR'
    elif warnings:
        status = 'GOOD'
    else:
        status = 'EXCELLENT'
    
    return {
        'status': status,
        'issues': issues,
        'warnings': warnings,
        'sample_count': len(df),
        'word_count': len(df['word'].unique()) if 'word' in df.columns else 0,
        'speaker_count': len(df['speaker'].unique()) if 'speaker' in df.columns else 0
    }