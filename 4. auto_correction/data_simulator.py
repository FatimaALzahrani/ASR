import numpy as np

def simulate_predictions_with_errors(y_true, error_rate=0.2):
    print(f"Simulating predictions with {error_rate*100:.1f}% error rate...")
    
    predictions = []
    vocabulary = list(set(y_true))
    
    for true_word in y_true:
        if np.random.random() < error_rate:
            if np.random.random() < 0.7:
                wrong_word = np.random.choice([w for w in vocabulary if w != true_word])
                predictions.append(wrong_word)
            else:
                if len(true_word) > 1:
                    char_idx = np.random.randint(0, len(true_word))
                    chars = list(true_word)
                    phonetic_map = {
                        'ت': 'ط', 'د': 'ت', 'س': 'ص', 'ك': 'ق',
                        'ح': 'خ', 'ع': 'غ', 'ا': 'أ', 'ي': 'ى'
                    }
                    if chars[char_idx] in phonetic_map:
                        chars[char_idx] = phonetic_map[chars[char_idx]]
                    wrong_word = ''.join(chars)
                    predictions.append(wrong_word)
                else:
                    predictions.append(true_word)
        else:
            predictions.append(true_word)
    
    return predictions