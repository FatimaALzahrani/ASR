import numpy as np
import librosa

class FeatureExtractor:
    def __init__(self, sr=22050):
        self.sr = sr
    
    def extract_features(self, file_path):
        try:
            audio, sr = librosa.load(file_path, sr=self.sr)
            
            if len(audio) == 0:
                return None
            
            features = []
            
            duration = len(audio) / sr
            features.append(duration)
            
            rms = librosa.feature.rms(y=audio)[0]
            features.extend([
                np.mean(rms),
                np.std(rms),
                np.max(rms),
                np.min(rms)
            ])
            
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            for i in range(13):
                features.extend([
                    np.mean(mfccs[i]),
                    np.std(mfccs[i]),
                    np.max(mfccs[i]),
                    np.min(mfccs[i])
                ])
            
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=13)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            for i in range(13):
                features.extend([
                    np.mean(mel_spec_db[i]),
                    np.std(mel_spec_db[i])
                ])
            
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
            
            features.extend([
                np.mean(spectral_centroids),
                np.std(spectral_centroids),
                np.mean(spectral_rolloff),
                np.std(spectral_rolloff),
                np.mean(spectral_bandwidth),
                np.std(spectral_bandwidth)
            ])
            
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            features.extend([
                np.mean(zcr),
                np.std(zcr)
            ])
            
            frame_length = 2048
            hop_length = 512
            frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
            frame_energy = np.sum(frames**2, axis=0)
            
            if len(frame_energy) > 10:
                silence_threshold = np.percentile(frame_energy, 20)
                signal_energy = np.mean(frame_energy[frame_energy > silence_threshold])
                noise_energy = np.mean(frame_energy[frame_energy <= silence_threshold])
                
                if noise_energy > 0:
                    snr = 10 * np.log10(signal_energy / noise_energy)
                else:
                    snr = 50
            else:
                snr = 30
            
            features.append(snr)
            
            speech_frames = frame_energy > silence_threshold if len(frame_energy) > 10 else np.ones(len(frame_energy), dtype=bool)
            speech_ratio = np.sum(speech_frames) / len(speech_frames) if len(speech_frames) > 0 else 1.0
            features.append(speech_ratio)
            
            return np.array(features)
            
        except Exception as e:
            print(f"Feature extraction error for {file_path}: {e}")
            return None