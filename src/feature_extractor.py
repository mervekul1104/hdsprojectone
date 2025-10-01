"""
Ses özellikleri çıkarımı
"""
import librosa
import numpy as np
from typing import Dict, List, Union

class FeatureExtractor:
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
    
    def extract_mfcc(self, audio_segment: np.ndarray) -> np.ndarray:
        """
        MFCC (Mel-frequency cepstral coefficients) özelliklerini çıkarır
        """
        try:
            mfcc = librosa.feature.mfcc(
                y=audio_segment, 
                sr=self.sample_rate, 
                n_mfcc=13,
                n_fft=2048,
                hop_length=512
            )
            # İstatistiksel özellikler
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            return np.concatenate([mfcc_mean, mfcc_std])
        except Exception as e:
            print(f"MFCC extraction error: {e}")
            return np.zeros(26)  # 13 mean + 13 std
    
    def extract_pitch(self, audio_segment: np.ndarray) -> Dict[str, float]:
        """
        Pitch (perde) özelliklerini çıkarır - cinsiyet ayrımı için kritik
        """
        try:
            # PYIN algoritması ile pitch tespiti
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio_segment, 
                fmin=50, 
                fmax=400,
                sr=self.sample_rate
            )
            
            # NaN değerleri temizle
            f0_clean = f0[~np.isnan(f0)]
            
            if len(f0_clean) > 0:
                return {
                    'pitch_mean': np.mean(f0_clean),
                    'pitch_std': np.std(f0_clean),
                    'pitch_median': np.median(f0_clean),
                    'pitch_range': np.max(f0_clean) - np.min(f0_clean),
                    'voiced_ratio': np.mean(voiced_flag) if voiced_flag is not None else 0
                }
            else:
                return {
                    'pitch_mean': 0.0,
                    'pitch_std': 0.0,
                    'pitch_median': 0.0,
                    'pitch_range': 0.0,
                    'voiced_ratio': 0.0
                }
        except Exception as e:
            print(f"Pitch extraction error: {e}")
            return {
                'pitch_mean': 0.0, 'pitch_std': 0.0, 'pitch_median': 0.0,
                'pitch_range': 0.0, 'voiced_ratio': 0.0
            }
    
    def extract_energy(self, audio_segment: np.ndarray) -> Dict[str, float]:
        """
        Enerji ve ses şiddeti özellikleri
        """
        try:
            # RMS enerjisi
            rms = librosa.feature.rms(y=audio_segment)
            rms_clean = rms[~np.isnan(rms)]
            
            # Zero-crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio_segment)
            
            if len(rms_clean) > 0:
                return {
                    'energy_mean': float(np.mean(rms_clean)),
                    'energy_std': float(np.std(rms_clean)),
                    'energy_max': float(np.max(rms_clean)),
                    'zcr_mean': float(np.mean(zcr)),
                    'zcr_std': float(np.std(zcr))
                }
            else:
                return {
                    'energy_mean': 0.0, 'energy_std': 0.0, 'energy_max': 0.0,
                    'zcr_mean': 0.0, 'zcr_std': 0.0
                }
        except Exception as e:
            print(f"Energy extraction error: {e}")
            return {
                'energy_mean': 0.0, 'energy_std': 0.0, 'energy_max': 0.0,
                'zcr_mean': 0.0, 'zcr_std': 0.0
            }
    
    def extract_spectral_features(self, audio_segment: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Spektral özellikleri çıkarır
        """
        try:
            # Chroma features
            chroma = librosa.feature.chroma_stft(
                y=audio_segment, 
                sr=self.sample_rate
            )
            
            # Spectral contrast
            spectral_contrast = librosa.feature.spectral_contrast(
                y=audio_segment, 
                sr=self.sample_rate
            )
            
            # Spectral centroid ve bandwidth
            spectral_centroid = librosa.feature.spectral_centroid(
                y=audio_segment, 
                sr=self.sample_rate
            )
            
            spectral_bandwidth = librosa.feature.spectral_bandwidth(
                y=audio_segment, 
                sr=self.sample_rate
            )
            
            # Mel-spectrogram
            mel_spectrogram = librosa.feature.melspectrogram(
                y=audio_segment, 
                sr=self.sample_rate
            )
            mel_spectrogram_db = librosa.power_to_db(mel_spectrogram)
            
            return {
                'chroma_mean': np.mean(chroma, axis=1),
                'chroma_std': np.std(chroma, axis=1),
                'spectral_contrast_mean': np.mean(spectral_contrast, axis=1),
                'spectral_centroid_mean': float(np.mean(spectral_centroid)),
                'spectral_bandwidth_mean': float(np.mean(spectral_bandwidth)),
                'mel_spectrogram_mean': np.mean(mel_spectrogram_db, axis=1)
            }
        except Exception as e:
            print(f"Spectral features extraction error: {e}")
            return {
                'chroma_mean': np.zeros(12),
                'chroma_std': np.zeros(12),
                'spectral_contrast_mean': np.zeros(7),
                'spectral_centroid_mean': 0.0,
                'spectral_bandwidth_mean': 0.0,
                'mel_spectrogram_mean': np.zeros(128)
            }
    
    def get_complete_feature_set(self, audio_segment: np.ndarray) -> Dict[str, Union[float, np.ndarray]]:
        """
        Tüm özellikleri birleştirir ve tek bir feature vektörü oluşturur
        """
        try:
            # Tüm özellikleri çıkar
            mfcc_features = self.extract_mfcc(audio_segment)
            pitch_features = self.extract_pitch(audio_segment)
            energy_features = self.extract_energy(audio_segment)
            spectral_features = self.extract_spectral_features(audio_segment)
            
            # Tüm özellikleri birleştir
            complete_features = {
                'mfcc': mfcc_features,
                **pitch_features,
                **energy_features,
                **spectral_features
            }
            
            # Düz vektör formatında da döndür (ML modelleri için)
            feature_vector = self._flatten_features(complete_features)
            complete_features['feature_vector'] = feature_vector
            
            return complete_features
            
        except Exception as e:
            print(f"Complete feature extraction error: {e}")
            return {}
    
    def _flatten_features(self, features_dict: Dict) -> np.ndarray:
        """
        Sözlük formatındaki özellikleri düz vektöre çevirir
        """
        flattened = []
        
        for key, value in features_dict.items():
            if key == 'feature_vector':
                continue
                
            if isinstance(value, (int, float)):
                flattened.append(value)
            elif isinstance(value, np.ndarray):
                flattened.extend(value.tolist())
            elif isinstance(value, dict):
                # Recursive olarak dict içindeki değerleri de ekle
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (int, float)):
                        flattened.append(sub_value)
                    elif isinstance(sub_value, np.ndarray):
                        flattened.extend(sub_value.tolist())
        
        return np.array(flattened)


# Test fonksiyonu
if __name__ == "__main__":
    # Test için rastgele ses verisi oluştur
    sample_audio = np.random.randn(16000)  # 1 saniyelik ses
    
    extractor = FeatureExtractor()
    features = extractor.get_complete_feature_set(sample_audio)
    
    print("Özellik çıkarımı başarılı!")
    print(f"Toplam özellik sayısı: {len(features.get('feature_vector', []))}")
    print(f"Pitch mean: {features.get('pitch_mean', 'N/A')}")