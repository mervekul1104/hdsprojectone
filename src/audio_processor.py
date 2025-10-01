"""
Ses dosyası yükleme ve ön işleme fonksiyonları
"""
import librosa
import numpy as np
import soundfile as sf
from typing import Tuple, List, Dict, Optional
import noisereduce as nr
from scipy import signal

class AudioProcessor:
    def __init__(self, target_sr: int = 16000, mono: bool = True):
        self.target_sr = target_sr
        self.mono = mono
    
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Ses dosyasını yükler ve temel işlemler uygular
        
        Args:
            file_path: Ses dosyası yolu
            
        Returns:
            Tuple (audio_data, sample_rate)
        """
        try:
            # Ses dosyasını yükle
            audio_data, original_sr = librosa.load(
                file_path, 
                sr=self.target_sr, 
                mono=self.mono
            )
            
            print(f"Ses dosyası yüklendi: {file_path}")
            print(f"Örnekleme oranı: {original_sr} -> {self.target_sr}")
            print(f"Ses uzunluğu: {len(audio_data)} örnek ({len(audio_data)/self.target_sr:.2f} saniye)")
            
            return audio_data, self.target_sr
            
        except Exception as e:
            print(f"Ses yükleme hatası: {e}")
            raise
    
    def preprocess_audio(self, audio_data: np.ndarray, sr: int) -> np.ndarray:
        """
        Ses verisine ön işleme uygular
        
        Args:
            audio_data: Ham ses verisi
            sr: Örnekleme oranı
            
        Returns:
            İşlenmiş ses verisi
        """
        try:
            print("Ses ön işleme başlıyor...")
            
            # 1. Gürültü azaltma
            cleaned_audio = self.remove_noise(audio_data, sr)
            
            # 2. Normalizasyon
            normalized_audio = self._normalize_audio(cleaned_audio)
            
            # 3. Pre-emphasis (yüksek frekansları vurgula)
            emphasized_audio = self._apply_pre_emphasis(normalized_audio)
            
            # 4. Ses seviyesi dengeleme
            balanced_audio = self._balance_volume(emphasized_audio)
            
            print("Ses ön işleme tamamlandı")
            return balanced_audio
            
        except Exception as e:
            print(f"Ön işleme hatası: {e}")
            return audio_data  # Hata durumunda orijinal sesi döndür
    
    def remove_noise(self, audio_data: np.ndarray, sr: int) -> np.ndarray:
        """
        Ses verisinden gürültüyü azaltır
        
        Args:
            audio_data: Ses verisi
            sr: Örnekleme oranı
            
        Returns:
            Gürültüsü azaltılmış ses verisi
        """
        try:
            # noisereduce kütüphanesi ile gürültü azaltma
            reduced_noise = nr.reduce_noise(
                y=audio_data,
                sr=sr,
                prop_decrease=0.75  # %75 gürültü azaltma
            )
            return reduced_noise
            
        except Exception as e:
            print(f"Gürültü azaltma hatası: {e}")
            return audio_data
    
    def detect_silence(self, audio_data: np.ndarray, sr: int, 
                      threshold: float = 0.01, min_silence_len: float = 0.5) -> List[Tuple[float, float]]:
        """
        Sessizlik bölgelerini tespit eder
        
        Args:
            audio_data: Ses verisi
            sr: Örnekleme oranı
            threshold: Sessizlik eşik değeri
            min_silence_len: Minimum sessizlik süresi (saniye)
            
        Returns:
            Sessizlik bölgeleri listesi [(başlangıç, bitiş), ...]
        """
        try:
            # Enerji hesapla
            frame_length = int(0.025 * sr)  # 25ms
            hop_length = int(0.01 * sr)     # 10ms
            
            energy = []
            for i in range(0, len(audio_data), hop_length):
                frame = audio_data[i:i + frame_length]
                if len(frame) == frame_length:
                    frame_energy = np.sum(frame**2)
                    energy.append(frame_energy)
            
            energy = np.array(energy)
            
            # Eşik değerini belirle (adaptif)
            if len(energy) > 0:
                energy_threshold = np.percentile(energy, 20)  # En düşük %20'lik dilim
            else:
                energy_threshold = threshold
            
            # Sessizlik bölgelerini bul
            silence_regions = []
            in_silence = False
            start_frame = 0
            
            for i, e in enumerate(energy):
                time = i * hop_length / sr
                
                if e <= energy_threshold and not in_silence:
                    start_frame = i
                    in_silence = True
                elif e > energy_threshold and in_silence:
                    end_frame = i
                    silence_duration = (end_frame - start_frame) * hop_length / sr
                    
                    if silence_duration >= min_silence_len:
                        start_time = start_frame * hop_length / sr
                        end_time = end_frame * hop_length / sr
                        silence_regions.append((start_time, end_time))
                    
                    in_silence = False
            
            # Son sessizlik bölgesini kontrol et
            if in_silence:
                end_frame = len(energy)
                silence_duration = (end_frame - start_frame) * hop_length / sr
                if silence_duration >= min_silence_len:
                    start_time = start_frame * hop_length / sr
                    end_time = end_frame * hop_length / sr
                    silence_regions.append((start_time, end_time))
            
            print(f"Tespit edilen sessizlik bölgeleri: {len(silence_regions)}")
            return silence_regions
            
        except Exception as e:
            print(f"Sessizlik tespit hatası: {e}")
            return []
    
    def split_into_segments(self, audio_data: np.ndarray, sr: int, 
                           min_segment_length: float = 1.0, 
                           max_segment_length: float = 10.0) -> List[Dict]:
        """
        Ses verisini konuşma segmentlerine ayırır
        
        Args:
            audio_data: Ses verisi
            sr: Örnekleme oranı
            min_segment_length: Minimum segment uzunluğu (saniye)
            max_segment_length: Maksimum segment uzunluğu (saniye)
            
        Returns:
            Segment listesi [{'start': ..., 'end': ..., 'audio': ...}, ...]
        """
        try:
            print("Ses segmentasyonu başlıyor...")
            
            # Konuşma bölgelerini tespit et (sessizlik bölgelerinin tersi)
            silence_regions = self.detect_silence(audio_data, sr)
            speech_segments = []
            
            # İlk konuşma bölgesi
            current_start = 0.0
            
            for silence_start, silence_end in silence_regions:
                segment_end = silence_start
                segment_duration = segment_end - current_start
                
                # Minimum uzunluk kontrolü
                if segment_duration >= min_segment_length:
                    # Maksimum uzunluk kontrolü - böl if necessary
                    if segment_duration <= max_segment_length:
                        speech_segments.append({
                            'start': current_start,
                            'end': segment_end,
                            'duration': segment_duration
                        })
                    else:
                        # Uzun segmenti böl
                        num_subsegments = int(np.ceil(segment_duration / max_segment_length))
                        subsegment_duration = segment_duration / num_subsegments
                        
                        for i in range(num_subsegments):
                            sub_start = current_start + i * subsegment_duration
                            sub_end = sub_start + subsegment_duration
                            
                            if sub_end - sub_start >= min_segment_length:
                                speech_segments.append({
                                    'start': sub_start,
                                    'end': sub_end,
                                    'duration': sub_end - sub_start
                                })
                
                current_start = silence_end
            
            # Son konuşma bölgesi
            total_duration = len(audio_data) / sr
            if current_start < total_duration:
                final_duration = total_duration - current_start
                if final_duration >= min_segment_length:
                    if final_duration <= max_segment_length:
                        speech_segments.append({
                            'start': current_start,
                            'end': total_duration,
                            'duration': final_duration
                        })
                    else:
                        # Uzun segmenti böl
                        num_subsegments = int(np.ceil(final_duration / max_segment_length))
                        subsegment_duration = final_duration / num_subsegments
                        
                        for i in range(num_subsegments):
                            sub_start = current_start + i * subsegment_duration
                            sub_end = sub_start + subsegment_duration
                            
                            if sub_end <= total_duration and (sub_end - sub_start) >= min_segment_length:
                                speech_segments.append({
                                    'start': sub_start,
                                    'end': sub_end,
                                    'duration': sub_end - sub_start
                                })
            
            # Ses verisini segmentlere ayır
            for segment in speech_segments:
                start_sample = int(segment['start'] * sr)
                end_sample = int(segment['end'] * sr)
                segment['audio'] = audio_data[start_sample:end_sample]
            
            print(f"Oluşturulan segment sayısı: {len(speech_segments)}")
            return speech_segments
            
        except Exception as e:
            print(f"Segmentasyon hatası: {e}")
            return []
    
    def _normalize_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Ses verisini normalize eder"""
        if len(audio_data) == 0:
            return audio_data
        
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            return audio_data / max_val
        return audio_data
    
    def _apply_pre_emphasis(self, audio_data: np.ndarray, coefficient: float = 0.97) -> np.ndarray:
        """Pre-emphasis filtresi uygular"""
        emphasized = np.append(audio_data[0], audio_data[1:] - coefficient * audio_data[:-1])
        return emphasized
    
    def _balance_volume(self, audio_data: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
        """Ses seviyesini dengeler"""
        current_rms = np.sqrt(np.mean(audio_data**2))
        if current_rms > 0:
            gain = target_rms / current_rms
            return np.clip(audio_data * gain, -1.0, 1.0)
        return audio_data
    
    def export_segment(self, audio_data: np.ndarray, sr: int, file_path: str):
        """Ses segmentini dosyaya kaydeder"""
        try:
            sf.write(file_path, audio_data, sr)
            print(f"Segment kaydedildi: {file_path}")
        except Exception as e:
            print(f"Segment kaydetme hatası: {e}")


# Kullanım örneği
if __name__ == "__main__":
    # AudioProcessor test
    processor = AudioProcessor()
    
    # Test için kısa bir ses oluştur
    test_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 3, 48000))  # 3 saniyelik 440Hz sinüs
    test_audio = test_audio * 0.5  # Ses seviyesini düşür
    
    # Sessizlik ekle
    silence = np.zeros(16000)  # 1 saniye sessizlik
    test_audio = np.concatenate([test_audio[:16000], silence, test_audio[16000:]])
    
    # İşlemleri test et
    processed_audio = processor.preprocess_audio(test_audio, 16000)
    silence_regions = processor.detect_silence(processed_audio, 16000)
    segments = processor.split_into_segments(processed_audio, 16000)
    
    print(f"İşlenmiş ses uzunluğu: {len(processed_audio)}")
    print(f"Sessizlik bölgeleri: {silence_regions}")
    print(f"Segmentler: {len(segments)}")
    
    for i, segment in enumerate(segments):
        print(f"Segment {i}: {segment['start']:.2f}s - {segment['end']:.2f}s "
              f"({segment['duration']:.2f}s)")