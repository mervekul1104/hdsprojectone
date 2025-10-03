import torch
from TTS.api import TTS
import logging
from pathlib import Path
from typing import Dict, List, Optional
import soundfile as sf
import librosa
import numpy as np

logger = logging.getLogger(__name__)

class VoiceCloner:
    def __init__(self, model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"):
        """
        Ses klonlama ve sentezleme sınıfı
        
        Args:
            model_name: Kullanılacak TTS modeli
        """
        self.model_name = model_name
        self.tts = None
        self._load_model()
        
    def _load_model(self):
        """TTS modelini yükle"""
        try:
            self.tts = TTS(self.model_name).to(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            
            # Optimize edilmiş TTS parametreleri
            self.tts_config = {
                "speed": 0.95,        # Biraz yavaşlat (daha doğal ses için)
                "temperature": 0.7,   # Daha az rastgelelik
                "length_scale": 1.1,  # Konuşma hızını ayarla
                "language": "tr"      # Varsayılan dil
            }
            
            logger.info("TTS modeli başarıyla yüklendi")
            
        except Exception as e:
            logger.error(f"TTS modeli yüklenemedi: {e}")
            self.tts = None
    
    def extract_speaker_reference(self, audio_file: str, start_time: float, end_time: float, 
                                 output_file: str, sample_rate: int = 22050) -> bool:
        """
        Konuşmacı referans sesini çıkar
        
        Args:
            audio_file: Orijinal ses dosyası
            start_time: Başlangıç zamanı (saniye)
            end_time: Bitiş zamanı (saniye)
            output_file: Çıktı dosyası yolu
            sample_rate: Örnekleme oranı
            
        Returns:
            Başarı durumu
        """
        try:
            # Ses dosyasını yükle
            audio, sr = librosa.load(audio_file, sr=sample_rate)
            
            # Zaman aralığını sample'a çevir
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            
            # Segmenti çıkar
            speaker_audio = audio[start_sample:end_sample]
            
            # Dosyayı kaydet
            sf.write(output_file, speaker_audio, sr)
            
            logger.info(f"Referans ses çıkarıldı: {output_file} ({len(speaker_audio)/sr:.2f}s)")
            return True
            
        except Exception as e:
            logger.error(f"Referans ses çıkarma hatası: {e}")
            return False
    
    def clone_voice(self, text: str, reference_voice: str, output_file: str, 
                   language: str = "tr", **kwargs) -> bool:
        """
        Metni referans sese göre sentezle (ses klonlama)
        
        Args:
            text: Sentezlenecek metin
            reference_voice: Referans ses dosyası
            output_file: Çıktı dosyası yolu
            language: Dil kodu
            **kwargs: Ek TTS parametreleri
            
        Returns:
            Başarı durumu
        """
        if not self.tts:
            logger.error("TTS modeli mevcut değil")
            return False
        
        if not Path(reference_voice).exists():
            logger.error(f"Referans ses dosyası bulunamadı: {reference_voice}")
            return False
        
        try:
            # TTS parametrelerini birleştir
            tts_params = {
                "speed": kwargs.get("speed", self.tts_config["speed"]),
                "temperature": kwargs.get("temperature", self.tts_config["temperature"]),
                "length_scale": kwargs.get("length_scale", self.tts_config["length_scale"]),
            }
            
            # Ses sentezleme
            self.tts.tts_to_file(
                text=text,
                speaker_wav=reference_voice,
                language=language,
                file_path=output_file,
                **tts_params
            )
            
            logger.info(f"Ses sentezlendi: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Ses klonlama hatası: {e}")
            return False
    
    def batch_clone_voices(self, segments: List[Dict], voice_files: Dict[str, str], 
                          output_dir: str) -> List[Dict]:
        """
        Toplu ses klonlama işlemi
        
        Args:
            segments: Segment listesi [{"text": "", "speaker": "", "start": float, "end": float}]
            voice_files: Konuşmacı ses dosyaları {"speaker": "file_path"}
            output_dir: Çıktı dizini
            
        Returns:
            Sentezlenmiş dosya bilgileri
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        synthesized_files = []
        
        for i, segment in enumerate(segments):
            speaker = segment["speaker"]
            text = segment["text"]
            
            if not text.strip():
                continue
            
            # Konuşmacının referans sesini kontrol et
            reference_voice = voice_files.get(speaker)
            if not reference_voice:
                logger.warning(f"Konuşmacı {speaker} için referans ses bulunamadı")
                continue
            
            # Çıktı dosyası
            output_file = output_dir / f"segment_{i:04d}_{speaker}.wav"
            
            # Ses klonlama
            success = self.clone_voice(
                text=text,
                reference_voice=reference_voice,
                output_file=str(output_file),
                language="tr"
            )
            
            if success:
                synthesized_files.append({
                    "file": str(output_file),
                    "start": segment["start"],
                    "end": segment["end"],
                    "speaker": speaker,
                    "text": text
                })
                
                logger.info(f"Segment {i + 1}/{len(segments)} sentezlendi")
        
        return synthesized_files

    def get_available_languages(self) -> List[str]:
        """Desteklenen dilleri listele"""
        if not self.tts:
            return []
        
        try:
            return self.tts.languages or []
        except:
            return ["tr", "en", "de", "fr", "es", "it"]  # Varsayılan dil listesi

    def get_voice_characteristics(self, audio_file: str) -> Dict:
        """
        Ses karakteristiklerini analiz et (isteğe bağlı)
        
        Args:
            audio_file: Ses dosyası
            
        Returns:
            Ses özellikleri
        """
        try:
            audio, sr = librosa.load(audio_file, sr=22050)
            
            # Temel ses analizi
            features = {
                "duration": len(audio) / sr,
                "rms_energy": np.mean(librosa.feature.rms(y=audio)),
                "pitch_mean": np.mean(librosa.piptrack(y=audio, sr=sr)[0]),
                "spectral_centroid": np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Ses analizi hatası: {e}")
            return {}


# Kullanım örneği
def example_usage():
    """Örnek kullanım"""
    
    # Ses klonlayıcıyı başlat
    cloner = VoiceCloner()
    
    # 1. Referans ses çıkar
    cloner.extract_speaker_reference(
        audio_file="orijinal_ses.wav",
        start_time=10.5,
        end_time=15.2,
        output_file="referans_ses.wav"
    )
    
    # 2. Tekil ses klonlama
    cloner.clone_voice(
        text="Merhaba, bu klonlanmış bir ses denemesidir.",
        reference_voice="referans_ses.wav",
        output_file="klonlanmis_ses.wav"
    )
    
    # 3. Toplu ses klonlama
    segments = [
        {
            "text": "İlk konuşma segmenti",
            "speaker": "SPEAKER_00",
            "start": 0.0,
            "end": 2.5
        },
        {
            "text": "İkinci konuşma segmenti", 
            "speaker": "SPEAKER_00",
            "start": 3.0,
            "end": 5.5
        }
    ]
    
    voice_files = {
        "SPEAKER_00": "referans_ses.wav"
    }
    
    synthesized = cloner.batch_clone_voices(
        segments=segments,
        voice_files=voice_files,
        output_dir="./cikti_sesler"
    )


if __name__ == "__main__":
    example_usage()