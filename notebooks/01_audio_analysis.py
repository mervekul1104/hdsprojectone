#!/usr/bin/env python3
"""
Ses Analizi ve Segmentasyon - Python Script Versiyonu
"""

import sys
import os
sys.path.append('../src')

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from audio_processor import AudioProcessor
from feature_extractor import FeatureExtractor
from utils import ProjectUtils, VisualizationUtils

def main():
    print("🎵 Ses Analizi ve Segmentasyon Başlıyor...")
    
    # Ses dosyasını yükle
    audio_path = "../data/raw/sample_audio.wav"
    
    if not os.path.exists(audio_path):
        print("❌ Ses dosyası bulunamadı! Test sesi oluşturuluyor...")
        import soundfile as sf
        t = np.linspace(0, 5, 5 * 16000)
        test_audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        sf.write(audio_path, test_audio, 16000)
        print(f"✅ Test sesi oluşturuldu: {audio_path}")
    
    # Ses işlemcisini başlat
    processor = AudioProcessor()
    
    # Ses dosyasını yükle
    audio_data, sr = processor.load_audio(audio_path)
    
    # Ön işleme
    processed_audio = processor.preprocess_audio(audio_data, sr)
    
    # Segmentasyon
    segments = processor.split_into_segments(processed_audio, sr)
    print(f"Toplam {len(segments)} segment bulundu")
    
    # Görselleştirme
    plt.figure(figsize=(15, 8))
    
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(audio_data, sr=sr, alpha=0.5, color='blue')
    plt.title('Orijinal Ses - Dalga Formu')
    plt.xlabel('Zaman (s)')
    plt.ylabel('Genlik')
    
    plt.subplot(2, 1, 2)
    librosa.display.waveshow(processed_audio, sr=sr, alpha=0.5, color='red')
    plt.title('İşlenmiş Ses - Dalga Formu')
    plt.xlabel('Zaman (s)')
    plt.ylabel('Genlik')
    
    plt.tight_layout()
    plt.savefig('../results/analysis/waveform_comparison.png')
    print("✅ Dalga formu görseli kaydedildi")
    
    # Segment bilgilerini göster
    for i, segment in enumerate(segments[:10]):
        print(f"Segment {i+1}: {segment['start_time']:.2f}s - {segment['end_time']:.2f}s "
            f"({segment['duration']:.2f}s)")

    
    # Özellik çıkarımı
    feature_extractor = FeatureExtractor(sample_rate=sr)
    
    if segments:
        first_segment = segments[0]['audio']
        features = feature_extractor.get_complete_feature_set(first_segment)
        
        print("\n📈 ÖZELLİK ÖZETİ:")
        print(f"Toplam özellik sayısı: {len(features.get('feature_vector', []))}")
        print(f"Pitch ortalaması: {features.get('pitch_mean', 'N/A'):.1f} Hz")
        print(f"Enerji ortalaması: {features.get('energy_mean', 'N/A'):.3f}")
        print(f"Spectral centroid: {features.get('spectral_centroid_mean', 'N/A'):.1f}")
    
    print("\n✅ Analiz tamamlandı!")

if __name__ == "__main__":
    main()