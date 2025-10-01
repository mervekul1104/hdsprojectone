#!/usr/bin/env python3
"""
Eksiksiz config dosyası oluştur
"""
import yaml
import os

# Eksiksiz config içeriği
config_content = {
    'audio': {
        'sample_rate': 16000,
        'frame_length': 0.025,
        'hop_length': 0.01,
        'channels': 1,
        'min_segment_length': 1.0,
        'max_segment_length': 10.0
    },
    'features': {
        'mfcc': {
            'n_mfcc': 13,
            'n_fft': 2048,
            'hop_length': 512
        },
        'pitch': {
            'fmin': 50,
            'fmax': 400
        },
        'spectral': {
            'n_fft': 2048,
            'hop_length': 512
        }
    },
    'clustering': {
        'algorithm': "dbscan",
        'eps': 0.5,
        'min_samples': 2,
        'metric': "euclidean"
    },
    'gender_classification': {
        'male_pitch_range': [85, 180],
        'female_pitch_range': [165, 255],
        'uncertain_range': [150, 180]
    },
    'segmentation': {
        'min_segment_length': 1.0,
        'max_segment_length': 10.0,
        'silence_threshold': 0.01,
        'min_silence_length': 0.5
    },
    'preprocessing': {
        'noise_reduction': True,
        'normalization': True,
        'pre_emphasis': True,
        'volume_balancing': True
    }
}

# Config klasörünü oluştur
os.makedirs('config', exist_ok=True)

# Config dosyasını yaz
with open('config/model_config.yaml', 'w', encoding='utf-8') as f:
    yaml.dump(config_content, f, default_flow_style=False, allow_unicode=True, indent=2)

print("✅ Config dosyası oluşturuldu: config/model_config.yaml")

# Kontrol et
with open('config/model_config.yaml', 'r', encoding='utf-8') as f:
    loaded_config = yaml.safe_load(f)
    print("✅ Config kontrolü:")
    print(f"   - min_segment_length: {loaded_config['audio'].get('min_segment_length', 'BULUNAMADI')}")
    print(f"   - max_segment_length: {loaded_config['audio'].get('max_segment_length', 'BULUNAMADI')}")
    print(f"   - segmentation: {loaded_config.get('segmentation', 'BULUNAMADI')}")