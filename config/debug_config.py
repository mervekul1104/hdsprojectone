#!/usr/bin/env python3
"""
Config hatasını debug etmek için
"""
import yaml
import os
import sys

print("🔍 CONFIG DEBUG BAŞLIYOR...")
print("=" * 50)

# 1. Config dosyası var mı?
config_path = "config/model_config.yaml"
print(f"1. Config dosyası: {config_path}")
print(f"   Dosya var mı: {os.path.exists(config_path)}")

if os.path.exists(config_path):
    # 2. Config dosyasını oku
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_content = yaml.safe_load(f)
        print("2. Config dosyası okundu")
        print(f"   Config içeriği: {config_content}")
    except Exception as e:
        print(f"❌ Config okuma hatası: {e}")
        config_content = {}
else:
    config_content = {}
    print("❌ Config dosyası bulunamadı!")

# 3. min_segment_length değerini kontrol et
print("\n3. min_segment_length değerleri:")
print(f"   audio.min_segment_length: {config_content.get('audio', {}).get('min_segment_length', 'BULUNAMADI')}")
print(f"   segmentation.min_segment_length: {config_content.get('segmentation', {}).get('min_segment_length', 'BULUNAMADI')}")

# 4. Varsayılan config oluştur
if not config_content:
    print("\n4. Varsayılan config oluşturuluyor...")
    default_config = {
        'audio': {
            'sample_rate': 16000,
            'min_segment_length': 1.0,
            'max_segment_length': 10.0
        },
        'segmentation': {
            'min_segment_length': 1.0,
            'max_segment_length': 10.0,
            'silence_threshold': 0.01,
            'min_silence_length': 0.5
        }
    }
    
    os.makedirs('config', exist_ok=True)
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(default_config, f, default_flow_style=False)
    print("✅ Varsayılan config oluşturuldu!")