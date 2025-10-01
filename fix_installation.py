#!/usr/bin/env python3
"""
Python 3.13 uyumlu kurulum script'i
"""
import subprocess
import sys

def run_command(command):
    """Komut çalıştır"""
    print(f"🔄 {command}")
    try:
        subprocess.run(command, shell=True, check=True)
        print(f"✅ Başarılı")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Hata: {e}")
        return False

def main():
    print("🐍 Python 3.13 Uyumlu Kurulum")
    print("=" * 40)
    
    # Önce pip'i güncelle
    run_command("python -m pip install --upgrade pip")
    
    # Temel paketleri tek tek yükle
    packages = [
        "numpy==1.26.0",
        "scipy==1.11.3", 
        "librosa==0.10.1",
        "scikit-learn==1.3.2",
        "matplotlib==3.8.0",
        "pandas==2.1.3",
        "torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu",
        "torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu",
        "PyYAML==6.0.1",
        "jupyter==1.0.0",
        "soundfile==0.12.1",
        "noisereduce==2.0.1"
    ]
    
    for package in packages:
        run_command(f"pip install {package}")
    
    print("\n🎉 Kurulum tamamlandı!")
    print("🚀 Test etmek için: python run_pipeline.py --audio data/raw/test_audio.wav")

if __name__ == "__main__":
    main()