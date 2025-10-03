#!/usr/bin/env python3
"""
Tek komutla tüm pipeline'ı çalıştırma
"""
import argparse
import sys
import os
from typing import Optional

# Proje yolunu ekle
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from main import SpeakerDiarizationSystem
from utils import ProjectUtils, AnalysisUtils, VisualizationUtils

def main():
    """Ana pipeline fonksiyonu"""
    parser = argparse.ArgumentParser(description='Konuşmacı Diarizasyon Sistemi')
    parser.add_argument('--audio', '-a', type=str, required=True, 
                       help='Ses dosyası yolu')
    parser.add_argument('--output', '-o', type=str, default='results',
                       help='Çıktı klasörü yolu (varsayılan: results)')
    parser.add_argument('--config', '-c', type=str, default='config/model_config.yaml',
                       help='Konfigürasyon dosyası yolu')
    parser.add_argument('--format', '-f', type=str, choices=['csv', 'excel', 'json', 'all'], 
                       default='all', help='Çıktı formatı')
    parser.add_argument('--visualize', '-v', action='store_true',
                       help='Görselleştirme oluştur')
    
    args = parser.parse_args()
    
    # Çıktı klasörünü oluştur
    ProjectUtils.ensure_directory(args.output)
    
    try:
        print("🚀 Konuşmacı Diarizasyon Pipeline'ı başlatılıyor...")
        print(f"📁 Ses dosyası: {args.audio}")
        print(f"📁 Çıktı klasörü: {args.output}")
        print(f"⚙️  Konfigürasyon: {args.config}")
        
        # Konfigürasyon yükle
        config = {}
        if os.path.exists(args.config):
            config = ProjectUtils.load_config(args.config)
        else:
            print("⚠️  Konfigürasyon dosyası bulunamadı, varsayılan ayarlar kullanılıyor")
        
        # Sistem oluştur
        system = SpeakerDiarizationSystem(config)
        
        # Ses dosyasını işle
        results = system.process_audio_file(args.audio)
        
        if not results:
            print("❌ İşlem başarısız! Sonuç alınamadı.")
            return 1
        
        # Çıktıları kaydet
        base_name = os.path.splitext(os.path.basename(args.audio))[0]
        timestamp = os.path.basename(args.audio).replace('.', '_').replace(' ', '_')
        
        # CSV çıktısı
        if args.format in ['csv', 'all']:
            csv_path = os.path.join(args.output, f'{base_name}_results.csv')
            system.export_results(results, 'csv', csv_path)
        
        # Excel çıktısı
        if args.format in ['excel', 'all']:
            excel_path = os.path.join(args.output, f'{base_name}_results.xlsx')
            system.export_results(results, 'excel', excel_path)
        
        # JSON raporu
        if args.format in ['json', 'all']:
            json_path = os.path.join(args.output, f'{base_name}_report.json')
            system.generate_report(results, json_path)
        
        # Özet rapor
        summary_report = AnalysisUtils.generate_summary_report(results)
        summary_path = os.path.join(args.output, f'{base_name}_summary.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary_report)
        print(f"✅ Özet rapor kaydedildi: {summary_path}")
        
        # Görselleştirme
        if args.visualize:
            print("\n🎨 Görselleştirme oluşturuluyor...")
            
            # Segment dağılımı
            dist_path = os.path.join(args.output, f'{base_name}_distribution.png')
            VisualizationUtils.plot_segment_distribution(results, dist_path)
            
            # Pitch analizi
            pitch_path = os.path.join(args.output, f'{base_name}_pitch_analysis.png')
            VisualizationUtils.plot_pitch_analysis(results, pitch_path)
        
        # Sonuçları ekranda göster
        print("\n" + "="*60)
        print("📋 ÖZET SONUÇLAR")
        print("="*60)
        
        # İlk 10 sonucu göster
        print(f"{i+1:2d}. {result['start_time']:6.1f}s - {result['end_time']:6.1f}s | "
      f"{result['speaker_id']:8} | "
      f"Pitch: {result['pitch_mean']:5.1f}Hz")

        if len(results) > 10:
            print(f"... ve {len(results) - 10} segment daha")
        
        print("\n✅ Pipeline başarıyla tamamlandı!")
        print(f"📁 Tüm çıktılar: {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Pipeline hatası: {e}")
        return 1

def batch_process(audio_directory: str, output_directory: str, config_path: Optional[str] = None):
    """
    Toplu ses dosyası işleme
    """
    try:
        print(f"🔄 Toplu işlem başlatılıyor: {audio_directory}")
        
        # Ses dosyalarını bul
        audio_files = ProjectUtils.get_audio_files(audio_directory)
        
        if not audio_files:
            print("❌ İşlenecek ses dosyası bulunamadı!")
            return
        
        # Konfigürasyon yükle
        config = {}
        if config_path and os.path.exists(config_path):
            config = ProjectUtils.load_config(config_path)
        
        # Sistem oluştur
        system = SpeakerDiarizationSystem(config)
        
        success_count = 0
        
        for audio_file in audio_files:
            try:
                print(f"\n{'='*50}")
                print(f"📄 İşleniyor: {os.path.basename(audio_file)}")
                print(f"{'='*50}")
                
                # Ses dosyasını işle
                results = system.process_audio_file(audio_file)
                
                if results:
                    # Çıktıları kaydet
                    base_name = os.path.splitext(os.path.basename(audio_file))[0]
                    file_output_dir = os.path.join(output_directory, base_name)
                    ProjectUtils.ensure_directory(file_output_dir)
                    
                    # CSV çıktısı
                    csv_path = os.path.join(file_output_dir, f'{base_name}_results.csv')
                    system.export_results(results, 'csv', csv_path)
                    
                    # Özet rapor
                    summary_report = AnalysisUtils.generate_summary_report(results)
                    summary_path = os.path.join(file_output_dir, f'{base_name}_summary.txt')
                    with open(summary_path, 'w', encoding='utf-8') as f:
                        f.write(summary_report)
                    
                    success_count += 1
                    print(f"✅ Başarılı: {os.path.basename(audio_file)}")
                else:
                    print(f"❌ Başarısız: {os.path.basename(audio_file)}")
                    
            except Exception as e:
                print(f"❌ Dosya işleme hatası ({audio_file}): {e}")
                continue
        
        print(f"\n🎉 Toplu işlem tamamlandı!")
        print(f"📊 Başarılı/Toplam: {success_count}/{len(audio_files)}")
        
    except Exception as e:
        print(f"❌ Toplu işlem hatası: {e}")

if __name__ == "__main__":
    # Örnek kullanım
    if len(sys.argv) == 1:
        print("""
🎯 KONUŞMACI DİARİZASYON SİSTEMİ

Kullanım:
  python run_pipeline.py --audio <ses_dosyası> [--output <çıktı_klasörü>]

Örnekler:
  python run_pipeline.py --audio data/raw/ses.wav
  python run_pipeline.py --audio data/raw/ses.wav --output my_results --visualize
  python run_pipeline.py --audio data/raw/ses.wav --format csv --config config/model_config.yaml

Toplu İşlem:
  from run_pipeline import batch_process
  batch_process("data/raw", "batch_results")
        """)
        sys.exit(1)
    
    # Pipeline'ı çalıştır
    sys.exit(main())