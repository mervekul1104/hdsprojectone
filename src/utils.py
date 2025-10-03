"""
Yardımcı fonksiyonlar ve araçlar
"""
import os
import json
import yaml
import numpy as np
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class ProjectUtils:
    @staticmethod
    def ensure_directory(directory_path: str) -> bool:
        """
        Klasörün var olduğundan emin ol, yoksa oluştur
        """
        try:
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)
                print(f"✅ Klasör oluşturuldu: {directory_path}")
            return True
        except Exception as e:
            print(f"❌ Klasör oluşturma hatası: {e}")
            return False

    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """
        YAML konfigürasyon dosyasını yükle
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
            print(f"✅ Konfigürasyon yüklendi: {config_path}")
            return config
        except Exception as e:
            print(f"❌ Konfigürasyon yükleme hatası: {e}")
            return {}

    @staticmethod
    def save_config(config: Dict[str, Any], config_path: str) -> bool:
        """
        YAML konfigürasyon dosyasını kaydet
        """
        try:
            with open(config_path, 'w', encoding='utf-8') as file:
                yaml.dump(config, file, default_flow_style=False, allow_unicode=True)
            print(f"✅ Konfigürasyon kaydedildi: {config_path}")
            return True
        except Exception as e:
            print(f"❌ Konfigürasyon kaydetme hatası: {e}")
            return False

    @staticmethod
    def get_audio_files(directory_path: str) -> List[str]:
        """
        Dizindeki ses dosyalarını listele
        """
        audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.aac'}
        audio_files = []
        
        try:
            for file in os.listdir(directory_path):
                if any(file.lower().endswith(ext) for ext in audio_extensions):
                    audio_files.append(os.path.join(directory_path, file))
            
            print(f"✅ {len(audio_files)} ses dosyası bulundu: {directory_path}")
            return audio_files
        except Exception as e:
            print(f"❌ Ses dosyası listeleme hatası: {e}")
            return []

    @staticmethod
    def format_time(seconds: float) -> str:
        """
        Saniyeyi dakika:saniye formatına çevir
        """
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"

    @staticmethod
    def calculate_audio_duration(audio_data: np.ndarray, sample_rate: int) -> float:
        """
        Ses verisinin süresini hesapla
        """
        return len(audio_data) / sample_rate

class VisualizationUtils:
    @staticmethod
    def plot_segment_distribution(results: List[Dict[str, Any]], save_path: Optional[str] = None):
        """
        Segment dağılımını görselleştir
        """
        try:
            speakers = [result['speaker_id'] for result in results]
            durations = [result['duration'] for result in results]
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Konuşmacı Analizi - Segment Dağılımı', fontsize=16)
            
            # Konuşmacı dağılımı
            speaker_counts = {spk: speakers.count(spk) for spk in set(speakers)}
            axes[0, 0].pie(speaker_counts.values(), labels=speaker_counts.keys(), autopct='%1.1f%%')
            axes[0, 0].set_title('Konuşmacı Dağılımı (Segment Sayısı)')
            
            # Süre dağılımı
            axes[0, 1].hist(durations, bins=20, alpha=0.7, edgecolor='black')
            axes[0, 1].set_title('Segment Süre Dağılımı')
            axes[0, 1].set_xlabel('Süre (saniye)')
            axes[0, 1].set_ylabel('Frekans')
            
            # Zaman çizelgesi
            times = [(result['start_time'] + result['end_time']) / 2 for result in results]
            axes[1, 0].scatter(times, speakers, c=[hash(spk) for spk in speakers], alpha=0.6)
            axes[1, 0].set_title('Konuşmacı Zaman Çizelgesi')
            axes[1, 0].set_xlabel('Zaman (saniye)')
            axes[1, 0].set_ylabel('Konuşmacı')
            
            # Boş bir subplot kapatmak için
            axes[1, 1].axis("off")
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✅ Görsel kaydedildi: {save_path}")
            
            plt.show()
        
        except Exception as e:
            print(f"❌ Görselleştirme hatası: {e}")

    @staticmethod
    def plot_pitch_analysis(results: List[Dict[str, Any]], save_path: Optional[str] = None):
        """
        Pitch analizini görselleştir
        """
        try:
            pitches = [result['pitch_mean'] for result in results if result['pitch_mean'] > 0]
            
            if not pitches:
                print("⚠️  Pitch verisi yok")
                return
            
            fig, ax = plt.subplots(figsize=(8, 5))
            fig.suptitle('Pitch Analizi', fontsize=16)
            
            # Pitch dağılımı
            ax.hist(pitches, bins=20, alpha=0.7, edgecolor='black')
            ax.set_xlabel('Pitch (Hz)')
            ax.set_ylabel('Frekans')
            ax.set_title('Pitch Dağılımı')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✅ Pitch görseli kaydedildi: {save_path}")
            
            plt.show()
        
        except Exception as e:
            print(f"❌ Pitch görselleştirme hatası: {e}")

class AnalysisUtils:
    @staticmethod
    def calculate_speaker_statistics(results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Konuşmacı istatistiklerini hesapla (gender bilgisi olmadan)
        """
        try:
            stats = {
                'total_segments': len(results.get("segments", [])),
                'total_duration': sum(result['duration'] for result in results.get("segments", [])),
                'speakers': {}
            }
            
            for result in results.get("segments", []):
                speaker = result['speaker_id']
                
                # Konuşmacı istatistikleri
                if speaker not in stats['speakers']:
                    stats['speakers'][speaker] = {
                        'segment_count': 0,
                        'total_duration': 0.0,
                        'avg_pitch': []
                    }
                
                stats['speakers'][speaker]['segment_count'] += 1
                stats['speakers'][speaker]['total_duration'] += result['duration']
                stats['speakers'][speaker]['avg_pitch'].append(result['pitch_mean'])
            
            # Ortalama pitch hesapla
            for speaker in stats['speakers']:
                pitches = stats['speakers'][speaker]['avg_pitch']
                stats['speakers'][speaker]['avg_pitch'] = (
                    np.mean([p for p in pitches if p > 0]) if pitches else 0
                )
            
            return stats
            
        except Exception as e:
            print(f"❌ İstatistik hesaplama hatası: {e}")
            return {}

    @staticmethod
    def generate_summary_report(results: List[Dict[str, Any]]) -> str:
        """
        Özet rapor oluştur
        """
        try:
            stats = AnalysisUtils.calculate_speaker_statistics(results)
            
            report = f"""
📊 KONUŞMACI DİARİZASYON ANALİZ RAPORU
====================================
Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Toplam Segment: {stats.get('total_segments', 0)}
Toplam Süre: {stats.get('total_duration', 0):.2f} saniye

KONUŞMACI İSTATİSTİKLERİ:
"""
            for speaker, speaker_stats in stats.get('speakers', {}).items():
                report += f"""
  {speaker}:
    - Segment Sayısı: {speaker_stats.get('segment_count', 0)}
    - Toplam Süre: {speaker_stats.get('total_duration', 0):.2f} saniye
    - Ortalama Pitch: {speaker_stats.get('avg_pitch', 0):.1f} Hz
"""
            return report

        except Exception as e:
            print(f"❌ Rapor oluşturma hatası: {e}")
            return "Rapor oluşturulamadı."
