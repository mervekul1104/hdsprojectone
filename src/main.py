"""
Ana uygulama - Tüm pipeline'ı birleştirir
"""
import numpy as np
import pandas as pd
import json
from typing import List, Dict, Any, Optional
import os
import sys

# Diğer modülleri içe aktar
from audio_processor import AudioProcessor
from feature_extractor import FeatureExtractor
from speaker_clustering import SpeakerClustering


class SpeakerDiarizationSystem:
    def __init__(self, config: Dict[str, Any] = None):
        """
        Konuşmacı Diarization Sistemi
        
        Args:
            config: Yapılandırma ayarları
        """
        self.config = config or self._default_config()
        
        # Config'den değerleri güvenli şekilde al
        sample_rate = self.config.get('audio', {}).get('sample_rate', 16000)
        clustering_algorithm = self.config.get('clustering', {}).get('algorithm', 'dbscan')
        similarity_threshold = self.config.get('advanced_analysis', {}).get('similarity_threshold', 0.7)
        
        self.audio_processor = AudioProcessor(target_sr=sample_rate)
        self.feature_extractor = FeatureExtractor(sample_rate=sample_rate)
        self.speaker_clustering = SpeakerClustering(
            algorithm=clustering_algorithm,
            similarity_threshold=similarity_threshold
        )

        print("🎉 Gelişmiş Speaker Diarization Sistemi başlatıldı!")
        print(f"⚙️  Sample Rate: {sample_rate}")
        print(f"⚙️  Clustering Algorithm: {clustering_algorithm}")
        print(f"⚙️  Similarity Threshold: {similarity_threshold}")
    
    def _default_config(self) -> Dict[str, Any]:
        """Varsayılan yapılandırma"""
        return {
            'audio': {
                'sample_rate': 16000,
                'min_segment_length': 0.8,  # Daha kısa segmentlere izin ver
                'max_segment_length': 15.0  # Daha uzun segmentlere izin ver
            },
            'clustering': {
                'algorithm': 'dbscan',
                'min_speakers': 1,
                'max_speakers': 10
            },
            'segmentation': {
                'min_segment_length': 0.8,
                'max_segment_length': 15.0,
                'silence_threshold': 0.015,  # Daha hassas sessizlik tespiti
                'min_silence_length': 0.3
            },
            'advanced_analysis': {
                'similarity_threshold': 0.6,  # Daha düşük eşik
                'min_speaker_segments': 1,    # 1 segmentli konuşmacılara izin ver
                'confidence_threshold': 0.4
            }
        }
    
    def process_audio_file(self, audio_path: str) -> Dict[str, Any]:
        """
        Gelişmiş ses analizi - Tüm bilgileri döndür
        
        Args:
            audio_path: Ses dosyası yolu
            
        Returns:
            Tüm analiz sonuçları
        """
        try:
            print(f"\n🎵 GELİŞMİŞ SES ANALİZİ: {os.path.basename(audio_path)}")
            print("=" * 60)
            
            # 1. Ses yükleme ve ön işleme
            print("1. Ses yükleniyor...")
            audio_data, sr = self.audio_processor.load_audio(audio_path)
            processed_audio = self.audio_processor.preprocess_audio(audio_data, sr)
            
            # 2. Segmentasyon - Config'den değerleri GÜVENLİ şekilde al
            min_segment = (self.config.get('audio', {})
                          .get('min_segment_length', 
                               self.config.get('segmentation', {})
                               .get('min_segment_length', 0.8)))
            max_segment = (self.config.get('audio', {})
                          .get('max_segment_length',
                               self.config.get('segmentation', {})
                               .get('max_segment_length', 15.0)))
            
            print(f"2. Segmentasyon: min={min_segment}s, max={max_segment}s")
            
            segments = self.audio_processor.split_into_segments(
                processed_audio, sr,
                min_segment_length=min_segment,
                max_segment_length=max_segment
            )
            
            if not segments:
                print("❌ Hiç segment bulunamadı!")
                return self._create_empty_result(audio_path, sr, audio_data)
            
            print(f"✅ Segmentasyon tamamlandı: {len(segments)} segment")
            
            # 3. Özellik çıkarımı
            print("3. Özellik çıkarımı başlıyor...")
            features_list = []
            valid_segments = []
            
            for i, segment in enumerate(segments):
                try:
                    audio_segment = segment['audio']
                    # 0.2 saniyeden kısa segmentleri atla
                    if len(audio_segment) < int(sr * 0.2):
                        continue
                    
                    features = self.feature_extractor.get_complete_feature_set(audio_segment)
                    if features and 'feature_vector' in features:
                        # Özellik vektörünü düzelt (NaN'ları temizle)
                        feature_vector = np.nan_to_num(features['feature_vector'], nan=0.0)
                        features_list.append(feature_vector)
                        valid_segments.append({
                            'segment_id': i,
                            'start_time': segment['start'],
                            'end_time': segment['end'],
                            'duration': segment['duration'],
                            'audio_features': features,
                            'audio_data': audio_segment
                        })
                        
                except Exception as e:
                    print(f"❌ Segment {i} özellik çıkarım hatası: {e}")
                    continue
            
            print(f"✅ Başarılı özellik çıkarımı: {len(valid_segments)}/{len(segments)} segment")
            
            if not valid_segments:
                print("❌ İşlenebilir segment bulunamadı!")
                return self._create_empty_result(audio_path, sr, audio_data)
            
            # 4. GELİŞMİŞ KONUŞMACI ANALİZİ
            print("4. Konuşmacı analizi başlıyor...")
            feature_vectors = [seg['audio_features']['feature_vector'] for seg in valid_segments]
            
            # Gelişmiş konuşmacı tespiti
            speaker_analysis = self.speaker_clustering.advanced_speaker_detection(feature_vectors)
            
            # Konuşmacı ID'lerini ata
            cluster_labels = np.array(speaker_analysis.get('cluster_labels', []))
            speaker_ids = self.speaker_clustering.assign_speaker_ids(cluster_labels)
            
            # 6. SONUÇLARI BİRLEŞTİR
            print("6. Sonuçlar birleştiriliyor...")
            final_results = self._combine_results(
                valid_segments, speaker_analysis, speaker_ids, audio_path, sr, audio_data
            )
            
            # 7. DETAYLI RAPOR
            self._generate_detailed_report(final_results)
            
            print(f"\n🎉 GELİŞMİŞ ANALİZ TAMAMLANDI!")
            return final_results
            
        except Exception as e:
            print(f"❌ Ses işleme hatası: {e}")
            import traceback
            traceback.print_exc()
            return self._create_empty_result(audio_path, 16000, np.array([]))
    
    def _create_empty_result(self, audio_path: str, sr: int, audio_data: np.ndarray) -> Dict[str, Any]:
        """Boş sonuç oluştur"""
        return {
            'file_info': {
                'filename': os.path.basename(audio_path),
                'duration': len(audio_data) / sr if len(audio_data) > 0 else 0,
                'total_segments': 0,
                'sample_rate': sr
            },
            'speaker_analysis': {
                'speaker_count': 0,
                'speaker_ids': [],
                'cluster_labels': [],
                'embeddings': {},
                'similarity_matrix': {},
                'same_speaker_pairs': [],
                'cluster_stats': {'total_segments': 0, 'total_speakers': 0}
            },
            'segments': [],
            'summary': {
                'total_speakers': 0,
                'same_speaker_alerts': 0,
                'analysis_timestamp': pd.Timestamp.now().isoformat()
            }
        }
    
    def _combine_results(self, valid_segments: List[Dict], speaker_analysis: Dict, 
                     speaker_ids: List[str], 
                     audio_path: str, sr: int, audio_data: np.ndarray) -> Dict[str, Any]:
     """
     Tüm analiz sonuçlarını birleştir (cinsiyet bilgisi hariç)
     """
     # Ana sonuç yapısı
     final_results = {
        'file_info': {
            'filename': os.path.basename(audio_path),
            'duration': len(audio_data) / sr,
            'total_segments': len(valid_segments),
            'sample_rate': sr
        },
        'speaker_analysis': speaker_analysis,
        'segments': [],
        'summary': {
            'total_speakers': speaker_analysis.get('speaker_count', 0),
            'same_speaker_alerts': len(speaker_analysis.get('same_speaker_pairs', [])),
            'analysis_timestamp': pd.Timestamp.now().isoformat()
        }
    }
    
     for i, segment in enumerate(valid_segments):
        speaker_id = speaker_ids[i] if i < len(speaker_ids) else "UNKNOWN"
        
        final_results['segments'].append({
            'segment_id': segment['segment_id'],
            'start_time': round(segment['start_time'], 2),
            'end_time': round(segment['end_time'], 2),
            'duration': round(segment['duration'], 2),
            'speaker_id': speaker_id,
            'pitch_mean': round(segment['audio_features'].get('pitch_mean', 0), 1),
            'energy_mean': round(segment['audio_features'].get('energy_mean', 0), 3)
        })
    
     return final_results

    def _generate_detailed_report(self, results: Dict[str, Any]):
     """Detaylı analiz raporunu yazdır (cinsiyet bilgisi hariç)"""
     if not results:
        return
    
     print(f"\n📊 DETAYLI ANALİZ RAPORU")
     print("=" * 60)
    
     speaker_info = results['speaker_analysis']
    
     print(f"🎯 TOPLAM KONUŞMACI SAYISI: {speaker_info['speaker_count']}")
     print(f"📝 TOPLAM SEGMENT: {results['file_info']['total_segments']}")
     print(f"⏱️  TOPLAM SÜRE: {results['file_info']['duration']:.2f}s")
    
     # Konuşmacı detayları
     if speaker_info['speaker_ids']:
        print(f"\n👥 KONUŞMACI DETAYLARI:")
        for speaker_id in speaker_info['speaker_ids']:
            segment_count = speaker_info['cluster_stats']['speaker_segment_counts'].get(speaker_id, 0)
            print(f"   {speaker_id}: {segment_count} segment")
     else:
        print(f"\n👥 KONUŞMACI DETAYLARI: Hiç konuşmacı tespit edilemedi")
    
     # Benzerlik analizi
     similarities = speaker_info.get('similarity_matrix', {})
     if similarities:
        print(f"\n🔍 KONUŞMACI BENZERLİK ANALİZİ:")
        for pair, data in similarities.items():
            similarity_pct = data['similarity_percentage']
            if data['same_speaker']:
                print(f"   ⚠️  {pair}: AYNI KİŞİ OLABİLİR (%{similarity_pct:.1f} benzer)")
    
     # Aynı kişi uyarıları
     same_speakers = speaker_info.get('same_speaker_pairs', [])
     if same_speakers:
        print(f"\n🚨 UYARI: AYNI KİŞİ OLABİLECEK KONUŞMACILAR:")
        for pair in same_speakers:
            print(f"   • {pair['speaker1']} ↔ {pair['speaker2']} (%{pair['similarity_percentage']:.1f} benzer)")
    
    def generate_report(self, results: Dict[str, Any], output_path: Optional[str] = None) -> None:
        """
        Pipeline uyumluluğu için: detaylı raporu yazdırır ve JSON/CSV gibi çıktı oluşturur.
        """
        # Konsola detaylı rapor bas
        self._generate_detailed_report(results)

        # Eğer output_path verilmişse JSON olarak kaydet
        if output_path:
            try:
                import json
                with open(output_path, "w", encoding="utf-8") as f:
                   json.dump(results, f, ensure_ascii=False, indent=2, default=str)
                print(f"✅ JSON raporu kaydedildi: {output_path}")
            except Exception as e:
                print(f"❌ JSON raporu kaydedilemedi: {e}")


    def export_results(self, results: Dict[str, Any], output_format: str = 'csv', 
                       output_path: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Sonuçları dışa aktar
        """
        try:
            if not results or 'segments' not in results or not results['segments']:
                print("❌ Dışa aktarılacak veri yok!")
                return None
            
            df = pd.DataFrame(results['segments'])
            
            columns_order = [
                'segment_id', 'start_time', 'end_time', 'duration',
                'speaker_id', 'pitch_mean', 'energy_mean'
            ]
            available_columns = [col for col in columns_order if col in df.columns]
            df = df[available_columns]
            
            if output_format.lower() == 'dataframe':
                return df
            elif output_format.lower() == 'csv':
                output_path = output_path or f"{results['file_info']['filename']}_results.csv"
                df.to_csv(output_path, index=False, encoding='utf-8')
                print(f"✅ CSV dosyası kaydedildi: {output_path}")
                return df
            elif output_format.lower() == 'excel':
                output_path = output_path or f"{results['file_info']['filename']}_results.xlsx"
                df.to_excel(output_path, index=False)
                print(f"✅ Excel dosyası kaydedildi: {output_path}")
                return df
            else:
                print(f"❌ Geçersiz çıktı formatı: {output_format}")
                return None

        except Exception as e:
            print(f"❌ Dışa aktarma hatası: {e}")
            return None
