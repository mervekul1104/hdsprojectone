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
from gender_classifier import GenderClassifier

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
        self.gender_classifier = GenderClassifier()
        
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
            'gender_classification': {
                'male_pitch_range': [80, 160],
                'female_pitch_range': [150, 280]
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
            
            # 5. GELİŞMİŞ CİNSİYET ANALİZİ
            print("5. Cinsiyet analizi başlıyor...")
            gender_analysis = self._advanced_gender_analysis(
                [seg['audio_features'] for seg in valid_segments],
                speaker_ids
            )
            
            # 6. SONUÇLARI BİRLEŞTİR
            print("6. Sonuçlar birleştiriliyor...")
            final_results = self._combine_results(
                valid_segments, speaker_analysis, gender_analysis, speaker_ids, audio_path, sr, audio_data
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
            'gender_analysis': {
                'segment_genders': [],
                'speaker_genders': {},
                'gender_statistics': {
                    'total_predictions': 0,
                    'valid_predictions': 0,
                    'gender_distribution': {}
                }
            },
            'segments': [],
            'summary': {
                'total_speakers': 0,
                'same_speaker_alerts': 0,
                'analysis_timestamp': pd.Timestamp.now().isoformat()
            }
        }
    
    def _advanced_gender_analysis(self, audio_features: List[Dict[str, Any]], speaker_ids: List[str]) -> Dict[str, Any]:
        """
        Gelişmiş cinsiyet analizi - Konuşmacı bazlı
        """
        try:
            # Segment bazlı cinsiyet tahmini
            segment_genders = self.gender_classifier.batch_predict_gender(audio_features)
            
            # Konuşmacı bazlı cinsiyet analizi
            speaker_genders = {}
            for i, (result, speaker_id) in enumerate(zip(segment_genders, speaker_ids)):
                if speaker_id != "UNKNOWN" and result['is_valid']:
                    if speaker_id not in speaker_genders:
                        speaker_genders[speaker_id] = {
                            'gender_predictions': [],
                            'confidences': [],
                            'segment_count': 0,
                            'pitch_values': []
                        }
                    
                    speaker_genders[speaker_id]['gender_predictions'].append(result['gender'])
                    speaker_genders[speaker_id]['confidences'].append(result['confidence'])
                    speaker_genders[speaker_id]['segment_count'] += 1
                    speaker_genders[speaker_id]['pitch_values'].append(result['pitch_mean'])
            
            # Her konuşmacı için nihai cinsiyet kararı ver
            final_speaker_genders = {}
            for speaker_id, data in speaker_genders.items():
                if data['gender_predictions']:
                    from collections import Counter
                    gender_counter = Counter(data['gender_predictions'])
                    dominant_gender, dominant_count = gender_counter.most_common(1)[0]
                    
                    avg_confidence = np.mean(data['confidences'])
                    avg_pitch = np.mean([p for p in data['pitch_values'] if p > 0])
                    
                    consistency = dominant_count / len(data['gender_predictions'])
                    confidence_score = avg_confidence * consistency
                    
                    final_speaker_genders[speaker_id] = {
                        'final_gender': dominant_gender,
                        'confidence': round(confidence_score, 3),
                        'segment_count': data['segment_count'],
                        'consistency': round(consistency, 3),
                        'average_pitch': round(avg_pitch, 1),
                        'all_predictions': data['gender_predictions'],
                        'prediction_confidence': round(avg_confidence, 3)
                    }
            
            return {
                'segment_genders': segment_genders,
                'speaker_genders': final_speaker_genders,
                'gender_statistics': self.gender_classifier.get_gender_statistics(segment_genders)
            }
            
        except Exception as e:
            print(f"❌ Cinsiyet analizi hatası: {e}")
            return {
                'segment_genders': [],
                'speaker_genders': {},
                'gender_statistics': {'total_predictions': 0, 'valid_predictions': 0}
            }
    
    def _combine_results(self, valid_segments: List[Dict], speaker_analysis: Dict, 
                        gender_analysis: Dict, speaker_ids: List[str], 
                        audio_path: str, sr: int, audio_data: np.ndarray) -> Dict[str, Any]:
        """
        Tüm analiz sonuçlarını birleştir
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
            'gender_analysis': gender_analysis,
            'segments': [],
            'summary': {
                'total_speakers': speaker_analysis.get('speaker_count', 0),
                'same_speaker_alerts': len(speaker_analysis.get('same_speaker_pairs', [])),
                'analysis_timestamp': pd.Timestamp.now().isoformat()
            }
        }
        
        # Segment detaylarını ekle
        for i, segment in enumerate(valid_segments):
            speaker_id = speaker_ids[i] if i < len(speaker_ids) else "UNKNOWN"
            segment_gender = gender_analysis['segment_genders'][i] if i < len(gender_analysis['segment_genders']) else {'gender': 'BELIRSIZ', 'confidence': 0}
            speaker_gender = gender_analysis['speaker_genders'].get(speaker_id, {})
            
            final_results['segments'].append({
                'segment_id': segment['segment_id'],
                'start_time': round(segment['start_time'], 2),
                'end_time': round(segment['end_time'], 2),
                'duration': round(segment['duration'], 2),
                'speaker_id': speaker_id,
                'gender': segment_gender.get('gender', 'BELIRSIZ'),
                'gender_confidence': segment_gender.get('confidence', 0),
                'speaker_gender': speaker_gender.get('final_gender', 'BELIRSIZ'),
                'speaker_gender_confidence': speaker_gender.get('confidence', 0),
                'pitch_mean': round(segment['audio_features'].get('pitch_mean', 0), 1),
                'energy_mean': round(segment['audio_features'].get('energy_mean', 0), 3),
                'is_valid_gender': segment_gender.get('is_valid', False)
            })
        
        return final_results
    
    def _generate_detailed_report(self, results: Dict[str, Any]):
        """Detaylı analiz raporunu yazdır"""
        if not results:
            return
        
        print(f"\n📊 DETAYLI ANALİZ RAPORU")
        print("=" * 60)
        
        speaker_info = results['speaker_analysis']
        gender_info = results['gender_analysis']
        
        print(f"🎯 TOPLAM KONUŞMACI SAYISI: {speaker_info['speaker_count']}")
        print(f"📝 TOPLAM SEGMENT: {results['file_info']['total_segments']}")
        print(f"⏱️  TOPLAM SÜRE: {results['file_info']['duration']:.2f}s")
        
        # Konuşmacı detayları
        if speaker_info['speaker_ids']:
            print(f"\n👥 KONUŞMACI DETAYLARI:")
            for speaker_id in speaker_info['speaker_ids']:
                segment_count = speaker_info['cluster_stats']['speaker_segment_counts'].get(speaker_id, 0)
                gender_data = gender_info['speaker_genders'].get(speaker_id, {})
                gender = gender_data.get('final_gender', 'BELIRSIZ')
                confidence = gender_data.get('confidence', 0)
                
                print(f"   {speaker_id}: {segment_count} segment | {gender} (%{confidence*100:.0f})")
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
        
        # Cinsiyet istatistikleri
        gender_stats = gender_info.get('gender_statistics', {})
        if gender_stats:
            print(f"\n🚻 CİNSİYET DAĞILIMI:")
            print(f"   Geçerli Tahmin: {gender_stats.get('valid_predictions', 0)}/{gender_stats.get('total_predictions', 0)}")
            if gender_stats.get('average_confidence'):
                print(f"   Ortalama Güven: %{gender_stats['average_confidence']*100:.0f}")
            
            for gender, count in gender_stats.get('gender_distribution', {}).items():
                print(f"   {gender}: {count} segment")
    
    def export_results(self, results: Dict[str, Any], output_format: str = 'csv', 
                      output_path: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Sonuçları dışa aktar
        
        Args:
            results: Analiz sonuçları
            output_format: 'csv', 'excel', veya 'dataframe'
            output_path: Çıktı dosyası yolu
            
        Returns:
            DataFrame
        """
        try:
            if not results or 'segments' not in results or not results['segments']:
                print("❌ Dışa aktarılacak veri yok!")
                return None
            
            # DataFrame oluştur
            df = pd.DataFrame(results['segments'])
            
            # Sütunları düzenle
            columns_order = ['segment_id', 'start_time', 'end_time', 'duration', 
                           'speaker_id', 'gender', 'gender_confidence', 'pitch_mean', 'energy_mean']
            
            # Sadece mevcut sütunları kullan
            available_columns = [col for col in columns_order if col in df.columns]
            df = df[available_columns]
            
            if output_format.lower() == 'dataframe':
                return df
            
            elif output_format.lower() == 'csv':
                if not output_path:
                    base_name = results['file_info'].get('filename', 'analysis')
                    output_path = f"{base_name}_results.csv"
                
                df.to_csv(output_path, index=False, encoding='utf-8')
                print(f"✅ CSV dosyası kaydedildi: {output_path}")
                return df
                
            elif output_format.lower() == 'excel':
                if not output_path:
                    base_name = results['file_info'].get('filename', 'analysis')
                    output_path = f"{base_name}_results.xlsx"
                
                df.to_excel(output_path, index=False)
                print(f"✅ Excel dosyası kaydedildi: {output_path}")
                return df
            
            else:
                print(f"❌ Geçersiz çıktı formatı: {output_format}")
                return None
            
        except Exception as e:
            print(f"❌ Dışa aktarma hatası: {e}")
            return None
    
    def export_detailed_results(self, results: Dict[str, Any], output_path: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Detaylı sonuçları dışa aktar
        """
        return self.export_results(results, 'excel', output_path)

# Kullanım örneği
if __name__ == "__main__":
    # Sistem oluştur
    diarization_system = SpeakerDiarizationSystem()
    
    # Test ses dosyası yolu
    test_audio_path = "data/raw/test_audio.wav"
    
    # Eğer test dosyası yoksa, demo modunda çalış
    if not os.path.exists(test_audio_path):
        print("ℹ️  Test ses dosyası bulunamadı, demo modunda çalışılıyor...")
        # Basit test sesi oluştur
        import soundfile as sf
        import numpy as np
        t = np.linspace(0, 5, 5 * 16000)
        test_audio = 0.3 * np.sin(2 * np.pi * 440 * t)
        sf.write(test_audio_path, test_audio, 16000)
        print(f"✅ Test sesi oluşturuldu: {test_audio_path}")
    
    # Ses dosyasını işle
    results = diarization_system.process_audio_file(test_audio_path)
    
    if results and results['segments']:
        # Sonuçları dışa aktar
        diarization_system.export_results(results, "analysis_results.csv")
        
        # Basit segment görüntüleme
        print("\n📋 İLK 10 SEGMENT:")
        segments = results.get('segments', [])
        for i, result in enumerate(segments[:10]):
            print(f"   {result['segment_id']:2d}. {result['start_time']:5.1f}s - {result['end_time']:5.1f}s | "
                  f"{result['speaker_id']:6} | {result['gender']:6} | "
                  f"Pitch: {result['pitch_mean']:5.1f}Hz")
        
        if len(segments) > 10:
            print(f"   ... ve {len(segments) - 10} segment daha")
    else:
        print("❌ Analiz sonucu alınamadı!")