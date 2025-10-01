"""
Cinsiyet sınıflandırma modülü - Gelişmiş özellikler
"""
import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class GenderClassifier:
    def __init__(self, model_type: str = "rule_based", use_advanced_features: bool = True):
        self.model_type = model_type
        self.use_advanced_features = use_advanced_features
        self.scaler = StandardScaler()
        self.model = None
        self.is_trained = False
        print(f"🔧 GenderClassifier başlatıldı - Model: {model_type}")
    
    def extract_gender_features(self, audio_features: Dict[str, Any]) -> np.ndarray:
        """
        Cinsiyet tahmini için özellikleri çıkar - DÜZELTİLMİŞ
        """
        try:
            # Pitch tabanlı özellikler (en önemlileri)
            pitch_features = [
                audio_features.get('pitch_mean', 0.0),
                audio_features.get('pitch_std', 0.0),
                audio_features.get('pitch_median', 0.0),
                audio_features.get('pitch_range', 0.0),
                audio_features.get('voiced_ratio', 0.0),
            ]
            
            # MFCC özelliklerinden bazıları (ilk 5'i)
            mfcc_features = audio_features.get('mfcc', [])
            if isinstance(mfcc_features, np.ndarray) and len(mfcc_features) > 5:
                mfcc_selected = mfcc_features[:5].tolist()
            elif isinstance(mfcc_features, list) and len(mfcc_features) > 5:
                mfcc_selected = mfcc_features[:5]
            else:
                mfcc_selected = [0.0] * 5
            
            # Spektral özellikler
            spectral_features = [
                audio_features.get('spectral_centroid_mean', 0.0),
                audio_features.get('spectral_bandwidth_mean', 0.0),
            ]
            
            # Enerji özellikleri
            energy_features = [
                audio_features.get('energy_mean', 0.0),
                audio_features.get('zcr_mean', 0.0),
            ]
            
            # Formant özellikleri (cinsiyet ayrımında önemli)
            formant_features = [
                audio_features.get('f1_mean', 0.0),
                audio_features.get('f2_mean', 0.0),
                audio_features.get('f3_mean', 0.0),
            ]
            
            # Tüm özellikleri birleştir
            all_features = pitch_features + mfcc_selected + spectral_features + energy_features + formant_features
            gender_features = np.array(all_features, dtype=np.float32)
            
            # NaN değerleri temizle
            gender_features = np.nan_to_num(gender_features, nan=0.0, posinf=0.0, neginf=0.0)
            
            return gender_features
            
        except Exception as e:
            print(f"❌ Cinsiyet özellik çıkarım hatası: {e}")
            return np.zeros(15, dtype=np.float32)  # Güncellenmiş boyut
    
    def train_classifier(self, training_data: List[Tuple[np.ndarray, str]]):
        """
        Cinsiyet sınıflandırıcıyı eğit
        """
        try:
            if not training_data:
                print("⚠️  Eğitim verisi yok, kural tabanlı yöntem kullanılacak")
                return
            
            X = [data[0] for data in training_data]
            y = [data[1] for data in training_data]
            
            # Özellik boyutunu kontrol et
            feature_size = len(X[0])
            print(f"📊 Eğitim özellik boyutu: {feature_size}")
            
            # Ölçeklendirme
            X_scaled = self.scaler.fit_transform(X)
            
            # Model seçimi ve eğitim
            if self.model_type == "random_forest":
                self.model = RandomForestClassifier(
                    n_estimators=100, 
                    random_state=42,
                    max_depth=10,
                    min_samples_split=5
                )
            elif self.model_type == "svm":
                self.model = SVC(
                    probability=True, 
                    random_state=42,
                    kernel='rbf',
                    C=1.0
                )
            else:
                self.model = RandomForestClassifier(n_estimators=50, random_state=42)
            
            self.model.fit(X_scaled, y)
            self.is_trained = True
            
            print(f"✅ Cinsiyet sınıflandırıcı eğitildi - {len(training_data)} örnek")
            
            # Model performansını göster
            train_score = self.model.score(X_scaled, y)
            print(f"   Eğitim doğruluğu: {train_score:.3f}")
            
        except Exception as e:
            print(f"❌ Sınıflandırıcı eğitim hatası: {e}")
            self.is_trained = False
    
    def predict_gender(self, audio_features: Dict[str, Any]) -> Tuple[str, float]:
        """
        Cinsiyet tahmini yap
        
        Returns:
            (cinsiyet, güven skoru)
        """
        try:
            # Özellikleri çıkar
            gender_features = self.extract_gender_features(audio_features)
            
            if self.is_trained and self.model is not None:
                # Makine öğrenmesi modeli ile tahmin
                features_scaled = self.scaler.transform([gender_features])
                prediction = self.model.predict(features_scaled)[0]
                probabilities = self.model.predict_proba(features_scaled)[0]
                probability = np.max(probabilities)
                
                # Güven skorunu iyileştir
                confidence = self._adjust_confidence(prediction, probability, audio_features)
                
                return prediction, float(confidence)
            else:
                # Kural tabanlı tahmin (pitch değerlerine göre)
                return self._rule_based_prediction(audio_features)
                
        except Exception as e:
            print(f"❌ Cinsiyet tahmin hatası: {e}")
            return "BELIRSIZ", 0.0
    
    def _rule_based_prediction(self, audio_features: Dict[str, Any]) -> Tuple[str, float]:
        """
        Pitch değerlerine dayalı kural tabanlı cinsiyet tahmini
        """
        pitch_mean = audio_features.get('pitch_mean', 0)
        pitch_std = audio_features.get('pitch_std', 0)
        pitch_median = audio_features.get('pitch_median', 0)
        voiced_ratio = audio_features.get('voiced_ratio', 0)
        spectral_centroid = audio_features.get('spectral_centroid_mean', 0)
        
        # Formant değerlerini al
        f1_mean = audio_features.get('f1_mean', 0)
        f2_mean = audio_features.get('f2_mean', 0)
        f3_mean = audio_features.get('f3_mean', 0)
        
        # Geçersiz pitch değerlerini kontrol et
        if pitch_mean <= 50 or pitch_mean >= 400 or voiced_ratio < 0.2:
            return "BELIRSIZ", 0.1
        
        # Temel güven skoru
        base_confidence = 0.7
        
        # Pitch değerine göre temel tahmin
        if pitch_mean < 135:
            gender = "ERKEK"
            # 80-135 Hz arası ideal erkek aralığı
            if 80 <= pitch_mean <= 135:
                base_confidence = 0.85
            else:
                base_confidence = 0.6
                
        elif pitch_mean > 185:
            gender = "KADIN"
            # 170-280 Hz arası ideal kadın aralığı
            if 170 <= pitch_mean <= 280:
                base_confidence = 0.85
            else:
                base_confidence = 0.6
                
        else:
            # Belirsiz aralık (135-185 Hz)
            gender = "BELIRSIZ"
            base_confidence = 0.4
            
            # Formantlara göre karar ver
            if f1_mean > 400 and f2_mean > 1400:  # Kadın formant aralığı
                gender = "KADIN"
                base_confidence = 0.7
            elif f1_mean < 350 and f2_mean < 1200:  # Erkek formant aralığı
                gender = "ERKEK"
                base_confidence = 0.7
            # Spektral özelliklere göre ince ayar
            elif spectral_centroid > 1800 and pitch_mean > 150:
                gender = "KADIN"
                base_confidence = 0.65
            elif spectral_centroid < 1400 and pitch_mean < 170:
                gender = "ERKEK"
                base_confidence = 0.65
        
        # Güven ayarlamaları
        confidence = base_confidence
        
        # Standart sapmaya göre ayarla
        if pitch_std > 40:  # Çok değişken pitch
            confidence *= 0.8
        
        # Pitch medyan ile uyum kontrolü
        if abs(pitch_mean - pitch_median) > 20:
            confidence *= 0.9
        
        # Voiced ratio düşükse güveni azalt
        if voiced_ratio < 0.5:
            confidence *= 0.8
            
        # Formant tutarlılık kontrolü
        if gender == "ERKEK" and f1_mean > 450:
            confidence *= 0.9
        elif gender == "KADIN" and f1_mean < 300:
            confidence *= 0.9
        
        # Son güven kontrolü
        confidence = max(0.1, min(0.95, confidence))
        
        return gender, confidence
    
    def _adjust_confidence(self, prediction: str, ml_confidence: float, audio_features: Dict[str, Any]) -> float:
        """
        ML tahmin güvenini ses özelliklerine göre ayarla
        """
        pitch_mean = audio_features.get('pitch_mean', 0)
        pitch_std = audio_features.get('pitch_std', 0)
        voiced_ratio = audio_features.get('voiced_ratio', 0)
        f1_mean = audio_features.get('f1_mean', 0)
        
        adjusted_confidence = ml_confidence
        
        # Pitch aralığı kontrolü
        if prediction == "ERKEK" and not (70 <= pitch_mean <= 160):
            adjusted_confidence *= 0.8
        elif prediction == "KADIN" and not (150 <= pitch_mean <= 300):
            adjusted_confidence *= 0.8
        
        # Formant kontrolü
        if prediction == "ERKEK" and f1_mean > 450:
            adjusted_confidence *= 0.8
        elif prediction == "KADIN" and f1_mean < 300:
            adjusted_confidence *= 0.8
        
        # Standart sapma kontrolü
        if pitch_std > 50:
            adjusted_confidence *= 0.7
        
        # Voiced ratio kontrolü
        if voiced_ratio < 0.4:
            adjusted_confidence *= 0.6
        
        return max(0.1, min(0.95, adjusted_confidence))
    
    def validate_prediction(self, prediction: str, confidence: float, audio_features: Dict[str, Any] = None) -> bool:
        """
        Tahminin geçerliliğini kontrol et
        """
        if confidence < 0.4:  # Daha yüksek eşik
            return False
        
        if prediction == "BELIRSIZ":
            return False
        
        # Ek kontroller
        if audio_features:
            pitch_mean = audio_features.get('pitch_mean', 0)
            voiced_ratio = audio_features.get('voiced_ratio', 0)
            
            if pitch_mean <= 50 or pitch_mean >= 400:
                return False
            
            if voiced_ratio < 0.3:
                return False
        
        return True
    
    def batch_predict_gender(self, features_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Toplu cinsiyet tahmini yap
        """
        results = []
        
        print(f"   🔍 {len(features_list)} segment için cinsiyet tahmini yapılıyor...")
        
        for i, features in enumerate(features_list):
            gender, confidence = self.predict_gender(features)
            is_valid = self.validate_prediction(gender, confidence, features)
            
            results.append({
                'segment_id': i,
                'gender': gender,
                'confidence': round(confidence, 3),
                'is_valid': is_valid,
                'pitch_mean': features.get('pitch_mean', 0),
                'voiced_ratio': features.get('voiced_ratio', 0)
            })
        
        return results
    
    def advanced_gender_analysis(self, audio_features: List[Dict[str, Any]], speaker_ids: List[str] = None) -> Dict[str, Any]:
        """
        Gelişmiş cinsiyet analizi - Konuşmacı bazlı
        
        Args:
            audio_features: Ses özellikleri listesi
            speaker_ids: Konuşmacı ID listesi (opsiyonel)
            
        Returns:
            Detaylı cinsiyet analiz sonuçları
        """
        try:
            print("🎯 Gelişmiş cinsiyet analizi başlıyor...")
            
            # Segment bazlı tahminler
            segment_results = self.batch_predict_gender(audio_features)
            
            # Konuşmacı bazlı analiz
            speaker_analysis = {}
            if speaker_ids and len(speaker_ids) == len(audio_features):
                speaker_analysis = self._analyze_speaker_genders(segment_results, speaker_ids)
            
            # İstatistikler
            statistics = self.get_detailed_statistics(segment_results, speaker_analysis)
            
            # Sonuçları birleştir
            advanced_results = {
                'segment_predictions': segment_results,
                'speaker_analysis': speaker_analysis,
                'statistics': statistics,
                'analysis_summary': self._generate_analysis_summary(statistics)
            }
            
            print("✅ Gelişmiş cinsiyet analizi tamamlandı")
            return advanced_results
            
        except Exception as e:
            print(f"❌ Gelişmiş cinsiyet analizi hatası: {e}")
            return {}
    
    def _analyze_speaker_genders(self, segment_results: List[Dict[str, Any]], speaker_ids: List[str]) -> Dict[str, Any]:
        """
        Konuşmacı bazlı cinsiyet analizi yap
        """
        speaker_genders = {}
        
        for i, (result, speaker_id) in enumerate(zip(segment_results, speaker_ids)):
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
                # En sık tahmin edilen cinsiyet
                gender_counter = Counter(data['gender_predictions'])
                dominant_gender, dominant_count = gender_counter.most_common(1)[0]
                
                # Ortalama güven
                avg_confidence = np.mean(data['confidences'])
                
                # Ortalama pitch
                avg_pitch = np.mean([p for p in data['pitch_values'] if p > 0])
                
                # Güven skorunu (segment sayısı ve tutarlılık)
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
        
        return final_speaker_genders
    
    def get_detailed_statistics(self, segment_results: List[Dict[str, Any]], speaker_analysis: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Detaylı istatistikler hesapla
        """
        genders = [pred['gender'] for pred in segment_results]
        confidences = [pred['confidence'] for pred in segment_results]
        valid_predictions = [pred for pred in segment_results if pred['is_valid']]
        valid_confidences = [pred['confidence'] for pred in valid_predictions]
        
        stats = {
            'total_predictions': len(segment_results),
            'valid_predictions': len(valid_predictions),
            'valid_ratio': len(valid_predictions) / len(segment_results) if segment_results else 0,
            'gender_distribution': dict(Counter(genders)),
            'valid_gender_distribution': dict(Counter([p['gender'] for p in valid_predictions])),
            'average_confidence': float(np.mean(confidences)) if confidences else 0.0,
            'average_valid_confidence': float(np.mean(valid_confidences)) if valid_confidences else 0.0,
            'confidence_std': float(np.std(confidences)) if confidences else 0.0,
        }
        
        # Konuşmacı istatistikleri
        if speaker_analysis:
            stats['speaker_gender_distribution'] = dict(Counter(
                [data['final_gender'] for data in speaker_analysis.values()]
            ))
            stats['total_speakers'] = len(speaker_analysis)
            stats['speaker_consistency'] = np.mean([data['consistency'] for data in speaker_analysis.values()])
        
        return stats
    
    def _generate_analysis_summary(self, statistics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analiz özeti oluştur
        """
        valid_ratio = statistics.get('valid_ratio', 0)
        avg_confidence = statistics.get('average_valid_confidence', 0)
        
        # Kalite değerlendirmesi
        if valid_ratio >= 0.8 and avg_confidence >= 0.7:
            quality = "YÜKSEK"
        elif valid_ratio >= 0.6 and avg_confidence >= 0.5:
            quality = "ORTA"
        else:
            quality = "DÜŞÜK"
        
        return {
            'quality_assessment': quality,
            'valid_prediction_ratio': round(valid_ratio, 3),
            'average_confidence': round(avg_confidence, 3),
            'primary_gender': max(statistics.get('valid_gender_distribution', {}).items(), key=lambda x: x[1])[0] if statistics.get('valid_gender_distribution') else "BELIRSIZ"
        }
    
    def get_gender_statistics(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Basit istatistikler (geriye dönük uyumluluk için)
        """
        return self.get_detailed_statistics(predictions)

# Test fonksiyonu
if __name__ == "__main__":
    # Test için örnek özellikler oluştur
    test_features = {
        'pitch_mean': 120.0,  # Erkek aralığı
        'pitch_std': 15.0,
        'pitch_median': 118.0,
        'pitch_range': 45.0,
        'voiced_ratio': 0.8,
        'mfcc': np.random.randn(13),
        'spectral_centroid_mean': 1200.0,
        'energy_mean': 0.1,
        'zcr_mean': 0.05,
        'f1_mean': 350.0,
        'f2_mean': 1100.0,
        'f3_mean': 2400.0
    }
    
    classifier = GenderClassifier(use_advanced_features=True)
    gender, confidence = classifier.predict_gender(test_features)
    
    print("✅ Cinsiyet sınıflandırma testi başarılı!")
    print(f"Tahmin: {gender}, Güven: {confidence:.3f}")
    
    # Özellik çıkarım testi
    features = classifier.extract_gender_features(test_features)
    print(f"Özellik boyutu: {len(features)}")
    print(f"Özellikler: {features}")
    
    # Gelişmiş analiz testi
    batch_features = [test_features] * 5
    speaker_ids = ["SPK_01", "SPK_01", "SPK_02", "SPK_02", "SPK_01"]
    
    advanced_results = classifier.advanced_gender_analysis(batch_features, speaker_ids)
    
    print(f"\n🎯 Gelişmiş Analiz Sonuçları:")
    print(f"   Kalite: {advanced_results['analysis_summary']['quality_assessment']}")
    print(f"   Geçerli Tahmin Oranı: {advanced_results['analysis_summary']['valid_prediction_ratio']:.1%}")
    
    for speaker, data in advanced_results['speaker_analysis'].items():
        print(f"   {speaker}: {data['final_gender']} (%{data['confidence']*100:.0f} güven)")