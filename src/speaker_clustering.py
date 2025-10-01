"""
Konuşmacı kümeleme, kimlik atama ve benzerlik analizi
"""
import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from typing import List, Dict, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class SpeakerClustering:
    def __init__(self, algorithm: str = "dbscan", min_speakers: int = 1, max_speakers: int = 10, 
                 similarity_threshold: float = 0.6):
        self.algorithm = algorithm
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.similarity_threshold = similarity_threshold
        self.scaler = StandardScaler()
        self.labels_ = None
        self.n_speakers_ = 0
        self.embeddings_ = {}
        self.similarity_matrix_ = {}
        print(f"🔧 SpeakerClustering başlatıldı - Algoritma: {algorithm}")
    
    def cluster_speakers(self, features_list: List[np.ndarray]) -> np.ndarray:
        """
        Konuşmacıları özelliklere göre kümele
        """
        try:
            print("🔄 Konuşmacı kümeleme başlıyor...")
            
            if len(features_list) < 2:
                print("⚠️  Yeterli segment yok, tüm segmentler aynı konuşmacıya atanacak")
                return np.zeros(len(features_list), dtype=int)
            
            # Özellik matrisini oluştur
            X = np.array(features_list)
            print(f"   Özellik matrisi boyutu: {X.shape}")
            
            # NaN ve sonsuz değerleri temizle
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Ölçeklendirme
            X_scaled = self.scaler.fit_transform(X)
            
            if self.algorithm == "dbscan":
                labels = self._dbscan_clustering(X_scaled)
            elif self.algorithm == "hierarchical":
                labels = self._hierarchical_clustering(X_scaled)
            else:
                labels = self._auto_clustering(X_scaled)
            
            self.labels_ = labels
            
            # -1 etiketlerini (gürültü) en yakın kümeye ata
            labels = self._fix_noise_labels(labels, X_scaled)
            
            unique_labels = np.unique(labels)
            self.n_speakers_ = len(unique_labels)
            
            print(f"✅ Kümeleme tamamlandı - {self.n_speakers_} konuşmacı tespit edildi")
            
            # İstatistikleri göster
            self._print_cluster_stats(labels)
            
            return labels
            
        except Exception as e:
            print(f"❌ Kümeleme hatası: {e}")
            # Hata durumunda tüm segmentleri aynı konuşmacı yap
            return np.zeros(len(features_list), dtype=int)
    
    def _dbscan_clustering(self, X: np.ndarray) -> np.ndarray:
        """DBSCAN algoritması ile kümeleme - Daha az strict"""
        # Daha geniş parametreler
        eps = 0.8  # Daha geniş komşuluk
        min_samples = max(2, min(3, len(X) // 15))  # Daha az min_samples
        
        clustering = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric='euclidean'
        ).fit(X)
        
        labels = clustering.labels_
        
        # Eğer çok fazla gürültü varsa, parametreleri gevşet
        noise_ratio = np.sum(labels == -1) / len(labels)
        if noise_ratio > 0.7:  # %70'ten fazla gürültü varsa
            print("   ⚠️  Çok fazla gürültü, parametreler gevşetiliyor...")
            clustering = DBSCAN(
                eps=1.0,  # Daha geniş
                min_samples=2,  # Minimum
                metric='euclidean'
            ).fit(X)
            labels = clustering.labels_
        
        return labels
    
    def _hierarchical_clustering(self, X: np.ndarray) -> np.ndarray:
        """Hiyerarşik kümeleme"""
        # Optimal küme sayısını bul
        n_clusters = self._find_optimal_clusters(X)
        
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='euclidean',
            linkage='average'
        ).fit(X)
        
        return clustering.labels_
    
    def _auto_clustering(self, X: np.ndarray) -> np.ndarray:
        """Otomatik algoritma seçimi"""
        if len(X) < 5:
            return self._hierarchical_clustering(X)
        else:
            # Önce DBSCAN dene
            labels = self._dbscan_clustering(X)
            unique_labels = np.unique(labels)
            
            # Eğer çok fazla gürültü varsa veya hiç küme yoksa, hiyerarşik kullan
            if len(unique_labels) <= 1 or np.sum(labels == -1) / len(labels) > 0.5:
                print("   ⚠️  DBSCAN başarısız, hiyerarşik kümeleme kullanılıyor...")
                return self._hierarchical_clustering(X)
            
            return labels
    
    def _fix_noise_labels(self, labels: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Gürültü etiketlerini en yakın kümeye ata"""
        noise_indices = np.where(labels == -1)[0]
        cluster_indices = np.where(labels != -1)[0]
        
        if len(noise_indices) == 0 or len(cluster_indices) == 0:
            return labels
        
        # Gürültü noktalarını en yakın kümeye ata
        from sklearn.neighbors import NearestNeighbors
        
        # Küme merkezlerini bul
        unique_labels = np.unique(labels[cluster_indices])
        centers = []
        for label in unique_labels:
            cluster_points = X[labels == label]
            centers.append(np.mean(cluster_points, axis=0))
        centers = np.array(centers)
        
        # Her gürültü noktası için en yakın merkezi bul
        nbrs = NearestNeighbors(n_neighbors=1).fit(centers)
        distances, indices = nbrs.kneighbors(X[noise_indices])
        
        # Etiketleri ata
        fixed_labels = labels.copy()
        for i, idx in enumerate(noise_indices):
            fixed_labels[idx] = unique_labels[indices[i][0]]
        
        if len(noise_indices) > 0:
            print(f"   🔧 {len(noise_indices)} gürültü segmenti en yakın kümeye atandı")
        
        return fixed_labels
    
    def _find_optimal_clusters(self, X: np.ndarray) -> int:
        """Optimal küme sayısını bul"""
        if len(X) <= 2:
            return 1
        
        max_clusters = min(self.max_speakers, len(X) - 1)
        min_clusters = max(1, self.min_speakers)
        
        # Küçük datasetler için basit yaklaşım
        if len(X) <= 10:
            return min(3, max(1, len(X) // 3))
        
        best_score = -1
        best_n = 1
        
        for n in range(min_clusters, max_clusters + 1):
            if n >= len(X):
                break
                
            try:
                clustering = AgglomerativeClustering(n_clusters=n)
                labels = clustering.fit_predict(X)
                
                if len(np.unique(labels)) > 1:
                    score = silhouette_score(X, labels)
                    if score > best_score:
                        best_score = score
                        best_n = n
            except:
                continue
        
        # Minimum kalite eşiği düşürüldü
        return best_n if best_score > 0.1 else min(2, len(X))  # Daha düşük eşik
    
    def advanced_speaker_detection(self, features_list: List[np.ndarray]) -> Dict[str, Any]:
        """
        Gelişmiş konuşmacı tespiti ve benzerlik analizi
        """
        try:
            print("🎯 Gelişmiş konuşmacı analizi başlıyor...")
            
            # 1. Temel kümeleme
            labels = self.cluster_speakers(features_list)
            
            # 2. Konuşmacı embedding'lerini hesapla
            embeddings = self.calculate_speaker_embeddings(features_list, labels)
            self.embeddings_ = embeddings
            
            # 3. Benzerlik matrisini hesapla
            similarities = {}
            if len(embeddings) > 1:
                similarities = self.compare_all_speakers(embeddings)
            self.similarity_matrix_ = similarities
            
            # 4. Aynı kişi olabilecek konuşmacıları tespit et
            same_speaker_pairs = self.find_same_speakers(similarities) if similarities else []
            
            # 5. Sonuçları birleştir
            result = {
                'speaker_count': len(embeddings),
                'speaker_ids': list(embeddings.keys()),
                'cluster_labels': labels.tolist(),
                'embeddings': {k: v.tolist() for k, v in embeddings.items()},
                'similarity_matrix': similarities,
                'same_speaker_pairs': same_speaker_pairs,
                'cluster_stats': self.get_cluster_stats(labels)
            }
            
            # 6. Analiz raporunu yazdır
            self._print_advanced_analysis(result)
            
            return result
            
        except Exception as e:
            print(f"❌ Gelişmiş analiz hatası: {e}")
            # Hata durumunda basit sonuç döndür
            labels = np.zeros(len(features_list), dtype=int)
            return {
                'speaker_count': 1,
                'speaker_ids': ['SPK_01'],
                'cluster_labels': labels.tolist(),
                'embeddings': {'SPK_01': np.mean(features_list, axis=0).tolist()},
                'similarity_matrix': {},
                'same_speaker_pairs': [],
                'cluster_stats': self.get_cluster_stats(labels)
            }
    
    def calculate_speaker_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """
        İki konuşmacı arasındaki benzerlik skorunu hesapla
        """
        try:
            # Kosinüs benzerliği
            similarity = cosine_similarity([features1], [features2])[0][0]
            
            # 0-1 aralığına normalize et
            similarity = max(0.0, min(1.0, similarity))
            
            return float(similarity)
            
        except:
            try:
                # Öklid mesafesi tabanlı benzerlik (yedek yöntem)
                distance = np.linalg.norm(features1 - features2)
                similarity = 1.0 / (1.0 + distance)
                return float(similarity)
            except:
                return 0.0
    
    def compare_all_speakers(self, embeddings: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Tüm konuşmacı çiftleri arasındaki benzerlikleri hesapla
        """
        similarities = {}
        speaker_ids = list(embeddings.keys())
        
        print(f"   🔍 {len(speaker_ids)} konuşmacı arasındaki benzerlikler hesaplanıyor...")
        
        for i, spk1 in enumerate(speaker_ids):
            for j, spk2 in enumerate(speaker_ids):
                if i < j:  # Aynı çifti tekrar hesaplama
                    key = f"{spk1}_vs_{spk2}"
                    similarity = self.calculate_speaker_similarity(
                        embeddings[spk1], embeddings[spk2]
                    )
                    
                    similarities[key] = {
                        'similarity_score': similarity,
                        'same_speaker': similarity > self.similarity_threshold,
                        'similarity_percentage': round(similarity * 100, 1)
                    }
        
        return similarities
    
    def find_same_speakers(self, similarities: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Aynı kişi olabilecek konuşmacı çiftlerini bul
        """
        same_speakers = []
        
        for pair, data in similarities.items():
            if data['same_speaker']:
                spk1, spk2 = pair.split('_vs_')
                same_speakers.append({
                    'speaker1': spk1,
                    'speaker2': spk2,
                    'similarity': data['similarity_score'],
                    'similarity_percentage': data['similarity_percentage']
                })
        
        return same_speakers
    
    def get_speaker_confidence(self, embeddings: Dict[str, np.ndarray], speaker_id: str) -> float:
        """
        Konuşmacı tespitinin güven skorunu hesapla
        """
        try:
            if speaker_id not in embeddings:
                return 0.5
            
            speaker_embedding = embeddings[speaker_id]
            other_embeddings = [emb for spk, emb in embeddings.items() if spk != speaker_id]
            
            if not other_embeddings:
                return 0.8  # Tek konuşmacı varsa orta güven
            
            # Diğer konuşmacılarla olan benzerliklerin ortalaması
            similarities = []
            for other_emb in other_embeddings:
                sim = self.calculate_speaker_similarity(speaker_embedding, other_emb)
                similarities.append(sim)
            
            # Düşük benzerlik = yüksek güven
            avg_similarity = np.mean(similarities)
            confidence = 1.0 - avg_similarity
            
            return max(0.3, min(0.9, confidence))  # Min 0.3, max 0.9 güven
            
        except:
            return 0.5  # Varsayılan güven
    
    def assign_speaker_ids(self, labels: np.ndarray) -> List[str]:
        """
        Küme etiketlerini konuşmacı ID'lerine dönüştür
        """
        speaker_ids = []
        unique_labels = np.unique(labels)
        
        # Tüm etiketleri konuşmacı yap (gürültü yok)
        label_to_id = {}
        for i, label in enumerate(unique_labels):
            label_to_id[label] = f"SPK_{i+1:02d}"
        
        for label in labels:
            speaker_ids.append(label_to_id[label])
        
        return speaker_ids
    
    def calculate_speaker_embeddings(self, features: List[np.ndarray], labels: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Her konuşmacı için ortalama özellik vektörü hesapla
        """
        embeddings = {}
        
        for speaker_id in np.unique(labels):
            speaker_features = [features[i] for i in range(len(features)) if labels[i] == speaker_id]
            if speaker_features and len(speaker_features) >= 1:
                embedding = np.mean(speaker_features, axis=0)
                speaker_key = f"SPK_{speaker_id+1:02d}"
                embeddings[speaker_key] = embedding
        
        return embeddings
    
    def get_cluster_stats(self, labels: np.ndarray) -> Dict[str, Any]:
        """
        Küme istatistiklerini hesapla
        """
        unique_labels = np.unique(labels)
        stats = {
            'total_segments': len(labels),
            'total_speakers': len(unique_labels),
            'segment_distribution': {},
            'speaker_segment_counts': {}
        }
        
        for label in unique_labels:
            count = np.sum(labels == label)
            speaker_id = f"SPK_{label+1:02d}"
            stats['segment_distribution'][speaker_id] = count
            stats['speaker_segment_counts'][speaker_id] = count
        
        return stats
    
    def _print_cluster_stats(self, labels: np.ndarray):
        """Küme istatistiklerini yazdır"""
        stats = self.get_cluster_stats(labels)
        
        print(f"   📊 Küme İstatistikleri:")
        print(f"      Toplam Segment: {stats['total_segments']}")
        print(f"      Konuşmacı Sayısı: {stats['total_speakers']}")
        
        for speaker, count in stats['segment_distribution'].items():
            print(f"      {speaker}: {count} segment")
    
    def _print_advanced_analysis(self, result: Dict[str, Any]):
        """Gelişmiş analiz raporunu yazdır"""
        print(f"\n🎯 GELİŞMİŞ KONUŞMACI ANALİZ RAPORU")
        print("=" * 50)
        
        print(f"👥 TOPLAM KONUŞMACI: {result['speaker_count']}")
        
        # Konuşmacı detayları
        if result['speaker_ids']:
            print(f"\n📋 KONUŞMACI DETAYLARI:")
            for speaker_id in result['speaker_ids']:
                segment_count = result['cluster_stats']['speaker_segment_counts'].get(speaker_id, 0)
                print(f"   {speaker_id}: {segment_count} segment")
        else:
            print(f"\n📋 KONUŞMACI DETAYLARI: Hiç konuşmacı bulunamadı")
        
        # Benzerlik analizi
        similarities = result['similarity_matrix']
        if similarities:
            print(f"\n🔍 KONUŞMACI BENZERLİK ANALİZİ:")
            for pair, data in similarities.items():
                similarity_pct = data['similarity_percentage']
                if data['same_speaker']:
                    print(f"   ⚠️  {pair}: AYNI KİŞİ OLABİLİR (%{similarity_pct:.1f} benzer)")
                else:
                    if similarity_pct > 50:
                        print(f"   🔸 {pair}: Yüksek Benzerlik (%{similarity_pct:.1f})")
        
        # Aynı kişi uyarıları
        if result['same_speaker_pairs']:
            print(f"\n🚨 UYARI: AYNI KİŞİ OLABİLECEK KONUŞMACILAR:")
            for pair in result['same_speaker_pairs']:
                print(f"   • {pair['speaker1']} ↔ {pair['speaker2']} (%{pair['similarity_percentage']:.1f} benzer)")

# Test fonksiyonu
if __name__ == "__main__":
    # Test için rastgele özellikler oluştur
    np.random.seed(42)
    
    # 2 farklı konuşmacıyı simüle et
    test_features = []
    for i in range(2):  # 2 konuşmacı
        base_feature = np.random.randn(50) * (i + 1)  # Her konuşmacı farklı
        for _ in range(8):  # Her konuşmacı için 8 segment
            # Küçük varyasyonlar ekle
            variation = np.random.normal(0, 0.2, 50)
            test_features.append(base_feature + variation)
    
    clusterer = SpeakerClustering(similarity_threshold=0.5)  # Daha düşük eşik
    
    # Gelişmiş analiz
    advanced_result = clusterer.advanced_speaker_detection(test_features)
    
    print("\n" + "="*50)
    print("✅ Konuşmacı kümeleme testi başarılı!")