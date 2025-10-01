import unittest
import numpy as np
import sys
import os

# src klasörünü path'e ekle
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from feature_extractor import FeatureExtractor

class TestFeatureExtractor(unittest.TestCase):
    
    def setUp(self):
        self.extractor = FeatureExtractor()
        self.test_audio = np.random.randn(16000)  # 1 saniyelik test sesi
    
    def test_mfcc_extraction(self):
        mfcc = self.extractor.extract_mfcc(self.test_audio)
        self.assertEqual(len(mfcc), 26)  # 13 mean + 13 std
    
    def test_pitch_extraction(self):
        pitch_features = self.extractor.extract_pitch(self.test_audio)
        self.assertIn('pitch_mean', pitch_features)
        self.assertIsInstance(pitch_features['pitch_mean'], float)
    
    def test_complete_feature_set(self):
        features = self.extractor.get_complete_feature_set(self.test_audio)
        self.assertIn('feature_vector', features)
        self.assertTrue(len(features['feature_vector']) > 0)

if __name__ == '__main__':
    unittest.main()