"""
信号自适应预处理模块单元测试

测试状态分类器、FFT通道、Laplace通道、Z通道
"""

import unittest
import numpy as np
from enlighten.memory.signal_preprocessor import (
    SignalAdaptivePreprocessor,
    SignalPreprocessorConfig,
    StateClassifier,
    FFTPreprocessor,
    LaplacePreprocessor,
    ZTransformPreprocessor,
    SignalWindow,
    SignalState,
)


class TestStateClassifier(unittest.TestCase):
    def setUp(self):
        self.config = SignalPreprocessorConfig(
            window_size=32,
            discrete_threshold=2,
            variance_threshold=0.1,
            confidence_threshold=0.8,
        )
        self.classifier = StateClassifier(self.config)

    def test_discrete_state(self):
        window = SignalWindow(
            entropy=np.array([0.5, 0.6, 0.7, 0.8]),
            confidence=np.array([0.9, 0.9, 0.9, 0.9]),
            interventions=np.array([0, 1, 2, 1]),
        )
        state = self.classifier.classify(window)
        self.assertEqual(state, SignalState.DISCRETE)

    def test_diverging_state(self):
        window = SignalWindow(
            entropy=np.array([0.01, 0.4, 0.6, 0.8, 0.99]),
            confidence=np.array([0.9, 0.85, 0.8, 0.75, 0.7]),
            interventions=np.array([0, 0, 0, 0, 0]),
        )
        state = self.classifier.classify(window)
        self.assertEqual(state, SignalState.DIVERGING)

    def test_converging_state(self):
        window = SignalWindow(
            entropy=np.array([0.8, 0.7, 0.6, 0.5, 0.4]),
            confidence=np.array([0.9, 0.9, 0.9, 0.9, 0.9]),
            interventions=np.array([0, 0, 0, 0, 0]),
        )
        state = self.classifier.classify(window)
        self.assertEqual(state, SignalState.CONVERGING)

    def test_default_converging(self):
        window = SignalWindow(
            entropy=np.array([0.5, 0.5, 0.5, 0.5]),
            confidence=np.array([0.5, 0.5, 0.5, 0.5]),
            interventions=np.array([0, 0, 0, 0]),
        )
        state = self.classifier.classify(window)
        self.assertEqual(state, SignalState.CONVERGING)


class TestFFTPreprocessor(unittest.TestCase):
    def setUp(self):
        self.config = SignalPreprocessorConfig(fft_high_freq_ratio_threshold=0.3)
        self.processor = FFTPreprocessor(self.config)

    def test_periodic_signal(self):
        t = np.linspace(0, 2*np.pi, 32)
        entropy_window = np.sin(t) + 0.1 * np.random.randn(32)

        features = self.processor.transform(entropy_window)

        self.assertIn('high_freq_ratio', features)
        self.assertIn('dominant_frequency', features)
        self.assertIn('is_periodic_hallucination', features)
        self.assertIsInstance(features['high_freq_ratio'], float)

    def test_default_features(self):
        entropy_window = np.array([0.5])
        features = self.processor.transform(entropy_window)

        self.assertEqual(features['high_freq_ratio'], 0.0)
        self.assertEqual(features['dominant_frequency'], 0.0)
        self.assertEqual(features['is_periodic_hallucination'], 0.0)

    def test_converging_signal(self):
        entropy_window = np.array([0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2])

        features = self.processor.transform(entropy_window)

        self.assertIsInstance(features['high_freq_ratio'], float)
        self.assertGreaterEqual(features['high_freq_ratio'], 0.0)


class TestLaplacePreprocessor(unittest.TestCase):
    def setUp(self):
        self.config = SignalPreprocessorConfig(laplace_tail_threshold=0.9)
        self.processor = LaplacePreprocessor(self.config)

    def test_high_uncertainty(self):
        signal_window = np.array([0.5, 0.5, 0.5, 0.5, 0.9])

        features = self.processor.transform(signal_window)

        self.assertIn('mu', features)
        self.assertIn('b', features)
        self.assertIn('tail_probability', features)
        self.assertIn('is_high_uncertainty', features)
        self.assertIsInstance(features['b'], float)
        self.assertGreater(features['b'], 0)

    def test_default_features(self):
        signal_window = np.array([0.5])
        features = self.processor.transform(signal_window)

        self.assertEqual(features['mu'], 0.0)
        self.assertEqual(features['b'], 0.0)
        self.assertEqual(features['tail_probability'], 0.5)

    def test_stable_signal(self):
        signal_window = np.array([0.5, 0.5, 0.5, 0.5, 0.5])

        features = self.processor.transform(signal_window)

        self.assertIsInstance(features['b'], float)
        self.assertLess(features['b'], 0.1)


class TestZTransformPreprocessor(unittest.TestCase):
    def setUp(self):
        self.config = SignalPreprocessorConfig(z_pole_threshold=0.95)
        self.processor = ZTransformPreprocessor(self.config)

    def test_marginally_effective(self):
        event_sequence = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2])

        features = self.processor.transform(event_sequence)

        self.assertIn('intervention_entropy', features)
        self.assertIn('halt_rate', features)
        self.assertIn('rewind_rate', features)
        self.assertIn('dominant_pole_magnitude', features)
        self.assertIn('is_marginally_effective', features)

    def test_default_features(self):
        event_sequence = np.array([0, 1])
        features = self.processor.transform(event_sequence)

        self.assertEqual(features['intervention_entropy'], 0.0)
        self.assertEqual(features['dominant_pole_magnitude'], 0.5)

    def test_no_intervention(self):
        event_sequence = np.array([0, 0, 0, 0, 0])

        features = self.processor.transform(event_sequence)

        self.assertEqual(features['halt_rate'], 0.0)
        self.assertEqual(features['rewind_rate'], 0.0)
        self.assertEqual(features['intervention_rate'], 0.0)


class TestSignalAdaptivePreprocessor(unittest.TestCase):
    def setUp(self):
        self.config = {
            'window_size': 32,
            'discrete_threshold': 2,
            'variance_threshold': 0.1,
            'confidence_threshold': 0.8,
            'fft_high_freq_ratio_threshold': 0.3,
            'laplace_tail_threshold': 0.9,
            'z_pole_threshold': 0.95,
        }
        self.processor = SignalAdaptivePreprocessor(self.config)

    def test_preprocess_converging(self):
        window = SignalWindow(
            entropy=np.array([0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2]),
            confidence=np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]),
            interventions=np.array([0, 0, 0, 0, 0, 0, 0, 0]),
        )

        features = self.processor.preprocess(window)

        self.assertEqual(features.state, SignalState.CONVERGING)
        self.assertIn('fft_features', features.__dict__)
        self.assertIn('laplace_features', features.__dict__)
        self.assertIn('z_features', features.__dict__)
        self.assertIn('raw_features', features.__dict__)

    def test_preprocess_diverging(self):
        window = SignalWindow(
            entropy=np.array([0.1, 0.25, 0.5, 0.75, 0.9, 0.92, 0.95, 0.98]),
            confidence=np.array([0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6]),
            interventions=np.array([0, 0, 0, 0, 0, 0, 0, 0]),
        )

        features = self.processor.preprocess(window)

        self.assertEqual(features.state, SignalState.DIVERGING)

    def test_preprocess_discrete(self):
        window = SignalWindow(
            entropy=np.array([0.5, 0.6, 0.7, 0.8]),
            confidence=np.array([0.9, 0.9, 0.9, 0.9]),
            interventions=np.array([0, 1, 2, 1]),
        )

        features = self.processor.preprocess(window)

        self.assertEqual(features.state, SignalState.DISCRETE)

    def test_get_active_features_converging(self):
        window = SignalWindow(
            entropy=np.array([0.8, 0.7, 0.6, 0.5, 0.4]),
            confidence=np.array([0.9, 0.9, 0.9, 0.9, 0.9]),
            interventions=np.array([0, 0, 0, 0, 0]),
        )

        features = self.processor.preprocess(window)
        active_features = self.processor.get_active_features(features)

        self.assertEqual(active_features['state_indicator'], 0.0)
        self.assertIn('high_freq_ratio', active_features)

    def test_get_active_features_diverging(self):
        window = SignalWindow(
            entropy=np.array([0.01, 0.4, 0.6, 0.8, 0.99]),
            confidence=np.array([0.9, 0.85, 0.8, 0.75, 0.7]),
            interventions=np.array([0, 0, 0, 0, 0]),
        )

        features = self.processor.preprocess(window)
        active_features = self.processor.get_active_features(features)

        self.assertEqual(active_features['state_indicator'], 1.0)
        self.assertIn('mu', active_features)

    def test_get_active_features_discrete(self):
        window = SignalWindow(
            entropy=np.array([0.5, 0.6, 0.7, 0.8]),
            confidence=np.array([0.9, 0.9, 0.9, 0.9]),
            interventions=np.array([0, 1, 2, 1]),
        )

        features = self.processor.preprocess(window)
        active_features = self.processor.get_active_features(features)

        self.assertEqual(active_features['state_indicator'], 2.0)
        self.assertIn('intervention_entropy', active_features)


class TestSignalWindow(unittest.TestCase):
    def test_signal_window_creation(self):
        window = SignalWindow(
            entropy=np.array([0.5, 0.6, 0.7]),
            confidence=np.array([0.9, 0.8, 0.7]),
            interventions=np.array([0, 0, 0]),
        )

        self.assertEqual(len(window.entropy), 3)
        self.assertEqual(len(window.confidence), 3)
        self.assertEqual(len(window.interventions), 3)
        self.assertIsNotNone(window.delta_entropy)

    def test_signal_window_delta_calculation(self):
        window = SignalWindow(
            entropy=np.array([0.5, 0.6, 0.7]),
            confidence=np.array([0.9, 0.8, 0.7]),
            interventions=np.array([0, 0, 0]),
        )

        expected_delta = np.array([0.1, 0.1])
        np.testing.assert_array_almost_equal(window.delta_entropy, expected_delta)


if __name__ == '__main__':
    unittest.main()
