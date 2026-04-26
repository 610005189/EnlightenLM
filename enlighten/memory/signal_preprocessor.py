"""
信号自适应预处理模块

根据架构设计文档 v3.0 实现：
- 对原始信号（熵、置信度、重复率）进行状态分类
- 根据状态选择合适的数学变换
- 输出结构化特征供 L3 使用

核心思路：收敛就用傅里叶，发散就用拉普拉斯，离散就用 Z
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any
import numpy as np


class SignalState(Enum):
    CONVERGING = "converging"
    DIVERGING = "diverging"
    DISCRETE = "discrete"


@dataclass
class SignalWindow:
    entropy: np.ndarray
    confidence: np.ndarray
    interventions: np.ndarray
    delta_entropy: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.delta_entropy is None:
            if len(self.entropy) > 1:
                self.delta_entropy = np.diff(self.entropy)
            else:
                self.delta_entropy = np.array([0.0])


@dataclass
class StructuredFeatures:
    state: SignalState
    fft_features: Dict[str, float]
    laplace_features: Dict[str, float]
    z_features: Dict[str, float]
    raw_features: Dict[str, float]


class SignalPreprocessorConfig:
    def __init__(
        self,
        window_size: int = 32,
        discrete_threshold: int = 2,
        variance_threshold: float = 0.1,
        confidence_threshold: float = 0.8,
        fft_high_freq_ratio_threshold: float = 0.3,
        laplace_tail_threshold: float = 0.9,
        z_pole_threshold: float = 0.95,
    ):
        self.window_size = window_size
        self.discrete_threshold = discrete_threshold
        self.variance_threshold = variance_threshold
        self.confidence_threshold = confidence_threshold
        self.fft_high_freq_ratio_threshold = fft_high_freq_ratio_threshold
        self.laplace_tail_threshold = laplace_tail_threshold
        self.z_pole_threshold = z_pole_threshold


class StateClassifier:
    def __init__(self, config: SignalPreprocessorConfig):
        self.config = config

    def classify(self, window: SignalWindow) -> SignalState:
        if self._is_discrete(window):
            return SignalState.DISCRETE
        if self._is_diverging(window):
            return SignalState.DIVERGING
        if self._is_converging(window):
            return SignalState.CONVERGING
        return SignalState.CONVERGING

    def _is_discrete(self, window: SignalWindow) -> bool:
        halt_rewind_count = np.sum(window.interventions >= 1)
        return halt_rewind_count >= self.config.discrete_threshold

    def _is_diverging(self, window: SignalWindow) -> bool:
        if len(window.entropy) < 3:
            return False

        entropy_variance = np.var(window.entropy)
        recent_deltas = window.delta_entropy[-3:] if len(window.delta_entropy) >= 3 else window.delta_entropy

        if entropy_variance > self.config.variance_threshold and np.mean(recent_deltas) > 0:
            return True
        return False

    def _is_converging(self, window: SignalWindow) -> bool:
        if len(window.entropy) < 5:
            return False

        entropy_monotonic = np.all(np.diff(window.entropy[-5:]) <= 0)

        recent_confidence = window.confidence[-5:] if len(window.confidence) >= 5 else window.confidence
        confidence_stable = np.mean(recent_confidence) > self.config.confidence_threshold

        return entropy_monotonic and confidence_stable


class FFTPreprocessor:
    def __init__(self, config: SignalPreprocessorConfig):
        self.config = config

    def transform(self, entropy_window: np.ndarray) -> Dict[str, float]:
        if len(entropy_window) < 4:
            return self._default_features()

        entropy_centered = entropy_window - np.mean(entropy_window)

        fft_result = np.fft.fft(entropy_centered)
        fft_freq = np.fft.fftfreq(len(entropy_window))

        power_spectrum = np.abs(fft_result) ** 2

        n = len(fft_result)
        k = min(3, n // 2)

        low_freq_energy = np.sum(power_spectrum[1:k+1])
        high_freq_energy = np.sum(power_spectrum[k+1:n//2]) if n > k+1 else 0
        total_energy = np.sum(power_spectrum[1:n//2]) if n > 1 else 1e-10

        high_freq_ratio = high_freq_energy / (total_energy + 1e-10)

        dominant_freq_idx = np.argmax(power_spectrum[1:n//2]) + 1 if n > 1 else 0
        dominant_freq = fft_freq[dominant_freq_idx] if dominant_freq_idx < len(fft_freq) else 0

        phase = np.angle(fft_result[dominant_freq_idx]) if dominant_freq_idx < len(fft_result) else 0

        return {
            "high_freq_ratio": float(high_freq_ratio),
            "dominant_frequency": float(np.abs(dominant_freq)),
            "dominant_phase": float(phase),
            "low_freq_energy": float(low_freq_energy),
            "high_freq_energy": float(high_freq_energy),
            "is_periodic_hallucination": float(high_freq_ratio > self.config.fft_high_freq_ratio_threshold),
        }

    def _default_features(self) -> Dict[str, float]:
        return {
            "high_freq_ratio": 0.0,
            "dominant_frequency": 0.0,
            "dominant_phase": 0.0,
            "low_freq_energy": 0.0,
            "high_freq_energy": 0.0,
            "is_periodic_hallucination": 0.0,
        }


class LaplacePreprocessor:
    def __init__(self, config: SignalPreprocessorConfig):
        self.config = config

    def transform(self, signal_window: np.ndarray) -> Dict[str, float]:
        if len(signal_window) < 2:
            return self._default_features()

        mu = np.median(signal_window)

        abs_deviations = np.abs(signal_window - mu)
        b = np.mean(abs_deviations)

        if b < 1e-10:
            b = 1e-10

        current_value = signal_window[-1]

        z_score = (current_value - mu) / b

        tail_prob = self._laplace_cdf(z_score)

        b_increase_ratio = b / (np.mean(signal_window[:-1]) + 1e-10) if len(signal_window) > 1 else 0

        return {
            "mu": float(mu),
            "b": float(b),
            "current_z_score": float(z_score),
            "tail_probability": float(tail_prob),
            "b_increase_ratio": float(b_increase_ratio),
            "is_high_uncertainty": float(tail_prob > self.config.laplace_tail_threshold),
        }

    def _laplace_cdf(self, z: float) -> float:
        return 0.5 * (1 + np.sign(z) * (1 - np.exp(-np.abs(z))))

    def _default_features(self) -> Dict[str, float]:
        return {
            "mu": 0.0,
            "b": 0.0,
            "current_z_score": 0.0,
            "tail_probability": 0.5,
            "b_increase_ratio": 0.0,
            "is_high_uncertainty": 0.0,
        }


class ZTransformPreprocessor:
    def __init__(self, config: SignalPreprocessorConfig):
        self.config = config

    def transform(self, event_sequence: np.ndarray) -> Dict[str, float]:
        if len(event_sequence) < 3:
            return self._default_features()

        unique_events = np.unique(event_sequence)
        event_counts = np.array([np.sum(event_sequence == e) for e in unique_events])
        total_events = len(event_sequence)

        event_probs = event_counts / total_events

        entropy = -np.sum(event_probs * np.log(event_probs + 1e-10))

        halt_count = np.sum(event_sequence >= 1)
        rewind_count = np.sum(event_sequence >= 2)
        intervention_rate = halt_count / total_events

        z_transform_result = self._compute_z_transform_pole(event_sequence)

        return {
            "intervention_entropy": float(entropy),
            "halt_rate": float(halt_count / total_events),
            "rewind_rate": float(rewind_count / total_events),
            "intervention_rate": float(intervention_rate),
            "dominant_pole_magnitude": float(z_transform_result),
            "is_marginally_effective": float(z_transform_result > self.config.z_pole_threshold),
        }

    def _compute_z_transform_pole(self, event_sequence: np.ndarray) -> float:
        if len(event_sequence) < 3:
            return 0.5

        event_normalized = event_sequence / (np.max(np.abs(event_sequence)) + 1e-10)

        polynomial = np.poly1d(event_normalized[::-1])

        try:
            roots = polynomial.roots
            if len(roots) == 0:
                return 0.5

            magnitudes = np.abs(roots)
            dominant_pole = np.max(magnitudes)

            return float(np.clip(dominant_pole, 0, 2))
        except Exception:
            return 0.5

    def _default_features(self) -> Dict[str, float]:
        return {
            "intervention_entropy": 0.0,
            "halt_rate": 0.0,
            "rewind_rate": 0.0,
            "intervention_rate": 0.0,
            "dominant_pole_magnitude": 0.5,
            "is_marginally_effective": 0.0,
        }


class SignalAdaptivePreprocessor:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = {}
        self.config = SignalPreprocessorConfig(
            window_size=config.get('window_size', 32),
            discrete_threshold=config.get('discrete_threshold', 2),
            variance_threshold=config.get('variance_threshold', 0.1),
            confidence_threshold=config.get('confidence_threshold', 0.8),
            fft_high_freq_ratio_threshold=config.get('fft_high_freq_ratio_threshold', 0.3),
            laplace_tail_threshold=config.get('laplace_tail_threshold', 0.9),
            z_pole_threshold=config.get('z_pole_threshold', 0.95),
        )
        self.state_classifier = StateClassifier(self.config)
        self.fft_processor = FFTPreprocessor(self.config)
        self.laplace_processor = LaplacePreprocessor(self.config)
        self.z_processor = ZTransformPreprocessor(self.config)

    def preprocess(self, window: SignalWindow) -> StructuredFeatures:
        state = self.state_classifier.classify(window)

        fft_features = self.fft_processor.transform(window.entropy)
        laplace_features = self.laplace_processor.transform(window.entropy)
        z_features = self.z_processor.transform(window.interventions)

        raw_features = {
            "entropy_mean": float(np.mean(window.entropy)),
            "entropy_variance": float(np.var(window.entropy)),
            "entropy_trend": float(np.mean(window.delta_entropy)) if len(window.delta_entropy) > 0 else 0.0,
            "confidence_mean": float(np.mean(window.confidence)),
            "confidence_min": float(np.min(window.confidence)),
            "intervention_count": int(np.sum(window.interventions >= 1)),
        }

        return StructuredFeatures(
            state=state,
            fft_features=fft_features,
            laplace_features=laplace_features,
            z_features=z_features,
            raw_features=raw_features,
        )

    def get_active_features(self, features: StructuredFeatures) -> Dict[str, float]:
        if features.state == SignalState.CONVERGING:
            return {
                **features.fft_features,
                "state_indicator": 0.0,
            }
        elif features.state == SignalState.DIVERGING:
            return {
                **features.laplace_features,
                "state_indicator": 1.0,
            }
        else:
            return {
                **features.z_features,
                "state_indicator": 2.0,
            }
