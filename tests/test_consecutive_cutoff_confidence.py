"""
测试连续截断信心机制

测试 ConsecutiveCutoffConfidence 类的核心功能：
1. 信心初始化和基本计算
2. 连续截断信心衰减机制
3. 动态阈值调整
4. 防止过度截断机制
5. 与 EnhancedBayesianL3Controller 的集成
"""

import pytest
import numpy as np
from typing import Dict, List

from enlighten.l3_controller import (
    ConsecutiveCutoffConfidence,
    EnhancedBayesianL3Controller
)


class TestConsecutiveCutoffConfidence:
    """测试 ConsecutiveCutoffConfidence 类"""

    def test_initialization(self):
        """测试初始化"""
        confidence = ConsecutiveCutoffConfidence()
        assert confidence is not None
        assert confidence.current_confidence == 0.5
        assert confidence.consecutive_cutoff_count == 0
        assert len(confidence.cutoff_history) == 0

    def test_initialization_with_custom_params(self):
        """测试带自定义参数的初始化"""
        confidence = ConsecutiveCutoffConfidence(
            initial_confidence=0.6,
            decay_factor=0.8,
            confidence_boost_per_cutoff=0.2,
            max_consecutive_cutoffs=3,
            over_cutoff_threshold=0.7,
            confidence_threshold_increase=0.15,
            min_confidence_threshold=0.4,
            max_confidence_threshold=0.85
        )
        assert confidence.initial_confidence == 0.6
        assert confidence.decay_factor == 0.8
        assert confidence.confidence_boost_per_cutoff == 0.2
        assert confidence.max_consecutive_cutoffs == 3
        assert confidence.min_confidence_threshold == 0.4
        assert confidence.max_confidence_threshold == 0.85

    def test_first_cutoff_boost(self):
        """测试首次截断的信心提升"""
        confidence = ConsecutiveCutoffConfidence(
            decay_factor=0.85,
            confidence_boost_per_cutoff=0.15,
            max_consecutive_cutoffs=5
        )

        entropy_stats = {'mean': 0.3, 'variance': 0.02, 'trend': -0.05}
        result = confidence.update(False, entropy_stats, van_event=False, p_harm_raw=0.2)

        assert confidence.consecutive_cutoff_count == 0
        assert result['consecutive_cutoff_count'] == 0

    def test_consecutive_cutoff_accumulation(self):
        """测试连续截断时信心的累积"""
        confidence = ConsecutiveCutoffConfidence(
            decay_factor=0.85,
            confidence_boost_per_cutoff=0.15,
            max_consecutive_cutoffs=5
        )

        entropy_stats = {'mean': 0.2, 'variance': 0.01, 'trend': -0.1}

        result1 = confidence.update(True, entropy_stats, van_event=False, p_harm_raw=0.3)
        assert result1['consecutive_cutoff_count'] == 1

        result2 = confidence.update(True, entropy_stats, van_event=False, p_harm_raw=0.3)
        assert result2['consecutive_cutoff_count'] == 2
        assert result2['confidence'] > result1['confidence']

        result3 = confidence.update(True, entropy_stats, van_event=False, p_harm_raw=0.3)
        assert result3['consecutive_cutoff_count'] == 3
        assert result3['confidence'] > result2['confidence']

    def test_decay_mechanism(self):
        """测试衰减机制：第N次截断的增量小于第N-1次"""
        confidence = ConsecutiveCutoffConfidence(
            decay_factor=0.85,
            confidence_boost_per_cutoff=0.15,
            max_consecutive_cutoffs=10
        )

        entropy_stats = {'mean': 0.2, 'variance': 0.01, 'trend': -0.1}

        prev_confidence = 0.0
        increments = []
        for i in range(5):
            result = confidence.update(True, entropy_stats, van_event=False, p_harm_raw=0.3)
            increment = result['confidence'] - prev_confidence
            increments.append(increment)
            prev_confidence = result['confidence']

        for i in range(1, len(increments)):
            assert increments[i] < increments[i-1], f"Increment {i} should be less than increment {i-1}"

    def test_consecutive_cutoff_reset(self):
        """测试连续截断计数器在非截断时重置"""
        confidence = ConsecutiveCutoffConfidence(
            decay_factor=0.85,
            confidence_boost_per_cutoff=0.15,
            max_consecutive_cutoffs=5
        )

        entropy_stats = {'mean': 0.2, 'variance': 0.01, 'trend': -0.1}

        for _ in range(3):
            confidence.update(True, entropy_stats, van_event=False, p_harm_raw=0.3)

        assert confidence.consecutive_cutoff_count == 3

        confidence.update(False, entropy_stats, van_event=False, p_harm_raw=0.1)

        assert confidence.consecutive_cutoff_count == 0

    def test_raw_confidence_with_van_event(self):
        """测试VAN事件时原始信心计算"""
        confidence = ConsecutiveCutoffConfidence()

        entropy_stats = {'mean': 0.5, 'variance': 0.02, 'trend': 0.0}
        result = confidence.update(False, entropy_stats, van_event=True, p_harm_raw=0.5)

        assert result['raw_confidence'] == 0.8

    def test_raw_confidence_with_low_entropy(self):
        """测试低熵时原始信心计算"""
        confidence = ConsecutiveCutoffConfidence()

        entropy_stats = {'mean': 0.15, 'variance': 0.01, 'trend': -0.1}
        result = confidence.update(False, entropy_stats, van_event=False, p_harm_raw=0.3)

        assert result['raw_confidence'] > 0.3

    def test_dynamic_threshold_increase(self):
        """测试连续截断过长时动态阈值增加"""
        confidence = ConsecutiveCutoffConfidence(
            decay_factor=0.85,
            confidence_boost_per_cutoff=0.15,
            max_consecutive_cutoffs=3,
            confidence_threshold_increase=0.1,
            min_confidence_threshold=0.5,
            max_confidence_threshold=0.9
        )

        entropy_stats = {'mean': 0.2, 'variance': 0.01, 'trend': -0.1}

        for i in range(5):
            result = confidence.update(True, entropy_stats, van_event=False, p_harm_raw=0.3)
            if i >= 2:
                assert result['is_over_cutoff_risk']
                assert result['dynamic_threshold'] > 0.6

    def test_over_cutoff_protection(self):
        """测试过度截断保护机制"""
        confidence = ConsecutiveCutoffConfidence(
            decay_factor=0.85,
            confidence_boost_per_cutoff=0.15,
            max_consecutive_cutoffs=3,
            over_cutoff_threshold=0.75,
            min_confidence_threshold=0.5,
            max_confidence_threshold=0.9
        )

        entropy_stats = {'mean': 0.5, 'variance': 0.02, 'trend': 0.0}

        for _ in range(3):
            confidence.update(True, entropy_stats, van_event=False, p_harm_raw=0.2)

        override, reason = confidence.should_override_cutoff()
        assert override
        assert "Over-cutoff protection" in reason

    def test_low_confidence_trend_protection(self):
        """测试低信心趋势保护"""
        confidence = ConsecutiveCutoffConfidence(
            initial_confidence=0.3,
            decay_factor=0.85,
            confidence_boost_per_cutoff=0.1,
            max_consecutive_cutoffs=5
        )

        entropy_stats = {'mean': 0.6, 'variance': 0.05, 'trend': 0.0}

        for _ in range(5):
            confidence.update(False, entropy_stats, van_event=False, p_harm_raw=0.1)

        override, reason = confidence.should_override_cutoff()
        assert override
        assert "Low confidence trend" in reason

    def test_no_override_without_excessive_cutoffs(self):
        """测试没有过度截断时不否决"""
        confidence = ConsecutiveCutoffConfidence(
            decay_factor=0.85,
            confidence_boost_per_cutoff=0.15,
            max_consecutive_cutoffs=5
        )

        entropy_stats = {'mean': 0.2, 'variance': 0.01, 'trend': -0.1}

        for _ in range(2):
            confidence.update(True, entropy_stats, van_event=False, p_harm_raw=0.3)

        override, reason = confidence.should_override_cutoff()
        assert not override
        assert reason is None

    def test_get_statistics(self):
        """测试获取统计信息"""
        confidence = ConsecutiveCutoffConfidence(
            decay_factor=0.85,
            confidence_boost_per_cutoff=0.15,
            max_consecutive_cutoffs=5
        )

        entropy_stats = {'mean': 0.3, 'variance': 0.02, 'trend': -0.05}

        for i in range(5):
            confidence.update(i % 2 == 0, entropy_stats, van_event=False, p_harm_raw=0.2)

        stats = confidence.get_statistics()

        assert 'current_confidence' in stats
        assert 'consecutive_cutoff_count' in stats
        assert 'total_cutoffs' in stats
        assert 'cutoff_ratio' in stats
        assert 'dynamic_threshold' in stats
        assert 'is_over_cutoff_risk' in stats

    def test_reset(self):
        """测试重置功能"""
        confidence = ConsecutiveCutoffConfidence(
            decay_factor=0.85,
            confidence_boost_per_cutoff=0.15,
            max_consecutive_cutoffs=5
        )

        entropy_stats = {'mean': 0.2, 'variance': 0.01, 'trend': -0.1}

        for _ in range(3):
            confidence.update(True, entropy_stats, van_event=False, p_harm_raw=0.3)

        assert confidence.consecutive_cutoff_count == 3
        assert len(confidence.cutoff_history) == 3

        confidence.reset()

        assert confidence.consecutive_cutoff_count == 0
        assert confidence.current_confidence == 0.5
        assert len(confidence.cutoff_history) == 0

    def test_confidence_bounds(self):
        """测试信心值边界限制"""
        confidence = ConsecutiveCutoffConfidence(
            decay_factor=0.9,
            confidence_boost_per_cutoff=0.5,
            max_consecutive_cutoffs=10
        )

        entropy_stats = {'mean': 0.1, 'variance': 0.01, 'trend': -0.2}

        for _ in range(20):
            result = confidence.update(True, entropy_stats, van_event=False, p_harm_raw=0.1)
            assert 0.0 <= result['confidence'] <= 1.0

    def test_threshold_bounds(self):
        """测试阈值边界限制"""
        confidence = ConsecutiveCutoffConfidence(
            decay_factor=0.85,
            confidence_boost_per_cutoff=0.15,
            max_consecutive_cutoffs=20,
            confidence_threshold_increase=0.5,
            min_confidence_threshold=0.5,
            max_confidence_threshold=0.9
        )

        entropy_stats = {'mean': 0.1, 'variance': 0.01, 'trend': -0.2}

        for _ in range(30):
            result = confidence.update(True, entropy_stats, van_event=False, p_harm_raw=0.2)
            assert 0.5 <= result['dynamic_threshold'] <= 0.9


class TestEnhancedBayesianL3ControllerWithConsecutiveConfidence:
    """测试 EnhancedBayesianL3Controller 与连续截断信心的集成"""

    def test_initialization_with_consecutive_confidence(self):
        """测试启用连续截断信心的初始化"""
        controller = EnhancedBayesianL3Controller(use_consecutive_confidence=True)
        assert controller.use_consecutive_confidence
        assert controller.consecutive_confidence is not None

    def test_initialization_without_consecutive_confidence(self):
        """测试禁用连续截断信心的初始化"""
        controller = EnhancedBayesianL3Controller(use_consecutive_confidence=False)
        assert not controller.use_consecutive_confidence

    def test_forward_with_consecutive_confidence(self):
        """测试启用连续截断信心时的前向传播"""
        controller = EnhancedBayesianL3Controller(use_consecutive_confidence=True)

        entropy_stats = {'mean': 0.3, 'variance': 0.02, 'trend': -0.05}

        for i in range(3):
            result = controller.forward(entropy_stats, van_event=False, p_harm=0.4)
            assert hasattr(result, 'cutoff')
            assert hasattr(result, 'tau')
            assert hasattr(result, 'reason')

    def test_consecutive_cutoff_prevents_excessive_cutoffs(self):
        """测试连续截断信心防止过度截断"""
        controller = EnhancedBayesianL3Controller(
            use_consecutive_confidence=True,
            consecutive_confidence_config={
                'max_consecutive_cutoffs': 3,
                'decay_factor': 0.85,
                'confidence_boost_per_cutoff': 0.15,
                'min_confidence_threshold': 0.5,
                'max_confidence_threshold': 0.9
            }
        )

        entropy_stats = {'mean': 0.3, 'variance': 0.02, 'trend': -0.05}

        cutoff_count = 0
        for i in range(10):
            result = controller.forward(entropy_stats, van_event=False, p_harm=0.4)
            if result.cutoff:
                cutoff_count += 1

        stats = controller.get_statistics()
        confidence_stats = stats.get('consecutive_confidence', {})

        assert confidence_stats.get('consecutive_cutoff_count', 0) <= 3

    def test_reset_clears_consecutive_confidence(self):
        """测试重置时清除连续截断信心状态"""
        controller = EnhancedBayesianL3Controller(use_consecutive_confidence=True)

        entropy_stats = {'mean': 0.2, 'variance': 0.01, 'trend': -0.1}

        for _ in range(3):
            controller.forward(entropy_stats, van_event=False, p_harm=0.4)

        controller.reset()

        stats = controller.get_statistics()
        if 'consecutive_confidence' in stats:
            cc_stats = stats['consecutive_confidence']
            assert cc_stats.get('consecutive_cutoff_count', 0) == 0

    def test_van_event_bypasses_consecutive_confidence(self):
        """测试VAN事件绕过连续截断信心机制"""
        controller = EnhancedBayesianL3Controller(use_consecutive_confidence=True)

        entropy_stats = {'mean': 0.5, 'variance': 0.02, 'trend': 0.0}

        for i in range(3):
            result = controller.forward(entropy_stats, van_event=True, p_harm=0.8)
            assert result.cutoff
            assert "VAN" in result.reason

    def test_statistics_include_consecutive_confidence(self):
        """测试统计信息包含连续截断信心数据"""
        controller = EnhancedBayesianL3Controller(use_consecutive_confidence=True)

        entropy_stats = {'mean': 0.3, 'variance': 0.02, 'trend': -0.05}

        for _ in range(5):
            controller.forward(entropy_stats, van_event=False, p_harm=0.3)

        stats = controller.get_statistics()

        assert 'consecutive_confidence' in stats
        cc_stats = stats['consecutive_confidence']
        assert 'current_confidence' in cc_stats
        assert 'consecutive_cutoff_count' in cc_stats
        assert 'dynamic_threshold' in cc_stats


class TestConsecutiveConfidenceScenarios:
    """测试连续截断信心在不同场景下的表现"""

    def test_normal_operation_scenario(self):
        """测试正常操作场景"""
        confidence = ConsecutiveCutoffConfidence(
            decay_factor=0.85,
            confidence_boost_per_cutoff=0.15,
            max_consecutive_cutoffs=5
        )

        entropy_stats = {'mean': 0.6, 'variance': 0.03, 'trend': 0.01}

        for i in range(10):
            result = confidence.update(False, entropy_stats, van_event=False, p_harm_raw=0.1)
            assert result['consecutive_cutoff_count'] == 0
            assert not result['is_over_cutoff_risk']

    def test_slow_drift_attack_scenario(self):
        """测试缓慢漂移攻击场景"""
        confidence = ConsecutiveCutoffConfidence(
            decay_factor=0.85,
            confidence_boost_per_cutoff=0.15,
            max_consecutive_cutoffs=5
        )

        mu_h = 0.5
        for i in range(10):
            entropy_stats = {
                'mean': mu_h - i * 0.03,
                'variance': 0.02,
                'trend': -0.02
            }
            result = confidence.update(True, entropy_stats, van_event=False, p_harm_raw=0.2 + i * 0.05)

            if i >= 4:
                assert result['consecutive_cutoff_count'] >= 1

    def test_recovery_after_cutoff_scenario(self):
        """测试截断后恢复场景"""
        confidence = ConsecutiveCutoffConfidence(
            decay_factor=0.85,
            confidence_boost_per_cutoff=0.15,
            max_consecutive_cutoffs=5
        )

        low_entropy = {'mean': 0.2, 'variance': 0.01, 'trend': -0.1}
        normal_entropy = {'mean': 0.6, 'variance': 0.03, 'trend': 0.01}

        for _ in range(3):
            confidence.update(True, low_entropy, van_event=False, p_harm_raw=0.4)

        assert confidence.consecutive_cutoff_count == 3

        confidence.update(False, normal_entropy, van_event=False, p_harm_raw=0.1)

        assert confidence.consecutive_cutoff_count == 0

    def test_alternating_cutoff_scenario(self):
        """测试交替截断场景（检测抖动）"""
        confidence = ConsecutiveCutoffConfidence(
            decay_factor=0.85,
            confidence_boost_per_cutoff=0.15,
            max_consecutive_cutoffs=5
        )

        low_entropy = {'mean': 0.2, 'variance': 0.01, 'trend': -0.1}
        normal_entropy = {'mean': 0.6, 'variance': 0.03, 'trend': 0.01}

        for i in range(6):
            entropy = low_entropy if i % 2 == 0 else normal_entropy
            result = confidence.update(i % 2 == 0, entropy, van_event=False, p_harm_raw=0.2)

            if i % 2 == 0:
                assert result['consecutive_cutoff_count'] == 1
            else:
                assert result['consecutive_cutoff_count'] == 0


class TestConsecutiveConfidenceEdgeCases:
    """测试连续截断信心的边界情况"""

    def test_empty_entropy_stats(self):
        """测试空熵统计"""
        confidence = ConsecutiveCutoffConfidence()

        result = confidence.update(True, None, van_event=False, p_harm_raw=0.3)

        assert 'confidence' in result
        assert 'raw_confidence' in result

    def test_single_frame(self):
        """测试单帧情况"""
        confidence = ConsecutiveCutoffConfidence()

        result = confidence.update(True, {'mean': 0.2, 'variance': 0.01, 'trend': -0.1}, van_event=False, p_harm_raw=0.3)

        assert result['consecutive_cutoff_count'] == 1
        assert not result['is_over_cutoff_risk']

    def test_all_same_frame(self):
        """测试全部相同帧"""
        confidence = ConsecutiveCutoffConfidence(max_consecutive_cutoffs=3)

        entropy_stats = {'mean': 0.2, 'variance': 0.01, 'trend': -0.1}

        for _ in range(10):
            result = confidence.update(True, entropy_stats, van_event=False, p_harm_raw=0.3)

        assert result['is_over_cutoff_risk']
        override, _ = confidence.should_override_cutoff()
        assert override

    def test_confidence_history_limit(self):
        """测试信心历史限制"""
        confidence = ConsecutiveCutoffConfidence()

        entropy_stats = {'mean': 0.3, 'variance': 0.02, 'trend': 0.0}

        for _ in range(150):
            confidence.update(False, entropy_stats, van_event=False, p_harm_raw=0.1)

        assert len(confidence.confidence_history) <= 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])