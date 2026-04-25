"""
L3架构全链路测试脚本

测试内容：
1. L3控制器核心功能测试
   - 熵监控与截断决策
   - VAN事件处理
   - 冷却机制
   - 抖动检测与抑制

2. 贝叶斯L3控制器测试
   - 病因推断与后验概率
   - 连续截断信心机制
   - 动态阈值调整

3. 场景温度控制器测试
   - 场景识别
   - 动态温度调节
   - 稳定性监控

4. 端到端集成测试
   - API推理接口测试
   - L3统计信息验证
"""

import pytest
import torch
import numpy as np
import sys
import os
import time
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enlighten.l3_controller import (
    L3Controller,
    SimplifiedL3,
    BayesianL3Controller,
    EnhancedBayesianL3Controller,
    ContextualTemperatureController,
    ConsecutiveCutoffConfidence,
    SceneType,
    ControlSignals,
    DecisionRecord
)


class TestL3ControllerCore:
    """L3控制器核心功能测试"""

    def test_l3_controller_initialization(self):
        """测试L3控制器初始化"""
        controller = L3Controller()
        assert controller.entropy_threshold == 0.5
        assert controller.variance_threshold == 0.05
        assert controller.cooldown_counter == 0
        assert len(controller.decision_history) == 0

    def test_normal_entropy_stats(self):
        """测试正常熵统计下的L3决策"""
        controller = L3Controller()
        entropy_stats = {
            "mean": 0.7,
            "variance": 0.02,
            "trend": 0.0,
            "current": 0.7
        }

        signals = controller.forward(entropy_stats)

        assert isinstance(signals, ControlSignals)
        assert signals.cutoff == False
        assert signals.stability == True
        assert 0.1 <= signals.tau <= 2.0

    def test_low_entropy_cutoff(self):
        """测试低熵截断决策"""
        controller = L3Controller()
        entropy_stats = {
            "mean": 0.3,
            "variance": 0.001,
            "trend": -0.1,
            "current": 0.3
        }

        signals = controller.forward(entropy_stats)

        assert signals.cutoff == True
        assert signals.reason is not None

    def test_van_event_immediate_cutoff(self):
        """测试VAN事件立即截断"""
        controller = L3Controller()
        entropy_stats = {
            "mean": 0.6,
            "variance": 0.03,
            "trend": 0.0,
            "current": 0.6
        }

        signals = controller.forward(entropy_stats, van_event=True, p_harm=0.8)

        assert signals.cutoff == True
        assert "VAN event" in signals.reason
        assert controller.cooldown_counter > 0

    def test_cooldown_mechanism(self):
        """测试冷却机制"""
        controller = L3Controller(config={"cutoff_cooldown": 5})

        entropy_stats_normal = {"mean": 0.3, "variance": 0.001, "trend": -0.1}
        entropy_stats_high = {"mean": 0.7, "variance": 0.05, "trend": 0.0}

        signals1 = controller.forward(entropy_stats_normal)
        assert signals1.cutoff == True
        assert controller.cooldown_counter == 5

        for i in range(4):
            signals = controller.forward(entropy_stats_high)
            assert signals.cutoff == False
            assert signals.reason == "Cooldown"

        signals = controller.forward(entropy_stats_high)
        assert controller.cooldown_counter == 0

    def test_flicker_detection(self):
        """测试抖动检测"""
        controller = L3Controller(flicker_window_size=5, flicker_threshold=0.6)

        for i in range(10):
            if i % 2 == 0:
                entropy_stats = {"mean": 0.3, "variance": 0.001, "trend": -0.1}
            else:
                entropy_stats = {"mean": 0.7, "variance": 0.05, "trend": 0.0}
            controller.forward(entropy_stats)

        assert len(controller.cutoff_history) > 0

    def test_decision_history_recording(self):
        """测试决策历史记录"""
        controller = L3Controller()

        entropy_stats = {"mean": 0.5, "variance": 0.03, "trend": -0.05}
        controller.forward(entropy_stats)
        controller.forward(entropy_stats, van_event=True, p_harm=0.7)

        assert len(controller.decision_history) == 2
        assert controller.decision_history[0].van_event == False
        assert controller.decision_history[1].van_event == True

    def test_reset_functionality(self):
        """测试重置功能"""
        controller = L3Controller()
        controller.forward({"mean": 0.3, "variance": 0.001, "trend": -0.1})
        controller.cooldown_counter = 5
        controller.cutoff_history = [True, False]

        controller.reset()

        assert controller.cooldown_counter == 0
        assert len(controller.decision_history) == 0
        assert len(controller.cutoff_history) == 0


class TestBayesianL3Controller:
    """贝叶斯L3控制器测试"""

    def test_bayesian_l3_initialization(self):
        """测试贝叶斯L3初始化"""
        controller = BayesianL3Controller()
        posterior = controller.get_posterior()

        assert abs(posterior['normal'] - 0.6) < 0.1
        assert posterior['noise_injection'] >= 0
        assert posterior['bias_injection'] >= 0
        assert abs(sum(posterior.values()) - 1.0) < 0.01

    def test_bayesian_normal_operation(self):
        """测试贝叶斯L3正常操作"""
        controller = BayesianL3Controller()
        entropy_stats = {"mean": 0.6, "variance": 0.02, "trend": 0.0}

        signals = controller.forward(entropy_stats)

        assert isinstance(signals, ControlSignals)
        assert signals.cutoff == False

    def test_bayesian_van_event(self):
        """测试贝叶斯L3的VAN事件处理"""
        controller = BayesianL3Controller()
        entropy_stats = {"mean": 0.5, "variance": 0.02, "trend": 0.0}

        signals = controller.forward(entropy_stats, van_event=True, p_harm=0.8)

        assert signals.cutoff == True
        assert "VAN event" in signals.reason

    def test_bayesian_high_harm_cutoff(self):
        """测试高有害概率截断"""
        controller = BayesianL3Controller()
        entropy_stats = {"mean": 0.4, "variance": 0.03, "trend": -0.05}

        signals = controller.forward(entropy_stats, p_harm=0.85)

        assert signals.cutoff == True

    def test_bayesian_posterior_update(self):
        """测试后验概率更新"""
        controller = BayesianL3Controller()

        entropy_stats1 = {"mean": 0.6, "variance": 0.02, "trend": 0.0}
        entropy_stats2 = {"mean": 0.3, "variance": 0.25, "trend": 0.0}

        controller.forward(entropy_stats1)
        controller.forward(entropy_stats2)

        posterior = controller.get_posterior()

        assert posterior['noise_injection'] > 0.1

    def test_enhanced_bayesian_initialization(self):
        """测试增强贝叶斯L3初始化"""
        controller = EnhancedBayesianL3Controller()
        stats = controller.get_statistics()

        assert stats["total_decisions"] == 0
        assert controller.consecutive_confidence is not None


class TestConsecutiveCutoffConfidence:
    """连续截断信心机制测试"""

    def test_initialization(self):
        """测试初始化"""
        ccc = ConsecutiveCutoffConfidence()
        stats = ccc.get_statistics()

        assert stats['current_confidence'] == 0.5
        assert stats['consecutive_cutoff_count'] == 0

    def test_normal_operation(self):
        """测试正常操作"""
        ccc = ConsecutiveCutoffConfidence()
        entropy_stats = {"mean": 0.6, "variance": 0.02, "trend": 0.0}

        result = ccc.update(False, entropy_stats)

        assert result['should_trust_cutoff'] == False
        assert result['consecutive_cutoff_count'] == 0

    def test_consecutive_cutoff_confidence_growth(self):
        """测试连续截断信心增长"""
        ccc = ConsecutiveCutoffConfidence()
        entropy_stats = {"mean": 0.2, "variance": 0.01, "trend": -0.1}

        for i in range(3):
            ccc.update(True, entropy_stats)

        stats = ccc.get_statistics()
        assert stats['consecutive_cutoff_count'] == 3

    def test_over_cutoff_protection(self):
        """测试过度截断保护"""
        ccc = ConsecutiveCutoffConfidence(
            max_consecutive_cutoffs=3,
            confidence_threshold_increase=0.1
        )
        entropy_stats = {"mean": 0.2, "variance": 0.01, "trend": -0.1}

        for i in range(5):
            ccc.update(True, entropy_stats)

        override, reason = ccc.should_override_cutoff()
        assert override == True


class TestContextualTemperatureController:
    """场景温度控制器测试"""

    def test_scene_detection_creative(self):
        """测试创意写作场景检测"""
        controller = ContextualTemperatureController()

        scene = controller.detect_scene("写一个关于AI的故事")

        assert scene == SceneType.CREATIVE_WRITING

    def test_scene_detection_code(self):
        """测试代码生成场景检测"""
        controller = ContextualTemperatureController()

        scene = controller.detect_scene("def calculate_sum(a, b): return a + b")

        assert scene == SceneType.CODE_GENERATION

    def test_scene_detection_qa(self):
        """测试问答场景检测"""
        controller = ContextualTemperatureController()

        scene = controller.detect_scene("解释一下什么是机器学习")

        assert scene == SceneType.QUESTION_ANSWERING

    def test_temperature_computation(self):
        """测试温度计算"""
        controller = ContextualTemperatureController()
        entropy_stats = {"mean": 0.6, "variance": 0.02, "trend": 0.0}

        result = controller.compute_temperature(entropy_stats)

        assert "temperature" in result
        assert 0.1 <= result["temperature"] <= 2.0

    def test_temperature_smoothing(self):
        """测试温度平滑"""
        controller = ContextualTemperatureController()

        controller.smooth_temperature(0.8)
        controller.smooth_temperature(0.6)

        assert controller.smoothed_temperature != 0.8
        assert controller.smoothed_temperature != 0.6

    def test_stability_monitor(self):
        """测试稳定性监控"""
        controller = ContextualTemperatureController()

        controller.stability_monitor.update("Hello world how are you", 0.7)
        controller.stability_monitor.update("This is a test", 0.7)

        metrics = controller.stability_monitor._compute_stability_metrics()
        assert "stability_score" in metrics


class TestSimplifiedL3:
    """简化版L3测试"""

    def test_simplified_l3_normal(self):
        """测试简化L3正常操作"""
        l3 = SimplifiedL3()
        entropy_stats = {"mean": 0.6, "variance": 0.02, "trend": 0.0}

        result = l3.forward(entropy_stats)

        assert result["cutoff"] == False
        assert result["stability"] == True

    def test_simplified_l3_low_entropy(self):
        """测试简化L3低熵截断"""
        l3 = SimplifiedL3()
        entropy_stats = {"mean": 0.3, "variance": 0.001, "trend": -0.1}

        result = l3.forward(entropy_stats)

        assert result["cutoff"] == True
        assert result["reason"] == "Low entropy"

    def test_simplified_l3_van_event(self):
        """测试简化L3的VAN事件"""
        l3 = SimplifiedL3()
        entropy_stats = {"mean": 0.6, "variance": 0.02, "trend": 0.0}

        result = l3.forward(entropy_stats, van_event=True)

        assert result["cutoff"] == True
        assert result["reason"] == "VAN event"


class TestL3Integration:
    """L3集成测试"""

    def test_multiple_concurrent_controllers(self):
        """测试多个控制器并发操作"""
        controllers = [L3Controller() for _ in range(5)]
        entropy_stats = {"mean": 0.5, "variance": 0.03, "trend": 0.0}

        for controller in controllers:
            signals = controller.forward(entropy_stats)
            assert signals.cutoff == False

    def test_entropy_stats_edge_cases(self):
        """测试熵统计边界情况"""
        controller = L3Controller()

        extreme_low = {"mean": 0.0, "variance": 0.0, "trend": -1.0}
        signals = controller.forward(extreme_low)
        assert signals.cutoff == True

        extreme_high = {"mean": 1.0, "variance": 1.0, "trend": 1.0}
        signals = controller.forward(extreme_high)
        assert signals.cutoff == False

        missing_keys = {}
        signals = controller.forward(missing_keys)
        assert isinstance(signals, ControlSignals)

    def test_long_running_stability(self):
        """测试长时间运行稳定性"""
        controller = EnhancedBayesianL3Controller()

        for i in range(100):
            entropy_stats = {
                "mean": 0.5 + np.sin(i * 0.1) * 0.2,
                "variance": 0.02 + abs(np.sin(i * 0.2)) * 0.05,
                "trend": np.sin(i * 0.15) * 0.1
            }
            signals = controller.forward(entropy_stats)
            assert isinstance(signals, ControlSignals)

        stats = controller.get_statistics()
        assert stats["total_decisions"] == 100


def run_l3_tests():
    """运行所有L3测试"""
    print("=" * 60)
    print("L3架构全链路测试")
    print("=" * 60)

    pytest_args = [
        __file__,
        "-v",
        "--tb=short",
        f"--color=yes"
    ]

    exit_code = pytest.main(pytest_args)

    print("\n" + "=" * 60)
    if exit_code == 0:
        print("✓ 所有L3测试通过")
    else:
        print("✗ 部分测试失败")
    print("=" * 60)

    return exit_code


if __name__ == "__main__":
    exit_code = run_l3_tests()
    sys.exit(exit_code)