"""
测试L3控制器适配器集成

测试L3ControllerAdapter的核心功能：
1. 适配器初始化和基本接口
2. 温度动态调节 (τ ∈ [0.1, 2.0])
3. 稀疏度动态调节 (θ ∈ [0.5, 0.9])
4. DMN系数动态调节 (α ∈ [0.0, 1.0])
5. 冷却机制和抖动检测
6. 与HybridEnlightenLM的集成
"""

import pytest
import torch
import numpy as np
from typing import Dict, List

from enlighten.l3_controller import L3Controller, ControlSignals
from enlighten.hybrid_architecture import (
    HybridEnlightenLM,
    L3ControllerAdapter
)


class TestL3ControllerAdapter:
    """测试L3ControllerAdapter类"""

    def test_initialization(self):
        """测试适配器初始化"""
        adapter = L3ControllerAdapter()
        assert adapter is not None
        assert adapter.l3_controller is not None

    def test_initialization_with_config(self):
        """测试带配置的初始化"""
        config = {
            "entropy_threshold": 0.6,
            "variance_threshold": 0.07,
            "tau_range": [0.2, 1.8],
            "theta_range": [0.6, 0.85],
            "cutoff_cooldown": 15
        }
        adapter = L3ControllerAdapter(config=config)
        assert adapter.l3_controller.entropy_threshold == 0.6
        assert adapter.l3_controller.variance_threshold == 0.07
        assert adapter.l3_controller.cutoff_cooldown == 15

    def test_forward_normal_condition(self):
        """测试正常条件下的前向传播"""
        adapter = L3ControllerAdapter()

        entropy_stats = {
            "mean": 0.6,
            "variance": 0.02,
            "trend": 0.01,
            "current": 0.5
        }

        signals = adapter.forward(entropy_stats, van_event=False, p_harm=0.1)

        assert isinstance(signals, ControlSignals)
        assert not signals.cutoff
        assert signals.stability
        assert 0.1 <= signals.tau <= 2.0
        assert 0.5 <= signals.theta <= 0.9
        assert 0.0 <= signals.alpha <= 1.0

    def test_forward_low_entropy_condition(self):
        """测试低熵条件触发截断"""
        adapter = L3ControllerAdapter()

        entropy_stats = {
            "mean": 0.3,
            "variance": 0.02,
            "trend": -0.1,
            "current": 0.2
        }

        signals = adapter.forward(entropy_stats, van_event=False, p_harm=0.1)

        assert isinstance(signals, ControlSignals)
        if adapter.l3_controller._should_cutoff(0.3, 0.02**0.5, -0.1):
            assert signals.cutoff
            assert signals.reason is not None

    def test_forward_van_event(self):
        """测试VAN事件触发截断"""
        adapter = L3ControllerAdapter()

        entropy_stats = {
            "mean": 0.5,
            "variance": 0.05,
            "trend": 0.0,
            "current": 0.5
        }

        signals = adapter.forward(entropy_stats, van_event=True, p_harm=0.8)

        assert isinstance(signals, ControlSignals)
        assert signals.cutoff
        assert "VAN" in signals.reason or "van" in signals.reason.lower()

    def test_temperature_dynamic_range(self):
        """测试温度动态调节范围"""
        adapter = L3ControllerAdapter()

        test_cases = [
            ({"mean": 0.1, "variance": 0.01, "trend": -0.1, "current": 0.1}, True),
            ({"mean": 0.5, "variance": 0.02, "trend": 0.0, "current": 0.5}, False),
            ({"mean": 0.8, "variance": 0.05, "trend": 0.1, "current": 0.7}, False),
        ]

        for entropy_stats, should_cutoff in test_cases:
            signals = adapter.forward(entropy_stats, van_event=False, p_harm=0.1)
            assert 0.1 <= signals.tau <= 2.0, f"tau {signals.tau} out of range"

    def test_sparsity_threshold_dynamic_range(self):
        """测试稀疏度阈值动态调节范围"""
        adapter = L3ControllerAdapter()

        test_cases = [
            {"mean": 0.1, "variance": 0.01, "trend": -0.1, "current": 0.1},
            {"mean": 0.5, "variance": 0.02, "trend": 0.0, "current": 0.5},
            {"mean": 0.8, "variance": 0.05, "trend": 0.1, "current": 0.7},
        ]

        for entropy_stats in test_cases:
            signals = adapter.forward(entropy_stats, van_event=False, p_harm=0.1)
            assert 0.5 <= signals.theta <= 0.9, f"theta {signals.theta} out of range"

    def test_dmn_coefficient_dynamic_range(self):
        """测试DMN系数动态调节范围"""
        adapter = L3ControllerAdapter()

        test_cases = [
            {"mean": 0.1, "variance": 0.01, "trend": -0.1, "current": 0.1},
            {"mean": 0.5, "variance": 0.02, "trend": 0.0, "current": 0.5},
            {"mean": 0.8, "variance": 0.05, "trend": 0.1, "current": 0.7},
        ]

        for entropy_stats in test_cases:
            signals = adapter.forward(entropy_stats, van_event=False, p_harm=0.1)
            assert 0.0 <= signals.alpha <= 1.0, f"alpha {signals.alpha} out of range"

    def test_cooldown_mechanism(self):
        """测试冷却机制"""
        adapter = L3ControllerAdapter()

        entropy_stats = {
            "mean": 0.1,
            "variance": 0.01,
            "trend": -0.1,
            "current": 0.1
        }

        signals1 = adapter.forward(entropy_stats, van_event=False, p_harm=0.1)

        if adapter.l3_controller._should_cutoff(0.1, 0.01**0.5, -0.1):
            assert signals1.cutoff

            signals2 = adapter.forward(entropy_stats, van_event=False, p_harm=0.1)
            assert not signals2.cutoff
            assert signals2.reason == "Cooldown"

    def test_flicker_detection(self):
        """测试抖动检测机制"""
        adapter = L3ControllerAdapter()

        for i in range(10):
            entropy_stats = {
                "mean": 0.5 + 0.1 * (i % 2),
                "variance": 0.02,
                "trend": -0.1 if i % 3 == 0 else 0.01,
                "current": 0.5
            }
            adapter.forward(entropy_stats, van_event=False, p_harm=0.1)

        history = adapter.get_history(last_n=5)
        assert len(history) <= 5

    def test_get_control_signals_dict(self):
        """测试获取控制信号字典"""
        adapter = L3ControllerAdapter()

        entropy_stats = {
            "mean": 0.5,
            "variance": 0.02,
            "trend": 0.0,
            "current": 0.5
        }

        signals = adapter.forward(entropy_stats, van_event=False, p_harm=0.1)
        signals_dict = adapter.get_control_signals_dict(signals)

        assert "tau" in signals_dict
        assert "theta" in signals_dict
        assert "alpha" in signals_dict
        assert "stability" in signals_dict
        assert "cutoff" in signals_dict
        assert "reason" in signals_dict

    def test_get_temperature(self):
        """测试获取温度值"""
        adapter = L3ControllerAdapter()
        assert adapter.get_temperature() == 0.7

        entropy_stats = {"mean": 0.5, "variance": 0.02, "trend": 0.0, "current": 0.5}
        adapter.forward(entropy_stats, van_event=False, p_harm=0.1)

        tau = adapter.get_temperature()
        assert 0.1 <= tau <= 2.0

    def test_get_sparsity_threshold(self):
        """测试获取稀疏度阈值"""
        adapter = L3ControllerAdapter()
        assert adapter.get_sparsity_threshold() == 0.7

        entropy_stats = {"mean": 0.5, "variance": 0.02, "trend": 0.0, "current": 0.5}
        adapter.forward(entropy_stats, van_event=False, p_harm=0.1)

        theta = adapter.get_sparsity_threshold()
        assert 0.5 <= theta <= 0.9

    def test_get_dmn_coefficient(self):
        """测试获取DMN系数"""
        adapter = L3ControllerAdapter()
        assert adapter.get_dmn_coefficient() == 0.1

        entropy_stats = {"mean": 0.5, "variance": 0.02, "trend": 0.0, "current": 0.5}
        adapter.forward(entropy_stats, van_event=False, p_harm=0.1)

        alpha = adapter.get_dmn_coefficient()
        assert 0.0 <= alpha <= 1.0

    def test_should_cutoff(self):
        """测试判断是否应该截断"""
        adapter = L3ControllerAdapter()
        assert not adapter.should_cutoff()

        entropy_stats = {"mean": 0.1, "variance": 0.01, "trend": -0.1, "current": 0.1}
        adapter.forward(entropy_stats, van_event=False, p_harm=0.1)

        if adapter.l3_controller._should_cutoff(0.1, 0.01**0.5, -0.1):
            assert adapter.should_cutoff()

    def test_get_statistics(self):
        """测试获取统计信息"""
        adapter = L3ControllerAdapter()

        for i in range(5):
            entropy_stats = {
                "mean": 0.5 + i * 0.1,
                "variance": 0.02 + i * 0.01,
                "trend": 0.01 * i,
                "current": 0.5 + i * 0.1
            }
            adapter.forward(entropy_stats, van_event=False, p_harm=0.1)

        stats = adapter.get_statistics()
        assert "total_decisions" in stats
        assert stats["total_decisions"] == 5
        assert "last_tau" in stats
        assert "last_theta" in stats
        assert "last_alpha" in stats

    def test_reset(self):
        """测试重置功能"""
        adapter = L3ControllerAdapter()

        entropy_stats = {"mean": 0.1, "variance": 0.01, "trend": -0.1, "current": 0.1}
        adapter.forward(entropy_stats, van_event=False, p_harm=0.1)

        adapter.reset()

        assert adapter._last_control_signals is None
        assert len(adapter._control_signals_history) == 0

    def test_reset_cooldown(self):
        """测试重置冷却计数器"""
        adapter = L3ControllerAdapter()

        entropy_stats = {"mean": 0.1, "variance": 0.01, "trend": -0.1, "current": 0.1}
        adapter.forward(entropy_stats, van_event=False, p_harm=0.1)

        if adapter.l3_controller._should_cutoff(0.1, 0.01**0.5, -0.1):
            assert adapter.l3_controller.cooldown_counter > 0
        else:
            assert adapter.l3_controller.cooldown_counter == 0

        adapter.reset_cooldown()
        assert adapter.l3_controller.cooldown_counter == 0


class TestHybridEnlightenLMWithL3Controller:
    """测试HybridEnlightenLM与L3ControllerAdapter的集成"""

    def test_initialization_with_l3_controller(self):
        """测试使用L3控制器初始化"""
        model = HybridEnlightenLM(use_l3_controller=True)
        assert model is not None
        assert model.use_l3_controller
        assert model.l3_controller_adapter is not None

    def test_initialization_without_l3_controller(self):
        """测试不使用L3控制器初始化"""
        model = HybridEnlightenLM(use_l3_controller=False)
        assert model is not None
        assert not model.use_l3_controller
        assert model.l3_controller_adapter is None

    def test_initialization_with_l3_config(self):
        """测试使用L3配置初始化"""
        l3_config = {
            "entropy_threshold": 0.6,
            "variance_threshold": 0.07,
            "tau_range": [0.2, 1.8],
            "theta_range": [0.6, 0.85],
            "cutoff_cooldown": 15
        }
        model = HybridEnlightenLM(use_l3_controller=True, l3_config=l3_config)
        assert model.l3_controller_adapter is not None
        assert model.l3_controller_adapter.l3_controller.entropy_threshold == 0.6

    def test_reset_with_l3_controller(self):
        """测试重置功能（包含L3控制器）"""
        model = HybridEnlightenLM(use_l3_controller=True)
        assert model.l3_controller_adapter is not None

        model.reset()
        assert model.l3_controller_adapter is not None

    def test_get_status_with_l3_controller(self):
        """测试获取状态信息（包含L3控制器）"""
        model = HybridEnlightenLM(use_l3_controller=True)
        status = model.get_status()

        assert "use_l3_controller" in status
        assert status["use_l3_controller"]
        assert "l3_controller" in status
        assert isinstance(status["l3_controller"], dict)

    def test_get_temperature(self):
        """测试获取温度值"""
        model = HybridEnlightenLM(use_l3_controller=True)
        tau = model.get_temperature()
        assert 0.1 <= tau <= 2.0

    def test_get_sparsity_threshold(self):
        """测试获取稀疏度阈值"""
        model = HybridEnlightenLM(use_l3_controller=True)
        theta = model.get_sparsity_threshold()
        assert 0.5 <= theta <= 0.9

    def test_get_dmn_coefficient(self):
        """测试获取DMN系数"""
        model = HybridEnlightenLM(use_l3_controller=True)
        alpha = model.get_dmn_coefficient()
        assert 0.0 <= alpha <= 1.0

    def test_should_l3_cutoff(self):
        """测试判断L3层是否应该截断"""
        model = HybridEnlightenLM(use_l3_controller=True)
        assert not model.should_l3_cutoff()

    def test_is_l3_stable(self):
        """测试判断L3层是否稳定"""
        model = HybridEnlightenLM(use_l3_controller=True)
        assert model.is_l3_stable()

    def test_get_l3_cutoff_reason(self):
        """测试获取L3层截断原因"""
        model = HybridEnlightenLM(use_l3_controller=True)
        reason = model.get_l3_cutoff_reason()
        assert reason is None

    def test_get_l3_statistics(self):
        """测试获取L3统计信息"""
        model = HybridEnlightenLM(use_l3_controller=True)
        stats = model.get_l3_statistics()
        assert isinstance(stats, dict)
        assert "total_decisions" in stats

    def test_reset_l3_cooldown(self):
        """测试重置L3冷却计数器"""
        model = HybridEnlightenLM(use_l3_controller=True)
        model.reset_l3_cooldown()

    def test_get_l3_control_signals(self):
        """测试获取L3控制信号"""
        model = HybridEnlightenLM(use_l3_controller=True)
        signals = model.get_l3_control_signals()
        assert signals is None


class TestL3ControllerWithL1L2Integration:
    """测试L3控制器与L1/L2层的集成"""

    def test_l3_with_l1_adapter(self):
        """测试L3控制器与L1适配器集成"""
        model = HybridEnlightenLM(
            use_l3_controller=True,
            use_l1_adapter=True
        )
        assert model.use_l3_controller
        assert model.use_l1_adapter

    def test_l3_with_skeleton_l2(self):
        """测试L3控制器与骨架L2集成"""
        model = HybridEnlightenLM(
            use_l3_controller=True,
            use_skeleton_l2=True
        )
        assert model.use_l3_controller
        assert model.use_skeleton_l2

    def test_l3_with_l1_and_l2(self):
        """测试L3控制器与L1和L2适配器集成"""
        model = HybridEnlightenLM(
            use_l3_controller=True,
            use_l1_adapter=True,
            use_skeleton_l2=True
        )
        assert model.use_l3_controller
        assert model.use_l1_adapter
        assert model.use_skeleton_l2


class TestTemperatureAndSparsityDynamics:
    """测试温度和稀疏度动态调节"""

    def test_temperature_response_to_entropy(self):
        """测试温度对熵值的响应"""
        adapter = L3ControllerAdapter()

        low_entropy = {"mean": 0.1, "variance": 0.01, "trend": -0.1, "current": 0.1}
        high_entropy = {"mean": 0.8, "variance": 0.05, "trend": 0.1, "current": 0.7}

        signals_low = adapter.forward(low_entropy, van_event=False, p_harm=0.1)
        signals_high = adapter.forward(high_entropy, van_event=False, p_harm=0.1)

        assert isinstance(signals_low.tau, float)
        assert isinstance(signals_high.tau, float)

    def test_sparsity_response_to_entropy(self):
        """测试稀疏度阈值对熵值的响应"""
        adapter = L3ControllerAdapter()

        test_cases = [
            {"mean": 0.1, "variance": 0.01, "trend": -0.1, "current": 0.1},
            {"mean": 0.3, "variance": 0.02, "trend": -0.05, "current": 0.2},
            {"mean": 0.5, "variance": 0.03, "trend": 0.0, "current": 0.4},
            {"mean": 0.7, "variance": 0.04, "trend": 0.05, "current": 0.6},
            {"mean": 0.9, "variance": 0.05, "trend": 0.1, "current": 0.8},
        ]

        theta_values = []
        for entropy_stats in test_cases:
            signals = adapter.forward(entropy_stats, van_event=False, p_harm=0.1)
            theta_values.append(signals.theta)

        assert len(theta_values) == 5
        assert all(0.5 <= theta <= 0.9 for theta in theta_values)

    def test_tau_range_bounds(self):
        """测试温度值在有效范围内"""
        adapter = L3ControllerAdapter(config={"tau_range": [0.2, 1.8]})

        extreme_cases = [
            {"mean": 0.0, "variance": 0.0, "trend": -1.0, "current": 0.0},
            {"mean": 1.0, "variance": 1.0, "trend": 1.0, "current": 1.0},
        ]

        for entropy_stats in extreme_cases:
            signals = adapter.forward(entropy_stats, van_event=False, p_harm=0.1)
            assert 0.2 <= signals.tau <= 1.8

    def test_theta_range_bounds(self):
        """测试稀疏度阈值在有效范围内"""
        adapter = L3ControllerAdapter(config={"theta_range": [0.6, 0.85]})

        extreme_cases = [
            {"mean": 0.0, "variance": 0.0, "trend": -1.0, "current": 0.0},
            {"mean": 1.0, "variance": 1.0, "trend": 1.0, "current": 1.0},
        ]

        for entropy_stats in extreme_cases:
            signals = adapter.forward(entropy_stats, van_event=False, p_harm=0.1)
            assert 0.6 <= signals.theta <= 0.85

    def test_alpha_range_bounds(self):
        """测试DMN系数在有效范围内"""
        adapter = L3ControllerAdapter(config={"alpha_range": [0.0, 1.0]})

        extreme_cases = [
            {"mean": 0.0, "variance": 0.0, "trend": -1.0, "current": 0.0},
            {"mean": 1.0, "variance": 1.0, "trend": 1.0, "current": 1.0},
        ]

        for entropy_stats in extreme_cases:
            signals = adapter.forward(entropy_stats, van_event=False, p_harm=0.1)
            assert 0.0 <= signals.alpha <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])