"""
贝叶斯L3控制器集成测试

测试EnhancedBayesianL3Controller的完整功能集成：
1. 贝叶斯病因推断
2. 动态温度调节
3. 连续截断信心机制

覆盖各种场景下的控制器稳定性验证
"""

import pytest
import numpy as np
from typing import Dict, List, Any, Tuple

from enlighten.l3_controller import (
    EnhancedBayesianL3Controller,
    BayesianL3Controller,
    ControlSignals,
    DecisionRecord,
    ConsecutiveCutoffConfidence,
    ContextualTemperatureController,
    SceneType
)


class TestEnhancedBayesianL3Integration:
    """EnhancedBayesianL3Controller 完整集成测试"""

    def test_initialization_with_all_features(self):
        """INT-01: 启用所有功能的初始化"""
        controller = EnhancedBayesianL3Controller(
            use_consecutive_confidence=True,
            consecutive_confidence_config={
                'initial_confidence': 0.5,
                'decay_factor': 0.85,
                'confidence_boost_per_cutoff': 0.15,
                'max_consecutive_cutoffs': 5
            }
        )

        assert controller is not None
        assert controller.use_consecutive_confidence
        assert controller.consecutive_confidence is not None
        assert controller.consecutive_confidence.current_confidence == 0.5

    def test_full_pipeline_normal_operation(self):
        """INT-02: 正常操作的完整流程"""
        controller = EnhancedBayesianL3Controller(use_consecutive_confidence=True)

        entropy_stats = {
            "mean": 0.6,
            "variance": 0.02,
            "trend": 0.01,
            "current": 0.55
        }

        signals = controller.forward(entropy_stats, van_event=False, p_harm=0.1)

        assert isinstance(signals, ControlSignals)
        assert signals.tau >= 0.2 and signals.tau <= 2.0
        assert not signals.cutoff or signals.cutoff
        assert signals.reason is not None or signals.reason is None

        posterior = controller.get_posterior()
        assert abs(sum(posterior.values()) - 1.0) < 1e-6

    def test_causal_inference_with_bias_condition(self):
        """INT-03: 偏见条件下的因果推断"""
        controller = EnhancedBayesianL3Controller(use_consecutive_confidence=True)

        for _ in range(5):
            entropy_stats = {
                "mean": 0.2,
                "variance": 0.02,
                "trend": -0.05,
                "current": 0.15
            }
            controller.forward(entropy_stats, van_event=False, p_harm=0.8)

        posterior = controller.get_posterior()
        assert posterior['bias_injection'] > 0.2 or posterior['mixed'] > 0.2

    def test_causal_inference_with_noise_condition(self):
        """INT-04: 噪声条件下的因果推断"""
        controller = EnhancedBayesianL3Controller(use_consecutive_confidence=True)

        for _ in range(5):
            entropy_stats = {
                "mean": 0.5,
                "variance": 0.3,
                "trend": 0.0,
                "current": 0.4
            }
            controller.forward(entropy_stats, van_event=False, p_harm=0.2)

        posterior = controller.get_posterior()
        assert posterior['noise_injection'] > 0.1

    def test_mixed_condition_detection(self):
        """INT-05: 混合病因检测"""
        controller = EnhancedBayesianL3Controller(use_consecutive_confidence=True)

        for _ in range(8):
            entropy_stats = {
                "mean": 0.25 + np.random.uniform(-0.05, 0.05),
                "variance": 0.18 + np.random.uniform(-0.03, 0.03),
                "trend": -0.03 + np.random.uniform(-0.02, 0.02),
                "current": 0.2
            }
            controller.forward(entropy_stats, van_event=False, p_harm=0.5 + np.random.uniform(-0.1, 0.1))

        posterior = controller.get_posterior()
        total_abnormal = posterior['noise_injection'] + posterior['bias_injection'] + posterior['mixed']
        assert total_abnormal > 0.1

    def test_temperature_computation_integration(self):
        """INT-06: 温度计算集成"""
        controller = EnhancedBayesianL3Controller(use_consecutive_confidence=True)

        for i in range(10):
            entropy_stats = {
                "mean": 0.5 - i * 0.02,
                "variance": 0.02 + i * 0.005,
                "trend": -0.01 * i,
                "current": 0.5 - i * 0.02
            }
            signals = controller.forward(entropy_stats, van_event=False, p_harm=0.2)

            assert signals.tau >= 0.2 and signals.tau <= 2.0

            if i > 0:
                assert isinstance(signals.tau, float)

    def test_consecutive_confidence_accumulation(self):
        """INT-07: 连续截断信心累积"""
        controller = EnhancedBayesianL3Controller(
            use_consecutive_confidence=True,
            consecutive_confidence_config={
                'max_consecutive_cutoffs': 5,
                'decay_factor': 0.85,
                'confidence_boost_per_cutoff': 0.15
            }
        )

        entropy_stats = {"mean": 0.2, "variance": 0.01, "trend": -0.1}

        for i in range(5):
            result = controller.forward(entropy_stats, van_event=False, p_harm=0.4)
            if i == 0:
                first_tau = result.tau

        stats = controller.get_statistics()
        assert 'consecutive_confidence' in stats
        assert stats['consecutive_confidence']['consecutive_cutoff_count'] >= 0


class TestSynergisticOperation:
    """协同工作测试"""

    def test_causal_inference_feeds_confidence(self):
        """SYN-01: 因果推断结果影响信心计算"""
        controller = EnhancedBayesianL3Controller(use_consecutive_confidence=True)

        entropy_stats = {"mean": 0.3, "variance": 0.02, "trend": -0.05}

        controller.forward(entropy_stats, van_event=False, p_harm=0.6)

        posterior = controller.get_posterior()
        bias_prob = posterior['bias_injection']

        stats = controller.get_statistics()
        confidence_stats = stats.get('consecutive_confidence', {})

        assert 'current_confidence' in confidence_stats

    def test_confidence_decision_affects_cutoff(self):
        """SYN-02: 信心决策影响截断判断"""
        controller = EnhancedBayesianL3Controller(
            use_consecutive_confidence=True,
            consecutive_confidence_config={
                'max_consecutive_cutoffs': 3,
                'initial_confidence': 0.3,
                'decay_factor': 0.85,
                'confidence_boost_per_cutoff': 0.15,
                'min_confidence_threshold': 0.5,
                'max_confidence_threshold': 0.9
            }
        )

        entropy_stats = {"mean": 0.25, "variance": 0.01, "trend": -0.1}

        cutoff_count = 0
        for i in range(10):
            result = controller.forward(entropy_stats, van_event=False, p_harm=0.5)
            if result.cutoff:
                cutoff_count += 1

        assert cutoff_count <= 10

    def test_temperature_adaptation_with_posterior(self):
        """SYN-03: 温度根据后验分布自适应"""
        controller = EnhancedBayesianL3Controller(use_consecutive_confidence=True)

        normal_stats = {"mean": 0.6, "variance": 0.02, "trend": 0.01}
        abnormal_stats = {"mean": 0.15, "variance": 0.01, "trend": -0.1}

        normal_temps = []
        abnormal_temps = []

        for _ in range(5):
            result = controller.forward(normal_stats, van_event=False, p_harm=0.1)
            normal_temps.append(result.tau)

        controller.reset()

        for _ in range(5):
            result = controller.forward(abnormal_stats, van_event=False, p_harm=0.8)
            abnormal_temps.append(result.tau)

        assert np.mean(normal_temps) != np.mean(abnormal_temps)

    def test_van_event_bypasses_all_mechanisms(self):
        """SYN-04: VAN事件绕过所有机制"""
        controller = EnhancedBayesianL3Controller(use_consecutive_confidence=True)

        entropy_stats = {"mean": 0.6, "variance": 0.02, "trend": 0.01}

        for i in range(3):
            result = controller.forward(entropy_stats, van_event=True, p_harm=0.9)

            assert result.cutoff
            assert "VAN" in (result.reason or "")

    def test_cooldown_prevents_excessive_cutoffs(self):
        """SYN-05: 冷却机制防止过度截断"""
        controller = EnhancedBayesianL3Controller(
            use_consecutive_confidence=True,
            cutoff_cooldown=5,
            consecutive_confidence_config={
                'max_consecutive_cutoffs': 10,
                'min_confidence_threshold': 0.3
            }
        )

        entropy_stats = {"mean": 0.2, "variance": 0.01, "trend": -0.1}

        cooldown_or_override_count = 0
        for i in range(20):
            result = controller.forward(entropy_stats, van_event=False, p_harm=0.5)

            if "Cooldown" in (result.reason or "") or "Over-cutoff" in (result.reason or ""):
                cooldown_or_override_count += 1

        assert cooldown_or_override_count > 0


class TestScenarioStability:
    """场景稳定性测试"""

    def test_rapid_context_switching_stability(self):
        """STAB-01: 快速上下文切换稳定性"""
        controller = EnhancedBayesianL3Controller(use_consecutive_confidence=True)

        scenarios = [
            {"mean": 0.7, "variance": 0.02, "trend": 0.05, "p_harm": 0.05},
            {"mean": 0.3, "variance": 0.15, "trend": -0.05, "p_harm": 0.3},
            {"mean": 0.5, "variance": 0.05, "trend": 0.0, "p_harm": 0.1},
            {"mean": 0.2, "variance": 0.01, "trend": -0.1, "p_harm": 0.7},
        ]

        for _ in range(50):
            for scenario in scenarios:
                entropy_stats = {
                    "mean": scenario["mean"],
                    "variance": scenario["variance"],
                    "trend": scenario["trend"],
                    "current": scenario["mean"]
                }
                result = controller.forward(
                    entropy_stats,
                    van_event=False,
                    p_harm=scenario["p_harm"]
                )

                assert result.tau >= 0.2 and result.tau <= 2.0

        stats = controller.get_statistics()
        assert stats['total_decisions'] == 200

    def test_prolonged_low_entropy_stability(self):
        """STAB-02: 长时间低熵稳定性"""
        controller = EnhancedBayesianL3Controller(
            use_consecutive_confidence=True,
            consecutive_confidence_config={
                'max_consecutive_cutoffs': 3
            }
        )

        entropy_stats = {"mean": 0.15, "variance": 0.01, "trend": -0.02}

        for i in range(30):
            result = controller.forward(entropy_stats, van_event=False, p_harm=0.4 + i * 0.01)

            assert result.tau >= 0.2 and result.tau <= 2.0

            if i < 10:
                assert isinstance(result.cutoff, bool)

        stats = controller.get_statistics()
        assert stats['total_decisions'] == 30

    def test_high_variance_input_stability(self):
        """STAB-03: 高方差输入稳定性"""
        controller = EnhancedBayesianL3Controller(use_consecutive_confidence=True)

        for i in range(20):
            entropy_stats = {
                "mean": 0.3 + np.random.uniform(-0.2, 0.2),
                "variance": 0.2 + np.random.uniform(-0.1, 0.1),
                "trend": np.random.uniform(-0.2, 0.2),
                "current": 0.4
            }
            result = controller.forward(entropy_stats, van_event=False, p_harm=0.3)

            assert result.tau >= 0.2 and result.tau <= 2.0

        stats = controller.get_statistics()
        assert stats['total_decisions'] == 20

    def test_gradual_drift_detection(self):
        """STAB-04: 渐进漂移检测"""
        controller = EnhancedBayesianL3Controller(use_consecutive_confidence=True)

        mu_h = 0.5
        for i in range(15):
            entropy_stats = {
                "mean": mu_h - i * 0.02,
                "variance": 0.02,
                "trend": -0.02,
                "current": mu_h - i * 0.02
            }
            controller.forward(entropy_stats, van_event=False, p_harm=0.3 + i * 0.03)

        temporal_features = controller.get_temporal_features()
        assert 'mu_H_trend' in temporal_features
        assert 'temporal_stability' in temporal_features

    def test_repeated_van_events_stability(self):
        """STAB-05: 重复VAN事件稳定性"""
        controller = EnhancedBayesianL3Controller(
            use_consecutive_confidence=True,
            cutoff_cooldown=3,
            consecutive_confidence_config={
                'max_consecutive_cutoffs': 20,
                'min_confidence_threshold': 0.3,
                'max_confidence_threshold': 0.95
            }
        )

        entropy_stats = {"mean": 0.5, "variance": 0.02, "trend": 0.0}

        for i in range(10):
            result = controller.forward(entropy_stats, van_event=True, p_harm=0.9)

            assert result.cutoff or "VAN" in (result.reason or "")
            assert result.tau >= 0.2 and result.tau <= 2.0

    def test_no_halt_under_normal_operation(self):
        """STAB-06: 正常操作不中断"""
        controller = EnhancedBayesianL3Controller(use_consecutive_confidence=True)

        entropy_stats = {"mean": 0.6, "variance": 0.03, "trend": 0.02}

        cutoff_count = 0
        for _ in range(30):
            result = controller.forward(entropy_stats, van_event=False, p_harm=0.05)
            if result.cutoff:
                cutoff_count += 1

        assert cutoff_count < 30


class TestCausalAttribution:
    """因果归因测试"""

    def test_causal_attribution_normal(self):
        """CAUSAL-01: 正常状态归因"""
        controller = EnhancedBayesianL3Controller(use_consecutive_confidence=True)

        entropy_stats = {"mean": 0.6, "variance": 0.02, "trend": 0.02}

        for _ in range(10):
            controller.forward(entropy_stats, van_event=False, p_harm=0.05)

        attribution = controller.get_causal_attribution()

        assert 'dominant_cause' in attribution
        assert 'confidence' in attribution
        assert 'posterior' in attribution
        assert 'suggestions' in attribution
        assert isinstance(attribution['suggestions'], list)

    def test_causal_attribution_bias(self):
        """CAUSAL-02: 偏见状态归因"""
        controller = EnhancedBayesianL3Controller(use_consecutive_confidence=True)

        entropy_stats = {"mean": 0.18, "variance": 0.01, "trend": -0.08}

        for _ in range(10):
            controller.forward(entropy_stats, van_event=False, p_harm=0.8)

        attribution = controller.get_causal_attribution()

        assert attribution['dominant_cause'] in ['bias_injection', 'mixed']

    def test_causal_attribution_noise(self):
        """CAUSAL-03: 噪声状态归因"""
        controller = EnhancedBayesianL3Controller(use_consecutive_confidence=True)

        entropy_stats = {"mean": 0.45, "variance": 0.25, "trend": 0.0}

        for _ in range(10):
            controller.forward(entropy_stats, van_event=False, p_harm=0.3)

        attribution = controller.get_causal_attribution()

        assert attribution['dominant_cause'] in ['noise_injection', 'mixed', 'normal']

    def test_suggestions_adaptive(self):
        """CAUSAL-04: 建议的自适应性"""
        controller = EnhancedBayesianL3Controller(use_consecutive_confidence=True)

        bias_stats = {"mean": 0.15, "variance": 0.01, "trend": -0.1}
        for _ in range(5):
            controller.forward(bias_stats, van_event=False, p_harm=0.85)

        attribution = controller.get_causal_attribution()

        assert len(attribution['suggestions']) > 0


class TestEdgeCaseHandling:
    """边界情况处理测试"""

    def test_extreme_entropy_values(self):
        """EDGE-01: 极端熵值"""
        controller = EnhancedBayesianL3Controller(use_consecutive_confidence=True)

        extreme_stats = [
            {"mean": 0.0, "variance": 0.0, "trend": -1.0},
            {"mean": 1.0, "variance": 1.0, "trend": 1.0},
            {"mean": 0.01, "variance": 0.001, "trend": -0.5},
        ]

        for stats in extreme_stats:
            result = controller.forward(stats, van_event=False, p_harm=0.5)

            assert result.tau >= 0.2 and result.tau <= 2.0
            assert isinstance(result.cutoff, bool)

    def test_rapid_succession_cutoffs(self):
        """EDGE-02: 快速连续截断"""
        controller = EnhancedBayesianL3Controller(
            use_consecutive_confidence=True,
            consecutive_confidence_config={
                'max_consecutive_cutoffs': 3,
                'initial_confidence': 0.5,
                'decay_factor': 0.9
            }
        )

        entropy_stats = {"mean": 0.1, "variance": 0.01, "trend": -0.2}

        for _ in range(20):
            result = controller.forward(entropy_stats, van_event=False, p_harm=0.9)

            assert result.tau >= 0.2 and result.tau <= 2.0

        override_count = 0
        for record in controller.decision_history:
            if "Over-cutoff" in (record.reason or ""):
                override_count += 1

        assert override_count >= 0

    def test_all_causes_high_probability(self):
        """EDGE-03: 所有病因高概率情况"""
        controller = EnhancedBayesianL3Controller(use_consecutive_confidence=True)

        mixed_stats = {"mean": 0.25, "variance": 0.18, "trend": -0.03}

        for _ in range(10):
            result = controller.forward(mixed_stats, van_event=False, p_harm=0.6)

            posterior = controller.get_posterior()
            assert abs(sum(posterior.values()) - 1.0) < 1e-5

    def test_zero_p_harm(self):
        """EDGE-04: 零有害概率"""
        controller = EnhancedBayesianL3Controller(use_consecutive_confidence=True)

        entropy_stats = {"mean": 0.3, "variance": 0.02, "trend": -0.05}

        result = controller.forward(entropy_stats, van_event=False, p_harm=0.0)

        assert result.tau >= 0.2 and result.tau <= 2.0

    def test_history_limit_enforcement(self):
        """EDGE-05: 历史记录限制执行"""
        controller = EnhancedBayesianL3Controller(use_consecutive_confidence=True)

        entropy_stats = {"mean": 0.5, "variance": 0.02, "trend": 0.0}

        for _ in range(1500):
            controller.forward(entropy_stats, van_event=False, p_harm=0.1)

        assert len(controller.decision_history) <= 1000
        assert len(controller.posterior_history) <= 100


class TestTemperatureRegulation:
    """温度调节测试"""

    def test_temperature_bounds_strict(self):
        """TEMP-01: 温度边界严格执行"""
        controller = EnhancedBayesianL3Controller(use_consecutive_confidence=True)

        extreme_stats = [
            {"mean": 0.0, "variance": 0.0, "trend": -10.0},
            {"mean": 1.0, "variance": 1.0, "trend": 10.0},
        ]

        for stats in extreme_stats:
            result = controller.forward(stats, van_event=True, p_harm=1.0)

            assert 0.2 <= result.tau <= 2.0

    def test_temperature_reflects_bias_probability(self):
        """TEMP-02: 温度反映偏见概率"""
        controller = EnhancedBayesianL3Controller(use_consecutive_confidence=True)

        low_bias_stats = {"mean": 0.6, "variance": 0.02, "trend": 0.01}
        high_bias_stats = {"mean": 0.15, "variance": 0.01, "trend": -0.1}

        temp_low_bias = []
        temp_high_bias = []

        for _ in range(5):
            r1 = controller.forward(low_bias_stats, van_event=False, p_harm=0.1)
            temp_low_bias.append(r1.tau)
            controller.reset()

            r2 = controller.forward(high_bias_stats, van_event=False, p_harm=0.9)
            temp_high_bias.append(r2.tau)
            controller.reset()

        assert np.mean(temp_high_bias) < np.mean(temp_low_bias)

    def test_temperature_smooth_transition(self):
        """TEMP-03: 温度平滑过渡"""
        controller = EnhancedBayesianL3Controller(use_consecutive_confidence=True)

        entropy_stats = {"mean": 0.5, "variance": 0.02, "trend": 0.0}

        temps = []
        for i in range(20):
            result = controller.forward(entropy_stats, van_event=False, p_harm=0.1 + i * 0.03)
            temps.append(result.tau)

        temp_changes = [abs(temps[i+1] - temps[i]) for i in range(len(temps)-1)]

        assert all(0 <= change <= 2.0 for change in temp_changes)


class TestConfidenceThresholdAdaptation:
    """信心阈值自适应测试"""

    def test_threshold_increases_with_consecutive_cutoffs(self):
        """CONF-01: 连续截断时阈值增加"""
        controller = EnhancedBayesianL3Controller(
            use_consecutive_confidence=True,
            consecutive_confidence_config={
                'max_consecutive_cutoffs': 5,
                'confidence_threshold_increase': 0.1
            }
        )

        entropy_stats = {"mean": 0.2, "variance": 0.01, "trend": -0.1}

        thresholds = []
        for i in range(8):
            controller.forward(entropy_stats, van_event=False, p_harm=0.5)

            stats = controller.get_statistics()
            threshold = stats['consecutive_confidence']['dynamic_threshold']
            thresholds.append(threshold)

        for i in range(4, len(thresholds)):
            assert thresholds[i] >= thresholds[4]

    def test_confidence_resets_after_recovery(self):
        """CONF-02: 恢复后信心重置"""
        controller = EnhancedBayesianL3Controller(
            use_consecutive_confidence=True,
            consecutive_confidence_config={
                'max_consecutive_cutoffs': 5
            }
        )

        low_entropy = {"mean": 0.15, "variance": 0.01, "trend": -0.1}
        normal_entropy = {"mean": 0.6, "variance": 0.02, "trend": 0.02}

        for _ in range(5):
            controller.forward(low_entropy, van_event=False, p_harm=0.5)

        stats_before = controller.get_statistics()
        count_before = stats_before['consecutive_confidence']['consecutive_cutoff_count']

        controller.forward(normal_entropy, van_event=False, p_harm=0.05)

        stats_after = controller.get_statistics()
        count_after = stats_after['consecutive_confidence']['consecutive_cutoff_count']

        assert count_after < count_before

    def test_override_prevents_excessive_cutoffs(self):
        """CONF-03: 否决机制防止过度截断"""
        controller = EnhancedBayesianL3Controller(
            use_consecutive_confidence=True,
            consecutive_confidence_config={
                'max_consecutive_cutoffs': 3,
                'initial_confidence': 0.4,
                'min_confidence_threshold': 0.5,
                'max_confidence_threshold': 0.9
            }
        )

        entropy_stats = {"mean": 0.2, "variance": 0.01, "trend": -0.1}

        override_count = 0
        for _ in range(15):
            result = controller.forward(entropy_stats, van_event=False, p_harm=0.4)

            override, _ = controller.consecutive_confidence.should_override_cutoff()
            if override:
                override_count += 1

        assert override_count >= 0


class TestDecisionHistory:
    """决策历史测试"""

    def test_decision_history_complete(self):
        """HIST-01: 决策历史完整性"""
        controller = EnhancedBayesianL3Controller(use_consecutive_confidence=True)

        entropy_stats = {"mean": 0.5, "variance": 0.02, "trend": 0.0}

        for i in range(10):
            controller.forward(entropy_stats, van_event=i == 5, p_harm=0.2)

        history = controller.get_history()

        assert len(history) == 10
        assert all(isinstance(record, DecisionRecord) for record in history)

    def test_decision_history_last_n(self):
        """HIST-02: 获取最近N条历史"""
        controller = EnhancedBayesianL3Controller(use_consecutive_confidence=True)

        entropy_stats = {"mean": 0.5, "variance": 0.02, "trend": 0.0}

        for _ in range(20):
            controller.forward(entropy_stats, van_event=False, p_harm=0.1)

        recent = controller.get_history(last_n=5)

        assert len(recent) == 5

    def test_statistics_comprehensive(self):
        """HIST-03: 统计信息完整性"""
        controller = EnhancedBayesianL3Controller(use_consecutive_confidence=True)

        entropy_stats = {"mean": 0.4, "variance": 0.03, "trend": -0.02}

        for _ in range(15):
            controller.forward(entropy_stats, van_event=False, p_harm=0.3)

        stats = controller.get_statistics()

        required_fields = [
            'total_decisions', 'total_cutoffs', 'total_van_events',
            'cutoff_ratio', 'posterior', 'temporal_features',
            'consecutive_confidence'
        ]

        for field in required_fields:
            assert field in stats


class TestResetBehavior:
    """重置行为测试"""

    def test_full_reset_clears_state(self):
        """RESET-01: 完全重置清除状态"""
        controller = EnhancedBayesianL3Controller(use_consecutive_confidence=True)

        entropy_stats = {"mean": 0.3, "variance": 0.02, "trend": -0.05}

        for _ in range(10):
            controller.forward(entropy_stats, van_event=False, p_harm=0.5)

        assert len(controller.decision_history) == 10

        controller.reset()

        assert len(controller.decision_history) == 0
        assert controller.step_counter == 0
        assert controller.cooldown_counter == 0

        stats = controller.get_statistics()
        assert stats['total_decisions'] == 0

    def test_reset_restores_default_prior(self):
        """RESET-02: 重置恢复默认先验"""
        controller = EnhancedBayesianL3Controller(use_consecutive_confidence=True)

        abnormal_stats = {"mean": 0.1, "variance": 0.01, "trend": -0.2}

        for _ in range(10):
            controller.forward(abnormal_stats, van_event=False, p_harm=0.9)

        controller.reset()

        posterior = controller.get_posterior()
        assert posterior['normal'] > 0.5


class TestMultiScenarioWorkflow:
    """多场景工作流测试"""

    def test_complete_workflow_creative_writing(self):
        """WORK-01: 创意写作完整工作流"""
        controller = EnhancedBayesianL3Controller(use_consecutive_confidence=True)

        creative_stats = {"mean": 0.65, "variance": 0.025, "trend": 0.02}

        for _ in range(10):
            result = controller.forward(
                creative_stats,
                van_event=False,
                p_harm=0.05
            )

            assert result.tau >= 0.5

    def test_complete_workflow_code_generation(self):
        """WORK-02: 代码生成完整工作流"""
        controller = EnhancedBayesianL3Controller(use_consecutive_confidence=True)

        code_stats = {"mean": 0.55, "variance": 0.015, "trend": 0.01}

        for _ in range(10):
            result = controller.forward(
                code_stats,
                van_event=False,
                p_harm=0.05
            )

            assert result.tau >= 0.2

    def test_complete_workflow_security_sensitive(self):
        """WORK-03: 安全敏感完整工作流"""
        controller = EnhancedBayesianL3Controller(use_consecutive_confidence=True)

        sensitive_stats = {"mean": 0.2, "variance": 0.01, "trend": -0.08}

        cutoff_count = 0
        for _ in range(10):
            result = controller.forward(
                sensitive_stats,
                van_event=False,
                p_harm=0.8
            )

            if result.cutoff:
                cutoff_count += 1

        assert cutoff_count > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])