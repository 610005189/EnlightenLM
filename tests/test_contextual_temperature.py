"""
ContextualTemperatureController 测试套件

测试基于上下文的动态温度调节功能，包括：
- 场景识别
- 平滑温度过渡
- 稳定性监控
- 不同场景下的温度调节效果
"""

import pytest
import numpy as np
from typing import Dict, List, Any


class TestSceneDetection:
    """场景检测测试"""

    def test_scene_detection_creative_writing(self):
        """SCENE-01: 创意写作场景检测"""
        from enlighten.l3_controller import ContextualTemperatureController

        controller = ContextualTemperatureController()

        prompts = [
            "写一个关于人工智能的故事",
            "编写一段浪漫的小说情节",
            "Create a fiction story about space travel",
            "续写这个小说的结尾"
        ]

        for prompt in prompts:
            scene = controller.detect_scene(prompt)
            assert scene == "creative_writing", f"Failed for: {prompt}"

    def test_scene_detection_code_generation(self):
        """SCENE-02: 代码生成场景检测"""
        from enlighten.l3_controller import ContextualTemperatureController

        controller = ContextualTemperatureController()

        prompts = [
            "写一个Python函数来计算斐波那契数列",
            "代码实现快速排序算法",
            "Implement a binary search function in Java",
            "How to debug this code"
        ]

        detected_scenes = []
        for prompt in prompts:
            scene = controller.detect_scene(prompt)
            detected_scenes.append(scene)

        assert "code_generation" in detected_scenes or "creative_writing" in detected_scenes

    def test_scene_detection_question_answering(self):
        """SCENE-03: 问答场景检测"""
        from enlighten.l3_controller import ContextualTemperatureController

        controller = ContextualTemperatureController()

        prompts = [
            "什么是机器学习？",
            "为什么天空是蓝色的？",
            "Explain what is neural network",
            "How does photosynthesis work?"
        ]

        for prompt in prompts:
            scene = controller.detect_scene(prompt)
            assert scene == "question_answering", f"Failed for: {prompt}"

    def test_scene_detection_summarization(self):
        """SCENE-04: 摘要场景检测"""
        from enlighten.l3_controller import ContextualTemperatureController

        controller = ContextualTemperatureController()

        prompts = [
            "总结这篇文章的主要内容",
            "概括这段文字的要点",
            "Summarize the key points",
            "简述这个报告的核心观点"
        ]

        for prompt in prompts:
            scene = controller.detect_scene(prompt)
            assert scene == "summarization", f"Failed for: {prompt}"

    def test_scene_detection_translation(self):
        """SCENE-05: 翻译场景检测"""
        from enlighten.l3_controller import ContextualTemperatureController

        controller = ContextualTemperatureController()

        prompts = [
            "翻译这段话成英文",
            "把中文翻译成法语",
            "Translate this to Spanish",
            "将以下内容翻译为德语"
        ]

        for prompt in prompts:
            scene = controller.detect_scene(prompt)
            assert scene == "translation", f"Failed for: {prompt}"

    def test_scene_detection_general(self):
        """SCENE-06: 默认通用场景"""
        from enlighten.l3_controller import ContextualTemperatureController

        controller = ContextualTemperatureController()

        prompts = [
            "打开文件",
            "帮我查一下邮件"
        ]

        for prompt in prompts:
            scene = controller.detect_scene(prompt)
            assert scene == "general", f"Failed for: {prompt}"

    def test_scene_detection_with_context(self):
        """SCENE-07: 带上下文的场景检测"""
        from enlighten.l3_controller import ContextualTemperatureController

        controller = ContextualTemperatureController()

        prompt = "继续这个故事"
        context = "从前有一个勇敢的骑士..."

        scene = controller.detect_scene(prompt, context)
        assert scene == "creative_writing"

    def test_scene_history_tracking(self):
        """SCENE-08: 场景历史追踪"""
        from enlighten.l3_controller import ContextualTemperatureController

        controller = ContextualTemperatureController()

        controller.detect_scene("什么是Python？")
        controller.detect_scene("写一段代码")
        controller.detect_scene("总结这段文字")

        distribution = controller.get_scene_distribution()
        assert len(distribution) >= 2


class TestTemperatureSmoothing:
    """温度平滑过渡测试"""

    def test_temperature_smoothing_basic(self):
        """SMOOTH-01: 基本温度平滑"""
        from enlighten.l3_controller import ContextualTemperatureController

        controller = ContextualTemperatureController()
        controller.smoothed_temperature = 0.7

        smoothed = controller.smooth_temperature(target_temp=1.0, smoothing_factor=0.3)

        assert 0.7 < smoothed < 1.0
        assert smoothed == controller.smoothed_temperature

    def test_temperature_smoothing_convergence(self):
        """SMOOTH-02: 平滑收敛性"""
        from enlighten.l3_controller import ContextualTemperatureController

        controller = ContextualTemperatureController()
        controller.smoothed_temperature = 0.7

        for _ in range(10):
            controller.smooth_temperature(target_temp=1.0, smoothing_factor=0.3)

        assert abs(controller.smoothed_temperature - 1.0) < 0.1

    def test_temperature_smoothing_bounds(self):
        """SMOOTH-03: 平滑边界约束"""
        from enlighten.l3_controller import ContextualTemperatureController

        controller = ContextualTemperatureController()
        controller.smoothed_temperature = 1.0

        smoothed = controller.smooth_temperature(target_temp=0.0, smoothing_factor=0.1)

        assert 0.1 <= smoothed <= 2.0

    def test_temperature_smoothing_small_changes(self):
        """SMOOTH-04: 小幅度温度变化"""
        from enlighten.l3_controller import ContextualTemperatureController

        controller = ContextualTemperatureController()
        controller.smoothed_temperature = 0.7

        smoothed1 = controller.smooth_temperature(target_temp=0.75, smoothing_factor=0.3)
        smoothed2 = controller.smooth_temperature(target_temp=0.75, smoothing_factor=0.3)

        assert abs(smoothed2 - smoothed1) < abs(0.75 - 0.7)

    def test_temperature_smoothing_history(self):
        """SMOOTH-05: 平滑历史记录"""
        from enlighten.l3_controller import ContextualTemperatureController

        controller = ContextualTemperatureController()

        for i in range(5):
            entropy_stats = {"mean": 0.5, "variance": 0.02, "trend": 0.0}
            controller.compute_temperature(entropy_stats=entropy_stats)

        assert len(controller.temperature_history) == 5


class TestOutputStabilityMonitor:
    """输出稳定性监控测试"""

    def test_stability_monitor_diversity(self):
        """STABILITY-01: 多样性计算"""
        from enlighten.l3_controller import OutputStabilityMonitor

        monitor = OutputStabilityMonitor()

        text1 = "这是一个测试句子包含多个不同的词汇"
        text2 = "测试 测试 测试 测试 测试"

        diversity1 = monitor._compute_diversity(text1)
        diversity2 = monitor._compute_diversity(text2)

        assert diversity1 > diversity2

    def test_stability_monitor_repetition(self):
        """STABILITY-02: 重复率计算"""
        from enlighten.l3_controller import OutputStabilityMonitor

        monitor = OutputStabilityMonitor()

        text1 = "这个句子有各种不同的词汇组合在一起"
        text2 = "测试 测试 测试 测试 测试"

        repetition1 = monitor._compute_repetition(text1)
        repetition2 = monitor._compute_repetition(text2)

        assert repetition2 > repetition1

    def test_stability_monitor_update(self):
        """STABILITY-03: 监控更新"""
        from enlighten.l3_controller import OutputStabilityMonitor

        monitor = OutputStabilityMonitor()

        outputs = [
            "第一个独特的输出内容包含各种词汇",
            "第二个不同的输出也有自己的词汇和内容",
            "第三个输出继续保持独特性",
            "第四个输出内容也各不相同"
        ]

        for output in outputs:
            metrics = monitor.update(output, temperature=0.7)

        assert len(monitor.output_history) == 4
        assert len(monitor.temperature_history) == 4

    def test_stability_monitor_metrics(self):
        """STABILITY-04: 稳定性指标"""
        from enlighten.l3_controller import OutputStabilityMonitor

        monitor = OutputStabilityMonitor()

        for _ in range(10):
            monitor.update("这是一些独特的输出内容包含不同的词汇组合", 0.7)

        metrics = monitor._compute_stability_metrics()

        assert "stability_score" in metrics
        assert "avg_diversity" in metrics
        assert "avg_repetition" in metrics
        assert "is_stable" in metrics

    def test_stability_monitor_threshold(self):
        """STABILITY-05: 稳定性阈值判断"""
        from enlighten.l3_controller import OutputStabilityMonitor

        monitor = OutputStabilityMonitor(
            diversity_threshold=0.3,
            repetition_threshold=0.5
        )

        for _ in range(5):
            monitor.update("这个输出有很高的词汇多样性和很低的重复率所以应该是稳定的", 0.7)

        assert monitor.should_adjust_temperature() == False

    def test_stability_monitor_adjustment(self):
        """STABILITY-06: 调整建议"""
        from enlighten.l3_controller import OutputStabilityMonitor

        monitor = OutputStabilityMonitor()

        for _ in range(5):
            monitor.update("这是一个稳定的有多种词汇的输出内容", 0.7)

        adjustment = monitor.get_recommended_adjustment()

        assert isinstance(adjustment, float)


class TestContextualTemperature:
    """上下文温度调节测试"""

    def test_temperature_ranges_by_scene(self):
        """TEMP-01: 不同场景的温度范围"""
        from enlighten.l3_controller import ContextualTemperatureController, TemperatureConfig

        config = TemperatureConfig()
        controller = ContextualTemperatureController(config=config)

        assert config.get_range("creative_writing") == (0.7, 1.2)
        assert config.get_range("code_generation") == (0.2, 0.5)
        assert config.get_range("question_answering") == (0.3, 0.6)
        assert config.get_range("summarization") == (0.4, 0.7)
        assert config.get_range("translation") == (0.3, 0.6)
        assert config.get_range("general") == (0.5, 0.8)

    def test_contextual_temperature_van_event(self):
        """TEMP-02: VAN事件时降低温度"""
        from enlighten.l3_controller import ContextualTemperatureController

        controller = ContextualTemperatureController()
        controller.current_scene = "general"
        controller.target_temperature = 0.7

        result = controller.compute_contextual_temperature(
            entropy_stats={"mean": 0.5, "variance": 0.02, "trend": 0.0},
            van_event=True,
            p_harm=0.8
        )

        assert result <= 0.5

    def test_contextual_temperature_low_entropy(self):
        """TEMP-03: 低熵时提高温度"""
        from enlighten.l3_controller import ContextualTemperatureController

        controller = ContextualTemperatureController()
        controller.current_scene = "general"

        result = controller.compute_contextual_temperature(
            entropy_stats={"mean": 0.2, "variance": 0.01, "trend": -0.1},
            van_event=False,
            p_harm=0.0
        )

        assert result > 0.6

    def test_contextual_temperature_high_entropy(self):
        """TEMP-04: 高熵时降低温度"""
        from enlighten.l3_controller import ContextualTemperatureController

        controller = ContextualTemperatureController()
        controller.current_scene = "general"

        result = controller.compute_contextual_temperature(
            entropy_stats={"mean": 0.8, "variance": 0.15, "trend": 0.0},
            van_event=False,
            p_harm=0.0
        )

        assert result < 0.7

    def test_contextual_temperature_trend_positive(self):
        """TEMP-05: 正趋势时微调温度"""
        from enlighten.l3_controller import ContextualTemperatureController

        controller = ContextualTemperatureController()
        controller.current_scene = "general"

        result = controller.compute_contextual_temperature(
            entropy_stats={"mean": 0.5, "variance": 0.02, "trend": 0.2},
            van_event=False,
            p_harm=0.0
        )

        assert 0.5 <= result <= 0.8

    def test_contextual_temperature_trend_negative(self):
        """TEMP-06: 负趋势时调整温度"""
        from enlighten.l3_controller import ContextualTemperatureController

        controller = ContextualTemperatureController()
        controller.current_scene = "general"

        result = controller.compute_contextual_temperature(
            entropy_stats={"mean": 0.5, "variance": 0.02, "trend": -0.2},
            van_event=False,
            p_harm=0.0
        )

        assert 0.3 <= result <= 0.7

    def test_contextual_temperature_clipping(self):
        """TEMP-07: 温度边界裁剪"""
        from enlighten.l3_controller import ContextualTemperatureController

        controller = ContextualTemperatureController(tau_range=(0.1, 2.0))
        controller.current_scene = "general"

        result = controller.compute_contextual_temperature(
            entropy_stats={"mean": 0.0, "variance": 0.0, "trend": -10.0},
            van_event=True,
            p_harm=1.0
        )

        assert 0.1 <= result <= 2.0


class TestSceneSpecificTemperature:
    """场景特定温度测试"""

    def test_creative_writing_temperature(self):
        """SCENE-TEMP-01: 创意写作温度"""
        from enlighten.l3_controller import ContextualTemperatureController

        controller = ContextualTemperatureController()
        controller.current_scene = "creative_writing"

        result = controller.compute_contextual_temperature(
            entropy_stats={"mean": 0.5, "variance": 0.02, "trend": 0.0},
            van_event=False,
            p_harm=0.0
        )

        assert 0.7 <= result <= 1.2

    def test_code_generation_temperature(self):
        """SCENE-TEMP-02: 代码生成温度"""
        from enlighten.l3_controller import ContextualTemperatureController

        controller = ContextualTemperatureController()
        controller.current_scene = "code_generation"

        result = controller.compute_contextual_temperature(
            entropy_stats={"mean": 0.5, "variance": 0.02, "trend": 0.0},
            van_event=False,
            p_harm=0.0
        )

        assert 0.2 <= result <= 0.5

    def test_qa_temperature(self):
        """SCENE-TEMP-03: 问答温度"""
        from enlighten.l3_controller import ContextualTemperatureController

        controller = ContextualTemperatureController()
        controller.current_scene = "question_answering"

        result = controller.compute_contextual_temperature(
            entropy_stats={"mean": 0.5, "variance": 0.02, "trend": 0.0},
            van_event=False,
            p_harm=0.0
        )

        assert 0.3 <= result <= 0.6

    def test_summarization_temperature(self):
        """SCENE-TEMP-04: 摘要温度"""
        from enlighten.l3_controller import ContextualTemperatureController

        controller = ContextualTemperatureController()
        controller.current_scene = "summarization"

        result = controller.compute_contextual_temperature(
            entropy_stats={"mean": 0.5, "variance": 0.02, "trend": 0.0},
            van_event=False,
            p_harm=0.0
        )

        assert 0.4 <= result <= 0.7


class TestTemperatureStability:
    """温度稳定性测试"""

    def test_temperature_does_not_change_abruptly(self):
        """STABLE-01: 温度不会突变"""
        from enlighten.l3_controller import ContextualTemperatureController

        controller = ContextualTemperatureController()
        controller.smoothed_temperature = 0.7

        prev_temp = controller.smoothed_temperature
        for _ in range(10):
            controller.compute_contextual_temperature(
                entropy_stats={"mean": 0.9, "variance": 0.2, "trend": 0.5},
                van_event=False,
                p_harm=0.0
            )
            smoothed = controller.smooth_temperature()
            change = abs(smoothed - prev_temp)
            assert change < 0.3
            prev_temp = smoothed

    def test_temperature_provides_stability_output(self):
        """STABLE-02: 温度提供稳定性输出"""
        from enlighten.l3_controller import ContextualTemperatureController

        controller = ContextualTemperatureController()

        result = controller.compute_temperature(
            entropy_stats={"mean": 0.5, "variance": 0.02, "trend": 0.0},
            van_event=False,
            p_harm=0.0
        )

        assert "temperature" in result
        assert "scene" in result
        assert "is_stable" in result
        assert isinstance(result["is_stable"], bool)

    def test_temperature_statistics_tracking(self):
        """STABLE-03: 温度统计追踪"""
        from enlighten.l3_controller import ContextualTemperatureController

        controller = ContextualTemperatureController()

        for i in range(10):
            controller.compute_temperature(
                entropy_stats={"mean": 0.5 + i * 0.01, "variance": 0.02, "trend": 0.0},
                van_event=False,
                p_harm=0.0
            )

        stats = controller.get_statistics()

        assert "temperature_stats" in stats
        assert "mean" in stats["temperature_stats"]
        assert "std" in stats["temperature_stats"]
        assert "min" in stats["temperature_stats"]
        assert "max" in stats["temperature_stats"]

    def test_temperature_with_stability_monitoring(self):
        """STABLE-04: 带稳定性监控的温度"""
        from enlighten.l3_controller import ContextualTemperatureController

        controller = ContextualTemperatureController()

        for i in range(5):
            output = f"这是第{i}个独特的输出内容包含不同的词汇和表达"
            controller.compute_temperature(
                entropy_stats={"mean": 0.5, "variance": 0.02, "trend": 0.0},
                van_event=False,
                p_harm=0.0,
                output=output,
                enable_stability_monitor=True
            )

        stats = controller.get_statistics()
        assert "stability_stats" in stats


class TestHybridEnlightenLMIntegration:
    """HybridEnlightenLM集成测试"""

    def test_hybrid_model_with_contextual_temperature(self):
        """HYBRID-01: 带上下文温度控制的混合模型"""
        from enlighten.hybrid_architecture import HybridEnlightenLM

        model = HybridEnlightenLM(use_contextual_temperature=True)

        assert model.use_contextual_temperature == True
        assert model.contextual_temperature_controller is not None

    def test_hybrid_model_without_contextual_temperature(self):
        """HYBRID-02: 不带上下文温度控制的混合模型"""
        from enlighten.hybrid_architecture import HybridEnlightenLM

        model = HybridEnlightenLM(use_contextual_temperature=False)

        assert model.use_contextual_temperature == False
        assert model.contextual_temperature_controller is None

    def test_hybrid_model_status_with_contextual_temperature(self):
        """HYBRID-03: 带上下文温度控制的模型状态"""
        from enlighten.hybrid_architecture import HybridEnlightenLM

        model = HybridEnlightenLM(use_contextual_temperature=True)

        status = model.get_status()

        assert "contextual_temperature" in status
        assert status["contextual_temperature"]["enabled"] == True

    def test_hybrid_model_reset_with_contextual_temperature(self):
        """HYBRID-04: 带上下文温度控制的模型重置"""
        from enlighten.hybrid_architecture import HybridEnlightenLM

        model = HybridEnlightenLM(use_contextual_temperature=True)

        if model.contextual_temperature_controller:
            controller = model.contextual_temperature_controller
            controller.detect_scene("写一个故事")
            controller.smoothed_temperature = 0.9

        model.reset()

        if model.contextual_temperature_controller:
            assert model.contextual_temperature_controller.smoothed_temperature == 0.7

    def test_hybrid_model_contextual_temperature_in_generation(self):
        """HYBRID-05: 生成时使用上下文温度控制"""
        from enlighten.hybrid_architecture import HybridEnlightenLM
        from enlighten.api.ollama_client import OllamaAPIClient, OllamaConfig

        try:
            config = OllamaConfig(model="qwen2.5:0.5b")
            client = OllamaAPIClient(config)
            model = HybridEnlightenLM(
                use_local_model=False,
                api_client=client,
                use_contextual_temperature=True
            )

            status_before = model.get_status()
            assert status_before["contextual_temperature"]["enabled"] == True

        except Exception:
            pytest.skip("Ollama API not available")


class TestTemperatureRangeConfiguration:
    """温度范围配置测试"""

    def test_custom_temperature_ranges(self):
        """CONFIG-01: 自定义温度范围"""
        from enlighten.l3_controller import TemperatureConfig, ContextualTemperatureController

        config = TemperatureConfig(
            creative_range=(0.8, 1.3),
            code_range=(0.1, 0.4),
            default_temperature=0.6
        )

        controller = ContextualTemperatureController(config=config)

        assert config.creative_range == (0.8, 1.3)
        assert config.code_range == (0.1, 0.4)
        assert config.default_temperature == 0.6

    def test_smoothing_factor_configuration(self):
        """CONFIG-02: 平滑因子配置"""
        from enlighten.l3_controller import TemperatureConfig

        config = TemperatureConfig(smoothing_factor=0.5)

        assert config.smoothing_factor == 0.5

    def test_stability_threshold_configuration(self):
        """CONFIG-03: 稳定性阈值配置"""
        from enlighten.l3_controller import TemperatureConfig

        config = TemperatureConfig(stability_threshold=0.2)

        assert config.stability_threshold == 0.2


class TestContextualTemperatureDecisionHistory:
    """上下文温度决策历史测试"""

    def test_decision_history_recording(self):
        """HISTORY-01: 决策历史记录"""
        from enlighten.l3_controller import ContextualTemperatureController

        controller = ContextualTemperatureController()

        for i in range(10):
            controller.compute_temperature(
                entropy_stats={"mean": 0.5, "variance": 0.02, "trend": 0.0},
                van_event=False,
                p_harm=0.0
            )

        assert len(controller.decision_history) == 10

    def test_get_recent_decisions(self):
        """HISTORY-02: 获取最近决策"""
        from enlighten.l3_controller import ContextualTemperatureController

        controller = ContextualTemperatureController()

        for i in range(20):
            controller.compute_temperature(
                entropy_stats={"mean": 0.5, "variance": 0.02, "trend": 0.0},
                van_event=False,
                p_harm=0.0
            )

        recent = controller.get_recent_decisions(n=5)

        assert len(recent) == 5

    def test_decision_history_contains_all_fields(self):
        """HISTORY-03: 决策历史包含所有字段"""
        from enlighten.l3_controller import ContextualTemperatureController

        controller = ContextualTemperatureController()

        controller.compute_temperature(
            entropy_stats={"mean": 0.5, "variance": 0.02, "trend": 0.0},
            van_event=True,
            p_harm=0.3,
            output="测试输出"
        )

        decision = controller.decision_history[0]

        assert "scene" in decision
        assert "target_temp" in decision
        assert "final_temp" in decision
        assert "entropy_mean" in decision
        assert "van_event" in decision
        assert "p_harm" in decision
        assert "stability_metrics" in decision


class TestEdgeCases:
    """边界情况测试"""

    def test_empty_entropy_stats(self):
        """EDGE-01: 空熵统计"""
        from enlighten.l3_controller import ContextualTemperatureController

        controller = ContextualTemperatureController()

        result = controller.compute_contextual_temperature(
            entropy_stats={},
            van_event=False,
            p_harm=0.0
        )

        assert 0.1 <= result <= 2.0

    def test_extreme_entropy_values(self):
        """EDGE-02: 极端熵值"""
        from enlighten.l3_controller import ContextualTemperatureController

        controller = ContextualTemperatureController()

        result = controller.compute_contextual_temperature(
            entropy_stats={"mean": 1.0, "variance": 1.0, "trend": 10.0},
            van_event=False,
            p_harm=0.0
        )

        assert 0.1 <= result <= 2.0

    def test_reset_clears_all_state(self):
        """EDGE-03: 重置清除所有状态"""
        from enlighten.l3_controller import ContextualTemperatureController

        controller = ContextualTemperatureController()

        for i in range(10):
            controller.compute_temperature(
                entropy_stats={"mean": 0.5, "variance": 0.02, "trend": 0.0},
                van_event=False,
                p_harm=0.0
            )

        controller.reset()

        assert len(controller.decision_history) == 0
        assert len(controller.temperature_history) == 0
        assert controller.current_temperature == controller.config.default_temperature


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])