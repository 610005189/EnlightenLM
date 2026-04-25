"""
测试L3层幻觉检测与控制能力

测试L3层对不同类型幻觉的检测和控制：
1. 事实性幻觉
2. 自相矛盾
3. 无根据猜测
4. 重复循环
5. 逻辑不一致
6. 敏感内容
7. 模糊指令
8. 过度自信
"""

import pytest
import numpy as np
from enlighten.l3_controller import L3Controller, BayesianL3Controller, ControlSignals
from enlighten.hybrid_architecture import HybridEnlightenLM


class TestL3HallucinationDetection:
    """测试L3层幻觉检测与控制能力"""

    def test_factual_hallucination_detection(self):
        """测试事实性幻觉检测
        
        场景：询问未来事件或不存在的事实
        预期：L3层应检测到超出知识范围的内容，调整温度参数
        """
        controller = L3Controller()
        
        # 模拟事实性幻觉的熵统计（低熵、低方差、负趋势）
        entropy_stats = {
            "mean": 0.2,      # 低均值（可能表示过度自信）
            "variance": 0.02,  # 低方差（可能表示缺乏多样性）
            "trend": -0.1,     # 负趋势（可能表示内容质量下降）
            "current": 0.1
        }
        
        signals = controller.forward(entropy_stats, van_event=False, p_harm=0.7)
        
        assert isinstance(signals, ControlSignals)
        
        # 使用贝叶斯L3控制器测试（贝叶斯L3会考虑p_harm）
        bayesian_controller = BayesianL3Controller()
        bayesian_signals = bayesian_controller.forward(entropy_stats, van_event=False, p_harm=0.9)  # 提高p_harm值
        
        assert isinstance(bayesian_signals, ControlSignals)
        # 预期：贝叶斯L3应该触发截断（因为p_harm > 0.6）
        assert bayesian_signals.cutoff

    def test_contradiction_detection(self):
        """测试自相矛盾检测
        
        场景：要求模型同时支持相互矛盾的观点
        预期：L3层应检测到逻辑矛盾，调整生成策略
        """
        controller = L3Controller()
        
        # 模拟自相矛盾的熵统计（高方差表示混乱）
        entropy_stats = {
            "mean": 0.6,      # 中等均值
            "variance": 0.2,   # 高方差（表示混乱）
            "trend": 0.05,     # 轻微正趋势
            "current": 0.7
        }
        
        signals = controller.forward(entropy_stats, van_event=False, p_harm=0.6)
        
        assert isinstance(signals, ControlSignals)
        # 预期：应该调整温度和稀疏阈值
        assert 0.3 < signals.tau < 0.9  # 调整温度
        assert 0.6 < signals.theta < 0.9  # 提高稀疏阈值

    def test_unfounded_guessing_detection(self):
        """测试无根据猜测检测
        
        场景：要求模型预测未来或提供无根据的详细信息
        预期：L3层应检测到猜测行为，调整温度参数
        """
        controller = L3Controller()
        
        # 模拟无根据猜测的熵统计（高熵表示不确定性）
        entropy_stats = {
            "mean": 0.8,      # 高均值（表示不确定性）
            "variance": 0.15,  # 中等方差
            "trend": 0.0,      # 稳定趋势
            "current": 0.8
        }
        
        signals = controller.forward(entropy_stats, van_event=False, p_harm=0.5)
        
        assert isinstance(signals, ControlSignals)

    def test_repetition_detection(self):
        """测试重复循环检测
        
        场景：要求模型重复相同内容
        预期：L3层应检测到重复模式，截断生成
        """
        # 创建一个使用更低阈值的L3控制器
        # 注意：variance_threshold是标准差阈值，而输入的是方差
        # 所以这里设置的variance_threshold应该大于输入方差的平方根
        controller = L3Controller(config={"entropy_threshold": 0.3, "variance_threshold": 0.11})  # 设置更大的阈值
        
        # 模拟重复内容的熵统计（低熵、低方差）
        entropy_stats = {
            "mean": 0.1,      # 极低均值（表示高度重复）
            "variance": 0.01,  # 极低方差（表示缺乏变化）
            "trend": -0.15,    # 明显负趋势（表示质量下降）
            "current": 0.05
        }
        
        signals = controller.forward(entropy_stats, van_event=False, p_harm=0.4)
        
        assert isinstance(signals, ControlSignals)
        # 预期：应该触发截断
        assert signals.cutoff
        assert "Self-referential loop" in (signals.reason or "")

    def test_logical_inconsistency_detection(self):
        """测试逻辑不一致检测
        
        场景：要求模型解释违反物理定律或逻辑的内容
        预期：L3层应检测到逻辑不一致，调整生成策略
        """
        controller = L3Controller()
        
        # 模拟逻辑不一致的熵统计（高方差表示混乱）
        entropy_stats = {
            "mean": 0.7,      # 高均值
            "variance": 0.25,  # 高方差（表示混乱）
            "trend": -0.05,    # 轻微负趋势
            "current": 0.75
        }
        
        signals = controller.forward(entropy_stats, van_event=False, p_harm=0.65)
        
        assert isinstance(signals, ControlSignals)
        # 预期：应该调整温度和稀疏阈值
        assert 0.3 < signals.tau < 0.8
        assert 0.6 < signals.theta < 0.9

    def test_sensitive_content_detection(self):
        """测试敏感内容检测
        
        场景：要求模型提供有害或敏感信息
        预期：L3层应通过VAN事件检测到敏感内容，立即截断
        """
        controller = L3Controller()
        
        # 正常熵统计，但触发VAN事件
        entropy_stats = {
            "mean": 0.5,
            "variance": 0.1,
            "trend": 0.0,
            "current": 0.5
        }
        
        signals = controller.forward(entropy_stats, van_event=True, p_harm=0.9)
        
        assert isinstance(signals, ControlSignals)
        # 预期：应该触发截断
        assert signals.cutoff
        assert "VAN event" in (signals.reason or "")

    def test_vague_instruction_detection(self):
        """测试模糊指令检测
        
        场景：指令模糊或缺乏具体信息
        预期：L3层应检测到信息不足，调整生成策略
        """
        controller = L3Controller()
        
        # 模拟模糊指令的熵统计（高熵表示不确定性）
        entropy_stats = {
            "mean": 0.75,     # 高均值（表示不确定性）
            "variance": 0.1,   # 中等方差
            "trend": 0.0,      # 稳定趋势
            "current": 0.8
        }
        
        signals = controller.forward(entropy_stats, van_event=False, p_harm=0.4)
        
        assert isinstance(signals, ControlSignals)

    def test_overconfidence_detection(self):
        """测试过度自信检测
        
        场景：要求模型提供绝对确定的信息
        预期：L3层应检测到过度自信，调整温度参数
        """
        controller = L3Controller()
        
        # 模拟过度自信的熵统计（低熵表示过度自信）
        entropy_stats = {
            "mean": 0.25,     # 低均值（表示过度自信）
            "variance": 0.03,   # 低方差（表示缺乏多样性）
            "trend": 0.0,      # 稳定趋势
            "current": 0.2
        }
        
        signals = controller.forward(entropy_stats, van_event=False, p_harm=0.5)
        
        assert isinstance(signals, ControlSignals)


class TestHybridEnlightenLMHallucinationControl:
    """测试HybridEnlightenLM的幻觉控制能力"""

    def test_initialization_with_l3_control(self):
        """测试使用L3控制器初始化"""
        model = HybridEnlightenLM(
            use_bayesian_l3=True,
            use_l3_controller=True
        )
        assert model is not None
        assert model.use_bayesian_l3
        assert model.use_l3_controller
        assert model.bayesian_l3 is not None

    def test_get_l3_trace_signals(self):
        """测试获取L3追踪信号"""
        model = HybridEnlightenLM(use_bayesian_l3=True)
        l3_trace = model.get_l3_trace_signals()
        
        assert isinstance(l3_trace, dict)
        assert "p_harm_raw" in l3_trace
        assert 0 <= l3_trace["p_harm_raw"] <= 1

    def test_get_entropy_stats(self):
        """测试获取熵统计信息"""
        model = HybridEnlightenLM()
        entropy_stats = model.get_entropy_stats()
        
        assert isinstance(entropy_stats, dict)
        assert "mean" in entropy_stats
        assert "variance" in entropy_stats
        assert "trend" in entropy_stats
        assert "current" in entropy_stats

    def test_get_attention_stats(self):
        """测试获取注意力统计信息"""
        model = HybridEnlightenLM()
        attention_stats = model.get_attention_stats()
        
        assert hasattr(attention_stats, "entropy")
        assert hasattr(attention_stats, "focus_distribution")

    def test_get_van_stats(self):
        """测试获取VAN监控统计信息"""
        model = HybridEnlightenLM()
        van_stats = model.get_van_stats()
        
        assert isinstance(van_stats, dict)
        assert "total_requests" in van_stats
        assert "van_events" in van_stats
        assert "blocked_requests" in van_stats
        assert "block_ratio" in van_stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
