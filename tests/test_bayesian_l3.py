"""
测试贝叶斯L3控制器

测试BayesianL3Controller的核心功能：
1. 贝叶斯更新算法
2. 动态温度调节
3. 连续截断信心计算
4. 与HybridEnlightenLM的集成
"""

import pytest
import numpy as np
from enlighten.l3_controller import BayesianL3Controller, ControlSignals
from enlighten.hybrid_architecture import HybridEnlightenLM


class TestBayesianL3Controller:
    """测试BayesianL3Controller类"""

    def test_initialization(self):
        """测试初始化"""
        controller = BayesianL3Controller()
        assert controller is not None
        posterior = controller.get_posterior()
        assert "normal" in posterior
        assert "noise_injection" in posterior
        assert "bias_injection" in posterior
        assert np.isclose(sum(posterior.values()), 1.0)

    def test_forward_normal_condition(self):
        """测试正常条件下的前向传播"""
        controller = BayesianL3Controller()
        
        # 正常条件的熵统计
        entropy_stats = {
            "mean": 0.6,      # 高均值
            "variance": 0.02,  # 低方差
            "trend": 0.01,     # 正趋势
            "current": 0.5
        }
        
        signals = controller.forward(entropy_stats, van_event=False, p_harm=0.1)
        
        assert isinstance(signals, ControlSignals)
        assert not signals.cutoff
        assert signals.tau > 0.7  # 正常条件下温度应该较高

        # 检查后验概率
        posterior = controller.get_posterior()
        assert posterior["normal"] > 0.5  # 正常条件的后验概率应该最高

    def test_forward_noise_condition(self):
        """测试噪声条件下的前向传播"""
        controller = BayesianL3Controller()
        
        # 噪声条件的熵统计
        entropy_stats = {
            "mean": 0.5,
            "variance": 0.3,  # 高方差
            "trend": -0.1,    # 负趋势
            "current": 0.4
        }
        
        signals = controller.forward(entropy_stats, van_event=False, p_harm=0.2)
        
        assert isinstance(signals, ControlSignals)
        assert not signals.cutoff

        # 检查后验概率
        posterior = controller.get_posterior()
        assert posterior["noise_injection"] > 0.3  # 噪声条件的后验概率应该较高

    def test_forward_bias_condition(self):
        """测试偏见条件下的前向传播"""
        controller = BayesianL3Controller()
        
        # 偏见条件的熵统计
        entropy_stats = {
            "mean": 0.2,      # 更低的均值
            "variance": 0.03,  # 更低的方差
            "trend": -0.1,    # 更明显的负趋势
            "current": 0.1
        }
        
        signals = controller.forward(entropy_stats, van_event=False, p_harm=0.8)  # 更高的风险值
        
        assert isinstance(signals, ControlSignals)

        # 检查后验概率
        posterior = controller.get_posterior()
        print(f"Bias injection posterior: {posterior['bias_injection']}")
        assert posterior["bias_injection"] > 0.3  # 偏见条件的后验概率应该较高

    def test_forward_van_event(self):
        """测试VAN事件下的前向传播"""
        controller = BayesianL3Controller()
        
        entropy_stats = {
            "mean": 0.5,
            "variance": 0.1,
            "trend": 0.0,
            "current": 0.5
        }
        
        signals = controller.forward(entropy_stats, van_event=True, p_harm=0.8)
        
        assert isinstance(signals, ControlSignals)
        assert signals.cutoff  # VAN事件应该触发截断
        assert "VAN event" in (signals.reason or "")

    def test_forward_high_harm(self):
        """测试高风险值下的前向传播"""
        controller = BayesianL3Controller()
        
        entropy_stats = {
            "mean": 0.4,
            "variance": 0.1,
            "trend": 0.0,
            "current": 0.4
        }
        
        signals = controller.forward(entropy_stats, van_event=False, p_harm=0.9)  # 非常高的风险值
        
        assert isinstance(signals, ControlSignals)
        assert signals.cutoff  # 高风险值应该触发截断
        assert "High harm probability" in (signals.reason or "")

    def test_reset(self):
        """测试重置功能"""
        controller = BayesianL3Controller()
        
        # 运行一次前向传播
        entropy_stats = {
            "mean": 0.3,
            "variance": 0.3,
            "trend": -0.1,
            "current": 0.3
        }
        controller.forward(entropy_stats, van_event=False, p_harm=0.5)
        
        # 检查后验概率已经改变
        posterior_before = controller.get_posterior()
        assert not np.isclose(posterior_before["normal"], 0.8)
        
        # 重置
        controller.reset()
        
        # 检查后验概率已重置
        posterior_after = controller.get_posterior()
        assert np.isclose(posterior_after["normal"], 0.8)
        assert np.isclose(posterior_after["noise_injection"], 0.1)
        assert np.isclose(posterior_after["bias_injection"], 0.1)

    def test_get_statistics(self):
        """测试获取统计信息"""
        controller = BayesianL3Controller()
        
        # 运行几次前向传播
        for i in range(5):
            entropy_stats = {
                "mean": 0.5 + i * 0.1,
                "variance": 0.05 + i * 0.01,
                "trend": 0.01 * i,
                "current": 0.5 + i * 0.1
            }
            controller.forward(entropy_stats, van_event=False, p_harm=0.1 + i * 0.1)
        
        stats = controller.get_statistics()
        assert "total_decisions" in stats
        assert stats["total_decisions"] == 5
        assert "posterior" in stats
        assert isinstance(stats["posterior"], dict)


class TestHybridEnlightenLMWithBayesianL3:
    """测试HybridEnlightenLM与贝叶斯L3控制器的集成"""

    def test_initialization_with_bayesian_l3(self):
        """测试使用贝叶斯L3控制器初始化"""
        model = HybridEnlightenLM(use_bayesian_l3=True)
        assert model is not None
        assert model.use_bayesian_l3
        assert model.bayesian_l3 is not None

    def test_initialization_without_bayesian_l3(self):
        """测试不使用贝叶斯L3控制器初始化"""
        model = HybridEnlightenLM(use_bayesian_l3=False)
        assert model is not None
        assert not model.use_bayesian_l3
        assert model.bayesian_l3 is None

    def test_reset_with_bayesian_l3(self):
        """测试重置功能（包含贝叶斯L3控制器）"""
        model = HybridEnlightenLM(use_bayesian_l3=True)
        assert model.bayesian_l3 is not None
        
        # 运行一次前向传播
        # 注意：这里我们不实际调用generate，因为它需要API客户端
        # 但我们可以直接测试reset方法
        model.reset()
        assert model.bayesian_l3 is not None

    def test_get_status_with_bayesian_l3(self):
        """测试获取状态信息（包含贝叶斯L3控制器）"""
        model = HybridEnlightenLM(use_bayesian_l3=True)
        status = model.get_status()
        assert "use_bayesian_l3" in status
        assert status["use_bayesian_l3"]
        assert "bayesian_l3_stats" in status
        assert isinstance(status["bayesian_l3_stats"], dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
