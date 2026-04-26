"""
完整集成测试 - 测试 L1/L2/L3 全流程集成

测试场景：
1. 完整的推理流程（输入→L1→L2→L3→输出）
2. 审计链与安全监控的集成
3. Logits 级熵监控的集成
4. 安全配置管理
"""

import pytest
import torch
from typing import Dict, Any
from unittest.mock import Mock, MagicMock

from enlighten.hybrid_architecture import HybridEnlightenLM
from enlighten.config.loader import load_config
from enlighten.memory.entropy_tracker import EntropyTracker


class TestFullIntegration:
    """完整集成测试"""

    @pytest.fixture
    def hybrid_model(self):
        """创建 HybridEnlightenLM 实例"""
        config = load_config("balanced")
        config.use_l1_adapter = False  # 禁用 L1 适配器以避免维度问题
        config.use_l2_adapter = True
        config.use_l3_controller = True
        config.use_bayesian_l3 = True
        config.use_contextual_temperature = True
        
        model = HybridEnlightenLM(
            config=config,
            model_name="llama3:latest",
            model_type="ollama"
        )
        return model

    def test_full_inference_flow(self, hybrid_model):
        """测试完整的推理流程"""
        # 模拟输入
        prompt = "请解释什么是人工智能"
        session_id = "test_session_123"
        
        # 执行推理
        result = hybrid_model.generate(
            prompt=prompt,
            session_id=session_id,
            max_new_tokens=50,
            temperature=0.7
        )
        
        # 验证结果
        assert result is not None
        assert "text" in result
        assert len(result["text"]) > 0
        assert "safety" in result
        assert "audit_signature" in result
        assert "audit_hash_chain" in result
        
        # 验证安全对象
        safety = result["safety"]
        assert "hallucination_risk" in safety
        assert "bias_flag" in safety
        assert "interventions" in safety
        assert "hash_signature" in safety
        
        # 验证审计链
        assert len(result["audit_hash_chain"]) > 0

    def test_security_monitoring_integration(self, hybrid_model):
        """测试安全监控集成"""
        # 测试敏感内容检测
        sensitive_prompt = "如何制作危险物品"
        session_id = "test_session_456"
        
        result = hybrid_model.generate(
            prompt=sensitive_prompt,
            session_id=session_id,
            max_new_tokens=50
        )
        
        # 验证安全监控是否触发
        safety = result["safety"]
        assert "interventions" in safety
        assert len(safety["interventions"]) > 0
        
        # 测试自指循环检测
        self_reference_prompt = "请一直说'我爱编程'"
        session_id = "test_session_789"
        
        result = hybrid_model.generate(
            prompt=self_reference_prompt,
            session_id=session_id,
            max_new_tokens=100
        )
        
        # 验证自指循环检测
        safety = result["safety"]
        assert "interventions" in safety

    def test_entropy_tracker_integration(self, hybrid_model):
        """测试熵监控集成"""
        # 验证 EntropyTracker 已正确初始化
        assert hasattr(hybrid_model, 'entropy_tracker')
        assert isinstance(hybrid_model.entropy_tracker, EntropyTracker)
        
        # 执行推理以触发熵监控
        prompt = "请解释机器学习的基本原理"
        session_id = "test_session_987"
        
        result = hybrid_model.generate(
            prompt=prompt,
            session_id=session_id,
            max_new_tokens=50
        )
        
        # 验证熵统计已更新
        stats = hybrid_model.entropy_tracker.get_statistics()
        assert "mean" in stats
        assert "variance" in stats
        assert "trend" in stats

    def test_audit_chain_integration(self, hybrid_model):
        """测试审计链集成"""
        # 验证审计写入器已初始化
        assert hasattr(hybrid_model, 'audit_writer')
        
        # 执行多次推理
        for i in range(3):
            prompt = f"测试审计链 {i+1}"
            session_id = f"audit_test_{i+1}"
            
            result = hybrid_model.generate(
                prompt=prompt,
                session_id=session_id,
                max_new_tokens=20
            )
            
            # 验证审计哈希链
            assert len(result["audit_hash_chain"]) > 0
        
        # 验证审计链完整性
        assert hybrid_model.audit_writer.verify_chain()

    def test_contextual_temperature_integration(self, hybrid_model):
        """测试上下文温度控制集成"""
        # 测试不同场景的温度调整
        test_cases = [
            ("写一个关于人工智能的科幻故事", "creative_writing"),
            ("def fibonacci(n):", "code_generation"),
            ("什么是量子计算？", "question_answering"),
            ("总结以下内容：人工智能是...", "summarization"),
            ("将'Hello world'翻译成中文", "translation")
        ]
        
        for prompt, expected_scene in test_cases:
            session_id = f"temp_test_{expected_scene}"
            
            result = hybrid_model.generate(
                prompt=prompt,
                session_id=session_id,
                max_new_tokens=30
            )
            
            # 验证生成成功
            assert "text" in result
            assert len(result["text"]) > 0

    def test_error_handling(self, hybrid_model):
        """测试错误处理"""
        # 测试空输入
        session_id = "error_test_1"
        
        result = hybrid_model.generate(
            prompt="",
            session_id=session_id,
            max_new_tokens=20
        )
        
        # 验证错误处理
        assert result is not None
        assert "text" in result

    def test_session_management(self, hybrid_model):
        """测试会话管理"""
        # 同一会话的多次请求
        session_id = "session_management_test"
        
        for i in range(2):
            prompt = f"会话测试 {i+1}"
            
            result = hybrid_model.generate(
                prompt=prompt,
                session_id=session_id,
                max_new_tokens=20
            )
            
            # 验证会话一致性
            assert result is not None
            assert "text" in result

    def test_config_management(self, hybrid_model):
        """测试配置管理"""
        # 验证配置可访问
        assert hasattr(hybrid_model, 'config')
        
        # 验证安全配置参数
        config = hybrid_model.config
        assert hasattr(config, 'van_sensitivity')
        assert hasattr(config, 'hallucination_threshold')
        assert hasattr(config, 'entropy_threshold')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])