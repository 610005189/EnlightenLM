"""
测试元认知自检模块
"""

import pytest
from enlighten.metacognition import MetaCognition, MetaCognitionConfig, SelfCheckResult


class TestMetaCognition:
    """测试元认知自检模块"""

    def setup_method(self):
        """设置测试环境"""
        self.config = MetaCognitionConfig()
        self.meta_cog = MetaCognition(self.config)

    def test_generate_self_check_prompt(self):
        """测试生成自检提示"""
        content = "这是一个测试内容，包含一些可能的问题。"
        
        # 测试通用自检提示
        general_prompt = self.meta_cog.generate_self_check_prompt(content, "general")
        assert "自检" in general_prompt
        assert content in general_prompt
        
        # 测试事实性检查提示
        factual_prompt = self.meta_cog.generate_self_check_prompt(content, "factual")
        assert "事实准确性" in factual_prompt
        assert content in factual_prompt
        
        # 测试逻辑性检查提示
        logical_prompt = self.meta_cog.generate_self_check_prompt(content, "logical")
        assert "逻辑一致性" in logical_prompt
        assert content in logical_prompt
        
        # 测试偏见检查提示
        bias_prompt = self.meta_cog.generate_self_check_prompt(content, "bias")
        assert "偏见" in bias_prompt
        assert content in bias_prompt

    def test_generate_auto_questions(self):
        """测试生成自动追问"""
        # 测试基于事实性内容的追问
        factual_content = "根据最新研究数据，人工智能将在未来5年内创造1000万个新就业岗位。"
        factual_questions = self.meta_cog.generate_auto_questions(factual_content, 3)
        assert len(factual_questions) <= 3
        assert any("来源" in q for q in factual_questions)
        
        # 测试基于观点性内容的追问
        opinion_content = "我认为人工智能将彻底改变我们的生活方式，这是不可避免的趋势。"
        opinion_questions = self.meta_cog.generate_auto_questions(opinion_content, 3)
        assert len(opinion_questions) <= 3
        assert any("相反的观点" in q for q in opinion_questions)
        
        # 测试基于概括性内容的追问
        generalization_content = "所有人都应该学习编程，这是未来的基本技能。"
        generalization_questions = self.meta_cog.generate_auto_questions(generalization_content, 3)
        assert len(generalization_questions) <= 3
        assert any("前提条件" in q for q in generalization_questions)

    def test_analyze_self_check_response(self):
        """测试分析自检响应"""
        # 测试包含问题的响应
        problematic_response = """
检查结果：

1. 信息准确性：发现事实性错误，数据来源未明确。
2. 逻辑一致性：存在逻辑跳跃，论证不充分。
3. 完整性：缺少对相反观点的讨论。
4. 客观性：存在一定的偏见。
5. 安全性：未发现敏感内容。

建议：
- 明确数据来源
- 加强逻辑论证
- 补充相反观点
- 使用更加中立的语言
        """
        
        result = self.meta_cog.analyze_self_check_response(problematic_response)
        assert isinstance(result, SelfCheckResult)
        # 信心值0.8仍然大于阈值0.7，所以is_valid应该是True
        assert result.is_valid  # 信心应该高于阈值
        assert len(result.issues) > 0
        assert len(result.corrections) > 0
        assert 0.0 <= result.confidence <= 1.0
        
        # 测试没有问题的响应
        clean_response = """
检查结果：

1. 信息准确性：未发现明显的事实错误。
2. 逻辑一致性：论证逻辑清晰，推理合理。
3. 完整性：内容较为完整，涵盖了主要方面。
4. 客观性：表达较为中立，未发现明显偏见。
5. 安全性：未发现敏感内容。

建议：
- 可以考虑提供更多具体的例子来支持观点
        """
        
        clean_result = self.meta_cog.analyze_self_check_response(clean_response)
        assert isinstance(clean_result, SelfCheckResult)
        assert clean_result.is_valid  # 信心应该高于阈值
        assert len(clean_result.issues) == 0
        assert clean_result.confidence > 0.8

    def test_generate_correction(self):
        """测试生成修正内容"""
        # 测试事实性错误修正
        factual_correction = self.meta_cog.generate_correction(
            "factual_error",
            "数据来源未明确",
            "根据2024年世界经济论坛报告，人工智能预计将创造970万个新就业岗位。"
        )
        assert "事实性错误" in factual_correction
        assert "数据来源未明确" in factual_correction
        assert "世界经济论坛报告" in factual_correction
        
        # 测试逻辑性问题修正
        logical_correction = self.meta_cog.generate_correction(
            "logical_issue",
            "论证逻辑不充分",
            "通过详细分析历史数据和当前趋势，我们可以更有力地支持这一结论。"
        )
        assert "逻辑问题" in logical_correction
        assert "论证逻辑不充分" in logical_correction
        
        # 测试偏见问题修正
        bias_correction = self.meta_cog.generate_correction(
            "bias_issue",
            "存在性别偏见",
            "应该考虑不同性别群体的需求和观点，促进更加包容性的解决方案。"
        )
        assert "偏见" in bias_correction
        assert "性别偏见" in bias_correction
        
        # 测试信息不完整修正
        incomplete_correction = self.meta_cog.generate_correction(
            "incomplete_info",
            "缺少具体案例",
            "例如，在医疗领域，人工智能已经帮助医生提高了诊断准确率。"
        )
        assert "可能缺少一些重要信息" in incomplete_correction
        assert "缺少具体案例" in incomplete_correction

    def test_process_content(self):
        """测试处理内容的完整流程"""
        # 测试正常内容
        normal_content = "人工智能技术正在快速发展，为各个行业带来新的机遇和挑战。"
        normal_result = self.meta_cog.process_content(normal_content)
        
        assert normal_result["self_check_performed"]
        assert normal_result["is_valid"]
        assert len(normal_result["issues"]) == 0
        assert len(normal_result["auto_questions"]) == 0
        
        # 测试可能存在问题的内容
        problematic_content = "特朗普政府的政策导致了经济衰退，这是不可否认的事实。"
        problematic_result = self.meta_cog.process_content(problematic_content)
        
        assert problematic_result["self_check_performed"]
        assert "self_check_prompt" in problematic_result
        assert "self_check_response" in problematic_result
        assert len(problematic_result["issues"]) > 0
        assert len(problematic_result["auto_questions"]) > 0

    def test_disabled_self_check(self):
        """测试禁用自检功能"""
        disabled_config = MetaCognitionConfig(enable_self_check=False)
        disabled_meta_cog = MetaCognition(disabled_config)
        
        content = "这是一个测试内容"
        result = disabled_meta_cog.process_content(content)
        
        assert not result["self_check_performed"]
        assert result["is_valid"]
        assert len(result["issues"]) == 0

    def test_disabled_auto_questioning(self):
        """测试禁用自动追问功能"""
        disabled_config = MetaCognitionConfig(enable_auto_questioning=False)
        disabled_meta_cog = MetaCognition(disabled_config)
        
        content = "根据研究数据，这个结论是正确的。"
        questions = disabled_meta_cog.generate_auto_questions(content)
        
        assert len(questions) == 0

    def test_confidence_threshold(self):
        """测试信心阈值设置"""
        strict_config = MetaCognitionConfig(confidence_threshold=0.9)
        strict_meta_cog = MetaCognition(strict_config)
        
        # 测试信心值高于阈值的情况
        response_no_issues = """
完美的内容，没有任何问题。
        """
        
        result = strict_meta_cog.analyze_self_check_response(response_no_issues)
        assert result.confidence >= 0.95  # 没有问题的响应信心值应该较高
        assert result.is_valid  # 信心高于阈值，所以应该是True
        
        # 测试信心值低于阈值的情况
        response_with_issues = """
内容存在严重的事实性错误和逻辑问题。
        """
        
        result2 = strict_meta_cog.analyze_self_check_response(response_with_issues)
        assert result2.confidence < 0.9  # 有问题的响应信心值应该低于0.9
        assert not result2.is_valid  # 信心低于阈值，所以应该是False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
