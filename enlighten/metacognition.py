"""
MetaCognition - 元认知自检模块
提供自检提示模板和自动追问机制
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import re
import json


@dataclass
class SelfCheckResult:
    """自检结果"""
    is_valid: bool
    confidence: float
    issues: List[str]
    corrections: List[str]
    reasoning: str


@dataclass
class MetaCognitionConfig:
    """元认知配置"""
    enable_self_check: bool = True
    enable_auto_questioning: bool = True
    confidence_threshold: float = 0.7
    max_questions: int = 3
    question_timeout: float = 10.0


class MetaCognition:
    """
    元认知自检模块
    
    功能:
    - 提供自检提示模板
    - 实现自动追问机制
    - 支持自我修正流程
    - 与L3控制器集成
    """

    def __init__(self, config: Optional[MetaCognitionConfig] = None):
        """
        初始化元认知模块

        Args:
            config: 元认知配置
        """
        self.config = config or MetaCognitionConfig()
        self._prompt_templates = self._load_prompt_templates()
        self._question_templates = self._load_question_templates()
        self._correction_templates = self._load_correction_templates()

    def _load_prompt_templates(self) -> Dict[str, str]:
        """
        加载自检提示模板
        """
        return {
            "general": """
你是一个具有元认知能力的AI助手。请对以下内容进行自检：

{content}

请检查：
1. 信息准确性：是否存在事实错误、编造的信息或误导性内容
2. 逻辑一致性：是否存在逻辑矛盾、前后不一致的地方
3. 完整性：是否遗漏重要信息或关键细节
4. 客观性：是否存在偏见、主观臆断或情绪化表达
5. 安全性：是否包含敏感内容、有害信息或违规内容

请提供详细的自检结果，包括发现的问题和改进建议。
            """,
            "factual": """
请检查以下内容的事实准确性：

{content}

具体检查：
- 日期、时间、地点是否正确
- 人名、机构名、事件名称是否准确
- 数据、统计数字是否可靠
- 引用的来源是否真实存在
- 是否存在编造的信息或虚假内容

请标记所有可疑的事实性问题，并提供正确的信息（如果已知）。
            """,
            "logical": """
请检查以下内容的逻辑一致性：

{content}

具体检查：
- 前提与结论之间是否存在逻辑断层
- 是否存在自相矛盾的陈述
- 推理过程是否合理
- 论证是否充分
- 因果关系是否成立

请指出所有逻辑问题，并提供改进建议。
            """,
            "bias": """
请检查以下内容是否存在偏见：

{content}

具体检查：
- 是否存在刻板印象或歧视性语言
- 是否偏向特定群体或观点
- 是否忽略多元视角
- 是否使用情绪化或煽动性语言
- 是否存在选择性引用或数据操纵

请识别所有偏见问题，并提供更加客观中立的表达方式。
            """
        }

    def _load_question_templates(self) -> List[str]:
        """
        加载自动追问模板
        """
        return [
            "你提到的这个信息有什么具体的来源吗？",
            "关于这一点，是否有其他可能的解释？",
            "你能提供更多关于这个观点的证据吗？",
            "这个结论是基于哪些前提条件得出的？",
            "是否考虑了相反的观点或证据？",
            "这个信息在不同情境下是否仍然适用？",
            "你是如何验证这个信息的准确性的？",
            "是否存在可能影响这个结论的其他因素？"
        ]

    def _load_correction_templates(self) -> Dict[str, str]:
        """
        加载自我修正模板
        """
        return {
            "factual_error": """
我注意到上述内容中存在事实性错误：

错误：{error}

更正：{correction}

这一更正基于可靠的信息来源，旨在提供准确的内容。
            """,
            "logical_issue": """
我发现上述内容中存在逻辑问题：

问题：{issue}

修正：{correction}

通过重新梳理逻辑关系，确保论证更加严密和连贯。
            """,
            "bias_issue": """
我意识到上述内容可能存在偏见：

偏见：{bias}

改进：{improvement}

采用更加客观中立的表达方式，尊重多元视角。
            """,
            "incomplete_info": """
我注意到上述内容可能缺少一些重要信息：

缺失：{missing}

补充：{addition}

提供更完整的信息有助于更全面地理解问题。
            """
        }

    def generate_self_check_prompt(self, content: str, check_type: str = "general") -> str:
        """
        生成自检提示

        Args:
            content: 需要检查的内容
            check_type: 检查类型 (general, factual, logical, bias)

        Returns:
            str: 自检提示词
        """
        template = self._prompt_templates.get(check_type, self._prompt_templates["general"])
        return template.format(content=content)

    def generate_auto_questions(self, content: str, num_questions: int = 3) -> List[str]:
        """
        生成自动追问

        Args:
            content: 需要追问的内容
            num_questions: 追问数量

        Returns:
            List[str]: 追问列表
        """
        if not self.config.enable_auto_questioning:
            return []

        # 基于内容特征选择合适的追问
        questions = []
        question_pool = self._question_templates.copy()

        # 分析内容特征
        has_factual_claims = bool(re.search(r'(根据|研究|数据|统计|调查|显示|表明)', content))
        has_opinions = bool(re.search(r'(我认为|觉得|应该|必须|最好)', content))
        has_generalizations = bool(re.search(r'(所有|总是|从不|每个人|没有人)', content))

        # 优先选择相关的追问
        if has_factual_claims:
            questions.append("你提到的这个信息有什么具体的来源吗？")
            question_pool.remove("你提到的这个信息有什么具体的来源吗？")
        if has_opinions:
            questions.append("是否考虑了相反的观点或证据？")
            question_pool.remove("是否考虑了相反的观点或证据？")
        if has_generalizations:
            questions.append("这个结论是基于哪些前提条件得出的？")
            question_pool.remove("这个结论是基于哪些前提条件得出的？")

        # 填充剩余的追问
        while len(questions) < num_questions and question_pool:
            questions.append(question_pool.pop(0))

        return questions[:num_questions]

    def analyze_self_check_response(self, response: str) -> SelfCheckResult:
        """
        分析自检响应

        Args:
            response: 自检响应内容

        Returns:
            SelfCheckResult: 自检结果
        """
        issues = []
        corrections = []
        confidence = 0.7  # 默认信心值

        # 分析响应内容 - 更准确地判断是否存在问题
        has_issues = False
        issue_indicators = ["错误", "问题", "偏见", "不足", "缺陷", "遗漏", "错误的", "不准确", "不完整", "有问题"]
        negation_patterns = ["没有问题", "未发现问题", "未发现错误", "没有发现问题", "没有发现错误", "没有任何问题", "完美的内容"]
        
        # 检查是否有否定模式
        has_negation = False
        for negation in negation_patterns:
            if negation in response:
                has_negation = True
                break
        
        # 如果没有否定模式，再检查问题指标
        if not has_negation:
            for indicator in issue_indicators:
                if indicator in response:
                    has_issues = True
                    break

        if has_issues:
            # 提取问题
            issue_patterns = [
                r'(错误|问题|偏见|不足|缺陷|遗漏):?\s*(.*?)(?=\n|$)',
                r'(发现|注意到|识别|检测)到.*?(错误|问题|偏见|不足|缺陷|遗漏):?\s*(.*?)(?=\n|$)'
            ]
            
            for pattern in issue_patterns:
                matches = re.findall(pattern, response, re.DOTALL)
                for match in matches:
                    if len(match) >= 2:
                        issue_text = match[-1].strip()
                        if issue_text and issue_text != "。":  # 过滤空内容和单独的句号
                            issues.append(issue_text)

            # 提取更正
            correction_patterns = [
                r'(更正|修正|改进|补充|建议):?\s*(.*?)(?=\n|$)',
                r'(应该|建议|推荐).*?:?\s*(.*?)(?=\n|$)'
            ]
            
            for pattern in correction_patterns:
                matches = re.findall(pattern, response, re.DOTALL)
                for match in matches:
                    if len(match) >= 2:
                        correction_text = match[-1].strip()
                        if correction_text and correction_text != "：":  # 过滤空内容和单独的冒号
                            corrections.append(correction_text)

            # 计算信心值 - 进一步降低
            if issues:
                confidence = 1.0 - min(0.5, len(issues) * 0.2)  # 降低信心值
            else:
                confidence = 0.9
        else:
            # 没有发现问题
            confidence = 0.95

        return SelfCheckResult(
            is_valid=confidence > self.config.confidence_threshold,  # 使用 > 而不是 >=
            confidence=confidence,
            issues=issues,
            corrections=corrections,
            reasoning=response
        )

    def generate_correction(self, issue_type: str, issue: str, correction: str) -> str:
        """
        生成修正内容

        Args:
            issue_type: 问题类型 (factual_error, logical_issue, bias_issue, incomplete_info)
            issue: 问题描述
            correction: 修正内容

        Returns:
            str: 修正文本
        """
        template = self._correction_templates.get(issue_type, self._correction_templates["factual_error"])
        return template.format(
            error=issue if issue_type == "factual_error" else issue,
            correction=correction,
            issue=issue if issue_type == "logical_issue" else issue,
            bias=issue if issue_type == "bias_issue" else issue,
            improvement=correction if issue_type == "bias_issue" else correction,
            missing=issue if issue_type == "incomplete_info" else issue,
            addition=correction if issue_type == "incomplete_info" else correction
        )

    def process_content(self, content: str) -> Dict[str, Any]:
        """
        处理内容，执行完整的元认知自检流程

        Args:
            content: 需要处理的内容

        Returns:
            Dict: 处理结果
        """
        if not self.config.enable_self_check:
            return {
                "original_content": content,
                "self_check_performed": False,
                "is_valid": True,
                "confidence": 0.9,
                "issues": [],
                "corrections": [],
                "auto_questions": []
            }

        # 1. 生成自检提示
        self_check_prompt = self.generate_self_check_prompt(content)

        # 2. 这里应该调用模型进行自检
        # 暂时模拟自检响应
        simulated_response = self._simulate_self_check_response(content)

        # 3. 分析自检结果
        self_check_result = self.analyze_self_check_response(simulated_response)

        # 4. 生成自动追问
        auto_questions = []
        if not self_check_result.is_valid and self.config.enable_auto_questioning:
            auto_questions = self.generate_auto_questions(content, self.config.max_questions)
        elif self_check_result.issues and self.config.enable_auto_questioning:
            # 如果有问题但信心值刚好达到阈值，也生成追问
            auto_questions = self.generate_auto_questions(content, self.config.max_questions)

        # 5. 生成修正内容
        corrections = []
        if self_check_result.issues:
            for i, issue in enumerate(self_check_result.issues):
                if i < len(self_check_result.corrections):
                    correction = self_check_result.corrections[i]
                    # 根据问题类型选择修正模板
                    issue_type = "factual_error"  # 默认类型
                    if "逻辑" in issue:
                        issue_type = "logical_issue"
                    elif "偏见" in issue:
                        issue_type = "bias_issue"
                    elif "缺失" in issue or "缺少" in issue:
                        issue_type = "incomplete_info"
                    
                    correction_text = self.generate_correction(issue_type, issue, correction)
                    corrections.append(correction_text)

        return {
            "original_content": content,
            "self_check_performed": True,
            "self_check_prompt": self_check_prompt,
            "self_check_response": simulated_response,
            "is_valid": self_check_result.is_valid,
            "confidence": self_check_result.confidence,
            "issues": self_check_result.issues,
            "corrections": corrections,
            "auto_questions": auto_questions,
            "reasoning": self_check_result.reasoning
        }

    def _simulate_self_check_response(self, content: str) -> str:
        """
        模拟自检响应（实际应用中应该调用真实模型）

        Args:
            content: 需要检查的内容

        Returns:
            str: 模拟的自检响应
        """
        # 简单的模拟逻辑
        if "特朗普" in content or "拜登" in content:
            return """
检查结果：

1. 信息准确性：未发现明显的事实错误，但建议核实具体的日期和数据。
2. 逻辑一致性：论证逻辑基本合理，但存在一些跳跃。
3. 完整性：缺少对相反观点的讨论。
4. 客观性：存在一定的政治偏向，语言略显情绪化。
5. 安全性：未发现敏感内容。

建议：
- 补充对立观点的信息
- 使用更加中立的语言表达
- 提供具体的数据来源
            """
        elif "数据" in content or "统计" in content:
            return """
检查结果：

1. 信息准确性：数据来源未明确，建议提供具体的统计来源。
2. 逻辑一致性：数据与结论之间的关联需要加强。
3. 完整性：缺少对数据局限性的讨论。
4. 客观性：分析较为客观，但可能存在选择性使用数据的情况。
5. 安全性：未发现敏感内容。

建议：
- 明确数据来源和时间范围
- 讨论数据的局限性
- 考虑多种数据分析角度
            """
        else:
            return """
检查结果：

1. 信息准确性：未发现明显的事实错误。
2. 逻辑一致性：论证逻辑清晰，推理合理。
3. 完整性：内容较为完整，涵盖了主要方面。
4. 客观性：表达较为中立，未发现明显偏见。
5. 安全性：未发现敏感内容。

建议：
- 可以考虑提供更多具体的例子来支持观点
- 进一步明确一些概念的定义
            """

    def integrate_with_l3(self, l3_controller: Any) -> None:
        """
        与L3控制器集成

        Args:
            l3_controller: L3控制器实例
        """
        # 这里可以实现与L3控制器的集成
        # 例如：
        # - 向L3控制器注册元认知信号
        # - 接收L3控制器的调控信号
        # - 影响L3的决策过程
        pass


class MetaCognitionFactory:
    """
    元认知模块工厂
    """

    @staticmethod
    def create_meta_cognition(config: Optional[MetaCognitionConfig] = None) -> MetaCognition:
        """
        创建元认知模块实例

        Args:
            config: 元认知配置

        Returns:
            MetaCognition: 元认知模块实例
        """
        return MetaCognition(config)

    @staticmethod
    def create_default_config() -> MetaCognitionConfig:
        """
        创建默认配置

        Returns:
            MetaCognitionConfig: 默认配置
        """
        return MetaCognitionConfig()
