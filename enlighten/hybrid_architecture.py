"""
Hybrid Architecture - 混合架构
支持本地模型和外部API的统一L1/L2/L3架构

本地模型模式:
- L1: 本地生成层 (Transformer模型)
- L2: 工作记忆层 (上下文+注意力统计)
- L3: VAN监控层 (熵值+自指循环+变异性分析)

API模式:
- L1: API调用层 (DeepSeek API)
- L2: 本地工作记忆层 (上下文+注意力统计)
- L3: 本地VAN监控层 (熵值+自指循环+变异性分析)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Any, List, Tuple, Union
from dataclasses import dataclass
from collections import deque
import math
import re
import hashlib
import time
import os

from .l3_controller import BayesianL3Controller, ControlSignals


@dataclass
class GenerationResult:
    """生成结果"""
    text: str
    tokens: int
    latency: float
    cutoff: bool
    cutoff_reason: Optional[str]
    entropy_stats: Dict[str, float]
    van_event: bool
    security_verified: bool


@dataclass
class AttentionStats:
    """注意力统计"""
    entropy: float
    variance: float
    trend: float
    focus_distribution: List[float]
    stability_score: float


class WorkingMemoryManager:
    """
    L2 工作记忆管理器

    功能:
    - 维护会话历史
    - 计算注意力统计
    - 管理上下文窗口
    - 追踪熵值变化
    """

    def __init__(
        self,
        max_history: int = 100,
        context_window: int = 4096,
        entropy_window: int = 20,
        attention_size: int = 32
    ):
        self.max_history = max_history
        self.context_window = context_window
        self.entropy_window = entropy_window
        self.attention_size = attention_size

        self.conversation_history: List[Dict[str, str]] = []
        self.attention_history: deque = deque(maxlen=entropy_window)
        self.entropy_history: deque = deque(maxlen=entropy_window)
        self.token_count = 0

    def add_turn(self, role: str, content: str) -> None:
        """添加一轮对话"""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": time.time()
        })
        self.token_count += len(content.split())

        if len(self.conversation_history) > self.max_history:
            self.conversation_history.pop(0)

    def get_context(self) -> str:
        """获取当前上下文"""
        context_parts = []
        for turn in self.conversation_history[-self.context_window:]:
            role = turn["role"]
            content = turn["content"]
            context_parts.append(f"{role}: {content}")
        return "\n".join(context_parts)

    def compute_attention_stats(self) -> AttentionStats:
        """计算注意力统计"""
        if not self.attention_history:
            return AttentionStats(
                entropy=1.0,
                variance=0.1,
                trend=0.0,
                focus_distribution=[1.0],
                stability_score=1.0
            )

        attention_data = np.array(list(self.attention_history))
        if attention_data.ndim == 1:
            attention_data = attention_data.reshape(1, -1)

        if attention_data.shape[0] == 0:
            return AttentionStats(
                entropy=1.0,
                variance=0.1,
                trend=0.0,
                focus_distribution=[1.0],
                stability_score=1.0
            )

        entropy_values = -attention_data * np.log2(attention_data + 1e-10)
        entropy_mean = np.mean(entropy_values)
        entropy_variance = np.var(entropy_values)

        mean_entropy_per_step = np.mean(entropy_values, axis=1)
        trend = 0.0
        if len(mean_entropy_per_step) >= 5:
            x = np.arange(len(mean_entropy_per_step))
            result = np.polyfit(x, mean_entropy_per_step, 1)
            if np.ndim(result) == 1:
                trend = float(result[0])
            else:
                trend = float(result)

        row_sums = attention_data.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1e-10, row_sums)
        focus_dist = attention_data / row_sums
        focus_dist = focus_dist[-1] if len(focus_dist) > 0 else np.array([1.0])

        stability = 1.0 - min(float(entropy_variance) * 10, 1.0)

        return AttentionStats(
            entropy=float(entropy_mean),
            variance=float(entropy_variance),
            trend=trend,
            focus_distribution=focus_dist.tolist() if isinstance(focus_dist, np.ndarray) else [1.0],
            stability_score=float(stability)
        )

    def compute_entropy_stats(self) -> Dict[str, float]:
        """计算熵统计"""
        if not self.entropy_history:
            return {
                "mean": 1.0,
                "variance": 0.1,
                "trend": 0.0,
                "current": 1.0,
                "stability": 1.0
            }

        entropy_data = np.array(list(self.entropy_history))
        mean = np.mean(entropy_data)
        variance = np.var(entropy_data)

        trend = 0.0
        if len(entropy_data) >= 5:
            x = np.arange(len(entropy_data))
            slope, _ = np.polyfit(x, entropy_data, 1)
            trend = slope

        stability = 1.0 - min(variance * 10, 1.0)

        return {
            "mean": float(mean),
            "variance": float(variance),
            "trend": float(trend),
            "current": float(entropy_data[-1]) if len(entropy_data) > 0 else 1.0,
            "stability": float(stability)
        }

    def update_attention(self, attention_weights, target_size: Optional[int] = None) -> None:
        """更新注意力权重"""
        target_size = target_size or self.attention_size

        if isinstance(attention_weights, np.ndarray):
            if len(attention_weights.shape) == 1:
                attention_weights = np.expand_dims(attention_weights, axis=0)
            avg_attention = attention_weights.mean(axis=0 if len(attention_weights.shape) == 2 else -1)
        elif isinstance(attention_weights, torch.Tensor):
            if len(attention_weights.shape) == 1:
                attention_weights = attention_weights.unsqueeze(0)
            avg_attention = attention_weights.mean(dim=0 if len(attention_weights.shape) == 2 else -2)
            if isinstance(avg_attention, torch.Tensor):
                avg_attention = avg_attention.cpu().numpy()
        else:
            raise TypeError(f"attention_weights must be numpy array or torch tensor, got {type(attention_weights)}")

        seq_len = len(avg_attention)
        if seq_len == 0:
            avg_attention = np.ones(target_size) / target_size
        elif seq_len != target_size:
            if seq_len > target_size:
                step = seq_len / target_size
                compressed = np.zeros(target_size)
                for i in range(target_size):
                    start = int(i * step)
                    end = int((i + 1) * step)
                    if start < seq_len:
                        compressed[i] = avg_attention[start:min(end, seq_len)].mean()
                avg_attention = compressed
            else:
                padded = np.ones(target_size) * (1.0 / target_size)
                padded[:seq_len] = avg_attention
                padded = padded / padded.sum()
                avg_attention = padded

        if len(avg_attention) != target_size:
            avg_attention = np.ones(target_size) / target_size

        self.attention_history.append(avg_attention)

        entropy = -np.sum(avg_attention * np.log2(avg_attention + 1e-10))
        max_entropy = np.log2(target_size)
        normalized_entropy = entropy / (max_entropy + 1e-10)
        self.entropy_history.append(normalized_entropy)

    def reset(self) -> None:
        """重置工作记忆"""
        self.conversation_history.clear()
        self.attention_history.clear()
        self.entropy_history.clear()
        self.token_count = 0


class VANMonitor:
    """
    L3 VAN 变异性-吸引子网络监控器

    功能:
    - 熵值监控: 追踪注意力熵的滑动统计
    - 自指循环检测: 检测重复模式和自参照
    - 变异性分析: 监控输出变异性
    - 截断决策: 基于多维度判断是否截断
    """

    DEFAULT_SENSITIVE_PATTERNS = [
        r"(hack|crack|bypass|exploit|漏洞|破解|入侵|攻击)",
        r"(password|credential|secret|密钥|密码凭据)",
        r"(malware|virus|ransomware|恶意软件|病毒)",
        r"(fraud|scam|phishing|诈骗|钓鱼|欺诈)",
        r"(illegal|criminal|illicit|非法|犯罪)",
        r"(self[self]+|自指|循环引用)",
    ]

    SELF_LOOP_PATTERNS = [
        r"(.+)\1{3,}",
        r"(.{10,})\1{2,}",
        r"(.{3,}?)(.{3,}?)\1\2",
    ]

    def __init__(
        self,
        van_threshold: float = 0.7,
        entropy_threshold: float = 0.3,
        variance_threshold: float = 0.05,
        cooldown_steps: int = 5,
        enabled: bool = True
    ):
        self.van_threshold = van_threshold
        self.entropy_threshold = entropy_threshold
        self.variance_threshold = variance_threshold
        self.cooldown_steps = cooldown_steps
        self.enabled = enabled

        self.sensitive_patterns = [re.compile(p, re.IGNORECASE) for p in self.DEFAULT_SENSITIVE_PATTERNS]
        self.self_loop_patterns = [re.compile(p) for p in self.SELF_LOOP_PATTERNS]

        self.cooldown_counter = 0
        self.output_history: deque = deque(maxlen=100)
        self.decision_history: List[Dict] = []
        self.van_event_count = 0
        self.total_requests = 0
        self.blocked_count = 0

    def check_input(self, text: str) -> Tuple[bool, Optional[str], float]:
        """
        检查输入内容

        Returns:
            (should_block, reason, risk_score)
        """
        if not self.enabled:
            return False, None, 0.0

        self.total_requests += 1

        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return False, None, 0.0

        risk_score = 0.0
        reasons = []

        for pattern in self.sensitive_patterns:
            if pattern.search(text):
                risk_score += 0.4
                reasons.append("sensitive_keyword")

        for pattern in self.self_loop_patterns:
            if pattern.search(text):
                risk_score += 0.3
                reasons.append("self_referential_pattern")

        char_set_ratio = len(set(text)) / max(len(text), 1)
        if char_set_ratio < 0.1 and len(text) > 20:
            risk_score += 0.2
            reasons.append("low_entropy")

        if risk_score >= self.van_threshold:
            self.blocked_count += 1
            self.van_event_count += 1
            self.cooldown_counter = self.cooldown_steps
            reason = f"VAN event: {', '.join(reasons)}"
            self._record_decision("input_blocked", risk_score, True, reason)
            return True, reason, risk_score

        self._record_decision("input_passed", risk_score, False, None)
        return False, None, risk_score

    def check_output(self, text: str, entropy_stats: Optional[Dict[str, float]] = None) -> Tuple[bool, Optional[str], float]:
        """
        检查输出内容

        Returns:
            (should_cutoff, reason, risk_score)
        """
        if not self.enabled:
            return False, None, 0.0

        self.total_requests += 1

        if self.cooldown_counter > 0:
            return True, "Cooldown active", 1.0

        if not text or len(text.strip()) == 0:
            self._record_decision("output_empty", 1.0, True, "Empty output")
            return True, "Empty output", 1.0

        risk_score = 0.0
        reasons = []

        for pattern in self.sensitive_patterns:
            matches = pattern.findall(text)
            if matches:
                risk_score += 0.3 * min(len(matches), 3)
                reasons.append("sensitive_content")

        repetitive_chars = len(re.findall(r'(.)\1{5,}', text))
        if repetitive_chars > 3:
            risk_score += 0.3
            reasons.append("repetitive_pattern")

        word_repetition = self._detect_word_repetition(text)
        if word_repetition > 0.5:
            risk_score += 0.3
            reasons.append("high_word_repetition")

        output_hash = hashlib.md5(text.encode()).hexdigest()[:16]
        self.output_history.append({
            "hash": output_hash,
            "length": len(text),
            "risk": risk_score,
            "timestamp": time.time()
        })

        if len(self.output_history) >= 10:
            recent = list(self.output_history)[-10:]
            hashes = [h["hash"] for h in recent]
            unique_ratio = len(set(hashes)) / len(hashes)
            if unique_ratio < 0.3:
                risk_score += 0.2
                reasons.append("low_variability")

        text_entropy = self._compute_text_entropy(text)
        if entropy_stats is None:
            entropy_stats = text_entropy

        if entropy_stats.get("entropy", 1.0) < self.entropy_threshold:
            risk_score += 0.25
            reasons.append("low_text_entropy")
        if entropy_stats.get("repetition", 0) > 0.5:
            risk_score += 0.2
            reasons.append("high_repetition")
        if entropy_stats.get("diversity", 1.0) < 0.3:
            risk_score += 0.15
            reasons.append("low_diversity")

        if risk_score >= self.van_threshold:
            self.blocked_count += 1
            self.van_event_count += 1
            self.cooldown_counter = self.cooldown_steps
            reason = f"VAN cutoff: {', '.join(reasons)}"
            self._record_decision("output_cutoff", risk_score, True, reason)
            return True, reason, risk_score

        self._record_decision("output_passed", risk_score, False, None)
        return False, None, risk_score

    def _detect_word_repetition(self, text: str) -> float:
        """检测词汇重复率"""
        tokens = text.split()
        if len(tokens) < 2:
            return 0.0
        unique_tokens = len(set(tokens))
        return 1.0 - (unique_tokens / len(tokens))

    def _compute_text_entropy(self, text: str) -> Dict[str, float]:
        """
        从文本计算统计熵值（用于输出检查）

        分析文本本身的结构特征：
        - 词汇多样性（Type-Token Ratio）
        - 重复率
        - 字符级熵
        """
        tokens = text.split()
        n = len(tokens)

        if n == 0:
            return {"entropy": 0.5, "diversity": 0.5, "repetition": 0.0}

        unique_tokens = len(set(tokens))
        diversity = unique_tokens / max(n, 1)

        repetition_score = 0.0
        if n > 1:
            bigrams = [f"{tokens[i]} {tokens[i+1]}" for i in range(n-1)]
            if bigrams:
                unique_bigrams = len(set(bigrams))
                repetition_score = 1.0 - (unique_bigrams / len(bigrams))

        char_set = set(text)
        char_entropy = len(char_set) / max(len(text), 1)

        combined_entropy = (diversity * 0.4 + (1 - repetition_score) * 0.3 + char_entropy * 0.3)

        return {
            "entropy": float(combined_entropy),
            "diversity": float(diversity),
            "repetition": float(repetition_score),
            "char_entropy": float(char_entropy)
        }

    def _record_decision(
        self,
        decision_type: str,
        risk_score: float,
        blocked: bool,
        reason: Optional[str]
    ) -> None:
        """记录决策历史"""
        record = {
            "type": decision_type,
            "risk": risk_score,
            "blocked": blocked,
            "reason": reason,
            "timestamp": time.time(),
            "cooldown_remaining": self.cooldown_counter
        }
        self.decision_history.append(record)
        if len(self.decision_history) > 1000:
            self.decision_history.pop(0)

    def should_cutoff_by_entropy(self, entropy_stats: Dict[str, float]) -> Tuple[bool, Optional[str]]:
        """
        基于熵统计判断是否截断

        条件:
        - 低注意力熵 (mean < entropy_threshold)
        - 低方差 (variance < variance_threshold)
        - 持续下降趋势 (trend < 0)
        """
        mean = entropy_stats.get("mean", 1.0)
        variance = entropy_stats.get("variance", 0.1)
        trend = entropy_stats.get("trend", 0.0)

        if mean < self.entropy_threshold and variance < self.variance_threshold and trend < 0:
            return True, "Entropy cutoff: low mean, low variance, negative trend"

        return False, None

    def get_statistics(self) -> Dict[str, Any]:
        """获取VAN监控统计"""
        return {
            "total_requests": self.total_requests,
            "van_events": self.van_event_count,
            "blocked_requests": self.blocked_count,
            "cooldown_remaining": self.cooldown_counter,
            "cooldown_active": self.cooldown_counter > 0,
            "block_ratio": self.blocked_count / max(self.total_requests, 1)
        }

    def reset(self) -> None:
        """重置VAN监控状态"""
        self.cooldown_counter = 0
        self.output_history.clear()
        self.decision_history.clear()
        self.van_event_count = 0
        self.total_requests = 0
        self.blocked_count = 0


class HybridEnlightenLM:
    """
    混合架构的 EnlightenLM

    支持本地模型和API调用，统一经过L2工作记忆和L3 VAN监控

    使用方式:
        # API模式 (默认)
        model = HybridEnlightenLM(use_local=False)
        result = model.generate("Hello")

        # 本地模型模式
        model = HybridEnlightenLM(use_local=True, local_model_name="distilgpt2")
        result = model.generate("Hello")
    """

    def __init__(
        self,
        use_local_model: bool = False,
        local_model_name: str = "distilgpt2",
        api_client=None,
        config: Optional[Union[Dict, Any]] = None,
        use_bayesian_l3: bool = False
    ):
        if config is None:
            config_dict = {}
        elif hasattr(config, 'working_memory'):
            config_dict = {
                "max_history": config.working_memory.capacity,
                "context_window": 4096,
                "entropy_window": config.entropy_monitor.window_size,
                "attention_size": getattr(config.working_memory, 'attention_size', 32),
                "van_threshold": config.cutoff.van_threshold,
                "entropy_threshold": config.cutoff.low_entropy_threshold,
                "variance_threshold": config.cutoff.low_variance_threshold,
                "cooldown_steps": config.cutoff.cooldown_steps,
                "van_enabled": True
            }
        else:
            config_dict = config

        self.use_local_model = use_local_model
        self.local_model_name = local_model_name
        self.api_client = api_client
        self.use_bayesian_l3 = use_bayesian_l3

        self.working_memory = WorkingMemoryManager(
            max_history=config_dict.get("max_history", 100),
            context_window=config_dict.get("context_window", 4096),
            entropy_window=config_dict.get("entropy_window", 20),
            attention_size=config_dict.get("attention_size", 32)
        )

        self.van_monitor = VANMonitor(
            van_threshold=config_dict.get("van_threshold", 0.7),
            entropy_threshold=config_dict.get("entropy_threshold", 0.3),
            variance_threshold=config_dict.get("variance_threshold", 0.05),
            cooldown_steps=config_dict.get("cooldown_steps", 5),
            enabled=config_dict.get("van_enabled", True)
        )

        # 初始化贝叶斯L3控制器
        self.bayesian_l3 = None
        if self.use_bayesian_l3:
            self.bayesian_l3 = BayesianL3Controller()

        self.local_model = None
        self.local_tokenizer = None

        if self.use_local_model:
            self._load_local_model()

    def _load_local_model(self) -> None:
        """加载本地模型"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger = __import__('logging').getLogger(__name__)
            logger.info(f"Loading local model: {self.local_model_name}")

            self.local_tokenizer = AutoTokenizer.from_pretrained(self.local_model_name)
            self.local_model = AutoModelForCausalLM.from_pretrained(self.local_model_name)

            if torch.cuda.is_available():
                self.local_model = self.local_model.cuda()

            logger.info("Local model loaded successfully")

        except Exception as e:
            logger = __import__('logging').getLogger(__name__)
            logger.error(f"Failed to load local model: {e}")
            raise

    def generate(
        self,
        prompt: str,
        max_length: int = 1024,
        temperature: float = 0.7,
        enable_trace: bool = False,
        trace_callback: Optional[Any] = None,
        **kwargs
    ) -> GenerationResult:
        """
        生成文本

        Args:
            prompt: 输入提示
            max_length: 最大生成长度
            temperature: 生成温度
            enable_trace: 是否启用Trace记录
            trace_callback: Trace记录回调函数，签名为 callback(mu_H, sigma_H2, k_H, p_harm_raw)

        Returns:
            GenerationResult: 生成结果
        """
        start_time = time.time()

        should_block, block_reason, input_risk = self.van_monitor.check_input(prompt)

        if should_block:
            return GenerationResult(
                text="[内容被安全监控系统拦截]",
                tokens=0,
                latency=time.time() - start_time,
                cutoff=True,
                cutoff_reason=block_reason,
                entropy_stats=self.working_memory.compute_entropy_stats(),
                van_event=True,
                security_verified=False
            )

        self.working_memory.add_turn("user", prompt)

        context = self.working_memory.get_context()

        if self.use_local_model:
            output_text, tokens = self._generate_local(context, max_length, temperature)
        else:
            output_text, tokens = self._generate_api(context, max_length)

        self.working_memory.add_turn("assistant", output_text)

        entropy_stats = self.working_memory.compute_entropy_stats()

        # 使用贝叶斯L3控制器进行决策
        if self.use_bayesian_l3 and self.bayesian_l3:
            # 检查VAN事件
            van_event, van_reason, van_risk = self.van_monitor.check_output(output_text, entropy_stats)
            
            # 获取贝叶斯L3控制器的决策
            control_signals = self.bayesian_l3.forward(
                entropy_stats=entropy_stats,
                van_event=van_event,
                p_harm=van_risk
            )
            
            if control_signals.cutoff:
                return GenerationResult(
                    text="[内容被贝叶斯L3监控系统截断 - " + (control_signals.reason or "高风险内容") + "]",
                    tokens=tokens,
                    latency=time.time() - start_time,
                    cutoff=True,
                    cutoff_reason=control_signals.reason,
                    entropy_stats=entropy_stats,
                    van_event=van_event,
                    security_verified=False
                )
        else:
            # 使用传统VAN监控
            should_cutoff, cutoff_reason, output_risk = self.van_monitor.check_output(output_text, entropy_stats)

            if should_cutoff:
                return GenerationResult(
                    text="[内容被VAN监控系统截断 - " + cutoff_reason + "]",
                    tokens=tokens,
                    latency=time.time() - start_time,
                    cutoff=True,
                    cutoff_reason=cutoff_reason,
                    entropy_stats=entropy_stats,
                    van_event=True,
                    security_verified=False
                )

            entropy_cutoff, entropy_reason = self.van_monitor.should_cutoff_by_entropy(entropy_stats)
            if entropy_cutoff:
                return GenerationResult(
                    text="[内容因熵值异常被截断 - " + entropy_reason + "]",
                    tokens=tokens,
                    latency=time.time() - start_time,
                    cutoff=True,
                    cutoff_reason=entropy_reason,
                    entropy_stats=entropy_stats,
                    van_event=True,
                    security_verified=False
                )

        if enable_trace and trace_callback is not None:
            signals = self.get_l3_trace_signals()
            trace_callback(
                mu_H=signals["mu_H"],
                sigma_H2=signals["sigma_H2"],
                k_H=signals["k_H"],
                p_harm_raw=signals["p_harm_raw"]
            )

        return GenerationResult(
            text=output_text,
            tokens=tokens,
            latency=time.time() - start_time,
            cutoff=False,
            cutoff_reason=None,
            entropy_stats=entropy_stats,
            van_event=False,
            security_verified=True
        )

    def _generate_local(
        self,
        context: str,
        max_length: int,
        temperature: float
    ) -> Tuple[str, int]:
        """使用本地模型生成"""
        if self.local_model is None or self.local_tokenizer is None:
            raise RuntimeError("Local model not loaded")

        inputs = self.local_tokenizer(context, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        attention_mask = inputs.get("attention_mask")
        outputs = self.local_model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            pad_token_id=self.local_tokenizer.pad_token_id or 0
        )

        generated_text = self.local_tokenizer.decode(outputs[0], skip_special_tokens=True)

        prompt_length = len(context.split())
        generated_tokens = len(generated_text.split()) - prompt_length

        attention_weights = outputs[0] if hasattr(outputs[0], 'shape') else None
        if attention_weights is not None:
            if isinstance(attention_weights, torch.Tensor):
                attention_weights = attention_weights.float().cpu().numpy()
            self.working_memory.update_attention(attention_weights)

        return generated_text, max(generated_tokens, 0)

    def _generate_api(
        self,
        context: str,
        max_length: int
    ) -> Tuple[str, int]:
        """使用API生成"""
        if self.api_client is None:
            raise RuntimeError("API client not configured")

        response_text, latency = self.api_client.generate(
            prompt=context,
            max_tokens=max_length
        )

        tokens = len(response_text.split())

        attention_from_text = self._compute_attention_from_text(response_text)
        self.working_memory.update_attention(attention_from_text)

        return response_text, tokens

    def _compute_attention_from_text(self, text: str) -> np.ndarray:
        """
        从文本特征计算"注意力权重"

        这是API模式下的近似计算，不依赖模型内部注意力，
        而是从文本的统计特征推断"认知聚焦程度"

        考虑因素:
        - token级别的条件概率多样性（近似困惑度）
        - 关键词/概念的出现集中度
        - 句子级别的语义连贯性

        Returns:
            固定长度(32)的注意力权重向量
        """
        FIXED_SIZE = 32
        tokens = text.split()
        n = len(tokens)

        if n == 0:
            return np.ones(FIXED_SIZE) / FIXED_SIZE

        focus_scores = np.ones(min(n, FIXED_SIZE))

        content_words = {'的', '是', '在', '有', '我', '你', '他', '她', '它', '和', '与', '或', '但', '而'}
        for i in range(min(n, FIXED_SIZE)):
            token = tokens[i]
            if token in content_words or len(token) <= 1:
                focus_scores[i] = 0.3
            elif len(token) >= 3:
                focus_scores[i] = 1.5

        if focus_scores.sum() > 0:
            focus_distribution = focus_scores / focus_scores.sum()
        else:
            focus_distribution = np.ones(FIXED_SIZE) / FIXED_SIZE

        return focus_distribution

    def _compute_text_entropy(self, text: str) -> Dict[str, float]:
        """
        从文本计算真实的统计熵值

        不依赖模型内部状态，而是分析文本本身的结构特征：
        - 词汇多样性（Type-Token Ratio）
        - 重复率
        - 句子长度变化
        - 字符级熵

        Returns:
            文本熵统计
        """
        tokens = text.split()
        n = len(tokens)

        if n == 0:
            return {"entropy": 1.0, "diversity": 0.5, "repetition": 0.0}

        unique_tokens = len(set(tokens))
        diversity = unique_tokens / max(n, 1)

        repetition_score = 0.0
        if n > 1:
            bigrams = [f"{tokens[i]} {tokens[i+1]}" for i in range(n-1)]
            unique_bigrams = len(set(bigrams))
            repetition_score = 1.0 - (unique_bigrams / max(len(bigrams), 1))

        char_set = set(text)
        if len(text) > 0:
            char_entropy = len(char_set) / len(text)
        else:
            char_entropy = 0.0

        sentence_lengths = [len(s.split()) for s in text.split('。') if s.strip()]
        if len(sentence_lengths) > 1:
            length_variance = np.var(sentence_lengths) if sentence_lengths else 0.0
        else:
            length_variance = 0.0

        combined_entropy = (diversity * 0.4 + (1 - repetition_score) * 0.3 + char_entropy * 0.3)

        return {
            "entropy": float(combined_entropy),
            "diversity": float(diversity),
            "repetition": float(repetition_score),
            "length_variance": float(length_variance),
            "char_entropy": float(char_entropy)
        }

    def get_attention_stats(self) -> AttentionStats:
        """获取注意力统计"""
        return self.working_memory.compute_attention_stats()

    def get_entropy_stats(self) -> Dict[str, float]:
        """获取熵统计"""
        return self.working_memory.compute_entropy_stats()

    def get_van_stats(self) -> Dict[str, Any]:
        """获取VAN监控统计"""
        return self.van_monitor.get_statistics()

    def reset(self) -> None:
        """重置所有状态"""
        self.working_memory.reset()
        self.van_monitor.reset()
        if self.bayesian_l3:
            self.bayesian_l3.reset()

    def get_status(self) -> Dict[str, Any]:
        """获取状态信息"""
        status = {
            "mode": "local" if self.use_local_model else "api",
            "model": self.local_model_name if self.use_local_model else "deepseek",
            "working_memory_tokens": self.working_memory.token_count,
            "conversation_turns": len(self.working_memory.conversation_history),
            "attention_stats": {
                "entropy": self.working_memory.compute_attention_stats().entropy,
                "stability": self.working_memory.compute_attention_stats().stability_score
            },
            "van_stats": self.van_monitor.get_statistics(),
            "use_bayesian_l3": self.use_bayesian_l3
        }
        
        if self.bayesian_l3:
            status["bayesian_l3_stats"] = self.bayesian_l3.get_statistics()
        
        return status

    def get_l3_trace_signals(self) -> Dict[str, float]:
        """
        获取L3贝叶斯控制器的输入信号

        对应论文内部信号:
        - mu_H (mean): 后验熵均值 -> entropy_stats['mean']
        - sigma_H2 (variance): 后验熵方差 -> entropy_stats['variance']
        - k_H (trend): 熵变化趋势 -> entropy_stats['trend']
        - p_harm_raw: VAN风险值 -> van_monitor 最新风险

        Returns:
            Dict containing mu_H, sigma_H2, k_H, p_harm_raw
        """
        entropy_stats = self.working_memory.compute_entropy_stats()

        last_risk = 0.0
        if self.van_monitor.decision_history:
            last_decision = self.van_monitor.decision_history[-1]
            last_risk = last_decision.get("risk_score", 0.0)

        return {
            "mu_H": entropy_stats["mean"],
            "sigma_H2": entropy_stats["variance"],
            "k_H": entropy_stats["trend"],
            "p_harm_raw": last_risk
        }