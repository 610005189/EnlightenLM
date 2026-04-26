"""
Entropy Tracker - 注意力熵统计追踪器
实时计算滑动窗口内的注意力熵统计
"""

import torch
import numpy as np
from collections import deque
from typing import Dict, Optional


class EntropyTracker:
    """
    注意力熵统计追踪器

    计算滑动窗口内的:
    - 均值 μ_H
    - 方差 σ_H²
    - 趋势 k_H (线性回归斜率)

    用于检测自指循环和注意力异常
    """

    def __init__(
        self,
        window_size: int = 100,
        compute_interval: int = 1,
        ema_decay: float = 0.99
    ):
        self.window_size = window_size
        self.compute_interval = compute_interval
        self.ema_decay = ema_decay

        self.attention_history = deque(maxlen=window_size)
        self.logits_history = deque(maxlen=window_size)
        self.hidden_history = deque(maxlen=window_size)
        self.step_count = 0

        self.ema_attention_entropy = 0.0
        self.ema_logits_entropy = 0.0

    def update_attention(self, attention_weights: torch.Tensor) -> None:
        """
        计算并记录当前时刻的注意力熵

        H = -Σ p_i log(p_i)

        Args:
            attention_weights: [batch, num_heads, seq_len, seq_len] or [batch, seq_len, seq_len]
        """
        self.step_count += 1

        if self.step_count % self.compute_interval != 0:
            return

        if attention_weights.dim() == 4:
            attn = attention_weights.mean(dim=(1, 2, 3))
        elif attention_weights.dim() == 3:
            attn = attention_weights.mean(dim=(1, 2))
        else:
            attn = attention_weights.mean(dim=-1)

        entropy = -torch.sum(
            attention_weights * torch.log(attention_weights + 1e-10),
            dim=-1
        ).mean().item()

        self.attention_history.append(entropy)

        if self.ema_attention_entropy == 0:
            self.ema_attention_entropy = entropy
        else:
            self.ema_attention_entropy = self.ema_decay * self.ema_attention_entropy + (1 - self.ema_decay) * entropy

    def update_logits(self, logits: torch.Tensor) -> None:
        """
        计算并记录当前时刻的 logits 熵

        H = -Σ p_i log(p_i)，其中 p_i 是 softmax(logits)

        Args:
            logits: [batch, seq_len, vocab_size]
        """
        self.step_count += 1

        if self.step_count % self.compute_interval != 0:
            return

        # 计算 softmax 概率
        probs = torch.softmax(logits, dim=-1)
        # 计算熵
        entropy = -torch.sum(
            probs * torch.log(probs + 1e-10),
            dim=-1
        ).mean().item()

        self.logits_history.append(entropy)

        if self.ema_logits_entropy == 0:
            self.ema_logits_entropy = entropy
        else:
            self.ema_logits_entropy = self.ema_decay * self.ema_logits_entropy + (1 - self.ema_decay) * entropy

    def update_hidden(self, hidden_states: torch.Tensor) -> None:
        """
        记录 hidden states 统计信息

        Args:
            hidden_states: [batch, seq_len, hidden_dim]
        """
        self.step_count += 1

        if self.step_count % self.compute_interval != 0:
            return

        # 计算 hidden states 的统计信息
        mean = hidden_states.mean().item()
        std = hidden_states.std().item()
        norm = torch.norm(hidden_states, dim=-1).mean().item()

        hidden_stats = {
            "mean": mean,
            "std": std,
            "norm": norm
        }

        self.hidden_history.append(hidden_stats)

    def get_statistics(self) -> Dict[str, float]:
        """
        获取滑动统计

        Returns:
            dict包含:
            - mean: μ_H 熵均值
            - variance: σ_H² 熵方差
            - trend: k_H 趋势 (斜率)
            - current: 当前熵值
            - ema: EMA平滑熵
        """
        if len(self.attention_history) < 2:
            return {
                "mean": 0.0,
                "variance": 0.0,
                "trend": 0.0,
                "current": 0.0,
                "ema": 0.0
            }

        history = torch.tensor(list(self.attention_history))

        mean = history.mean().item()
        variance = history.var().item()
        trend = self._compute_trend(history)
        current = history[-1].item()

        return {
            "mean": mean,
            "variance": variance,
            "trend": trend,
            "current": current,
            "ema": self.ema_attention_entropy
        }

    def get_logits_statistics(self) -> Dict[str, float]:
        """
        获取 logits 熵统计

        Returns:
            dict包含 logits 熵的统计信息
        """
        if len(self.logits_history) < 2:
            return {
                "mean": 0.0,
                "variance": 0.0,
                "trend": 0.0,
                "current": 0.0,
                "ema": 0.0
            }

        history = torch.tensor(list(self.logits_history))

        mean = history.mean().item()
        variance = history.var().item()
        trend = self._compute_trend(history)
        current = history[-1].item()

        return {
            "mean": mean,
            "variance": variance,
            "trend": trend,
            "current": current,
            "ema": self.ema_logits_entropy
        }

    def get_hidden_statistics(self) -> Dict[str, float]:
        """
        获取 hidden states 统计

        Returns:
            dict包含 hidden states 的统计信息
        """
        if len(self.hidden_history) < 2:
            return {
                "mean": 0.0,
                "std": 0.0,
                "norm": 0.0
            }

        means = [h["mean"] for h in self.hidden_history]
        stds = [h["std"] for h in self.hidden_history]
        norms = [h["norm"] for h in self.hidden_history]

        return {
            "mean": float(np.mean(means)),
            "std": float(np.mean(stds)),
            "norm": float(np.mean(norms))
        }

    def _compute_trend(self, history: torch.Tensor) -> float:
        """
        计算趋势 (简单线性回归斜率)

        k_H > 0: 注意力发散
        k_H < 0: 注意力聚焦
        k_H = 0: 稳定

        Returns:
            trend: 斜率
        """
        if len(history) < 2:
            return 0.0

        x = torch.arange(len(history)).float()
        x_mean = x.mean()
        y_mean = history.mean()

        numerator = torch.sum((x - x_mean) * (history - y_mean))
        denominator = torch.sum((x - x_mean) ** 2)

        if denominator == 0:
            return 0.0

        slope = numerator / denominator

        return slope.item()

    def should_cutoff(self, entropy_threshold: float = 0.5, variance_threshold: float = 0.05) -> bool:
        """
        判断是否应该截断

        条件:
        - 低注意力熵 (μ_H < entropy_threshold)
        - 低方差 (σ_H < variance_threshold)
        - 趋势为负 (k_H < 0)

        Args:
            entropy_threshold: 熵阈值
            variance_threshold: 方差阈值

        Returns:
            should_cutoff: 是否应该截断
        """
        stats = self.get_statistics()

        if stats["mean"] < entropy_threshold:
            if stats["variance"] < variance_threshold:
                if stats["trend"] < 0:
                    return True

        return False

    def reset(self) -> None:
        """
        重置追踪器
        """
        self.attention_history.clear()
        self.logits_history.clear()
        self.hidden_history.clear()
        self.step_count = 0
        self.ema_attention_entropy = 0.0
        self.ema_logits_entropy = 0.0


class EntropyStatistics:
    """
    熵统计数据结构
    用于存储和传递统计结果
    """

    def __init__(
        self,
        mean: float = 0.0,
        variance: float = 0.0,
        trend: float = 0.0,
        current: float = 0.0,
        ema: float = 0.0
    ):
        self.mean = mean
        self.variance = variance
        self.trend = trend
        self.current = current
        self.ema = ema

    def to_dict(self) -> Dict[str, float]:
        return {
            "mean": self.mean,
            "variance": self.variance,
            "trend": self.trend,
            "current": self.current,
            "ema": self.ema
        }

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'EntropyStatistics':
        return cls(**data)

    def __repr__(self) -> str:
        return (f"EntropyStatistics(mean={self.mean:.4f}, variance={self.variance:.4f}, "
                f"trend={self.trend:.4f}, current={self.current:.4f}, ema={self.ema:.4f})")
