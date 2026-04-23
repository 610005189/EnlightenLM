"""
L1 Generation Layer - L1生成层
双流注意力(DAN+VAN) + DMN抑制 + 遗忘门
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass

from .attention.dan import DANAttention, SimplifiedDAN
from .attention.van import VANAttention, SimplifiedVAN
from .attention.fusion import AttentionFusion, DynamicFusion, StabilityTracker
from .attention.sparse import SparseAttention


@dataclass
class L1Output:
    """L1层输出"""
    output_ids: torch.Tensor
    hidden_states: torch.Tensor
    attention_weights: torch.Tensor
    entropy_stats: Dict[str, float]
    van_event: bool
    p_harm: float
    control_signals: Dict[str, Any]


class L1Generation(nn.Module):
    """
    L1 生成层

    组成:
    - 双流注意力 (DAN + VAN)
    - 注意力融合
    - DMN抑制
    - 遗忘门

    数据流:
    Input → DAN → VAN → Fusion → DMN → ForgetGate → Output
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: Optional[Dict] = None
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

        config = config or {}
        self.embed_dim = config.get("embed_dim", 1024)
        self.num_heads = config.get("num_heads", 12)
        self.task_bias_dim = config.get("task_bias_dim", 128)
        self.van_level = config.get("van_level", "medium")
        self.sensitive_keywords = config.get("sensitive_keywords", [])

        self.dan = DANAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            task_bias_dim=self.task_bias_dim
        )

        from .attention.van import VANFunnel
        self.van = VANFunnel(
            level=self.van_level,
            embed_dim=self.embed_dim,
            vocab_size=tokenizer.vocab_size if tokenizer else 50000,
            sensitive_keywords=self.sensitive_keywords
        )

        self.fusion = AttentionFusion(
            embed_dim=self.embed_dim,
            gate_hidden_dim=self.embed_dim // 4
        )

        self.stability_tracker = StabilityTracker(threshold=0.1)

        self.sparse_attention = SparseAttention(
            embed_dim=self.embed_dim,
            memory_size=config.get("memory_size", 512)
        )

        self.dmn = DMNInhibition(embed_dim=self.embed_dim)
        self.forget_gate = ForgetGate(embed_dim=self.embed_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        control_signals: Optional[Dict] = None,
        task_bias: Optional[torch.Tensor] = None
    ) -> L1Output:
        """
        L1层前向传播

        Args:
            input_ids: [batch, seq_len] 输入token IDs
            attention_mask: [batch, seq_len]
            control_signals: 来自L3的调控信号
            task_bias: [batch, task_bias_dim] 任务偏置

        Returns:
            L1Output: 包含输出hidden_states和attention_weights
        """
        control_signals = control_signals or {}

        outputs = self.model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.last_hidden_state

        if task_bias is None:
            task_bias = torch.zeros(input_ids.size(0), self.task_bias_dim, device=input_ids.device)

        dan_output, _ = self.dan(
            hidden_states, hidden_states, hidden_states, task_bias
        )

        # 使用新的 VANFunnel 接口
        van_result = self.van(input_ids, hidden_states)
        van_event = van_result.van_event
        p_harm = van_result.p_harm

        # 简化的 VAN 输出（用于融合）
        van_output = hidden_states * (1 - p_harm)  # 基于有害概率调整

        tau = control_signals.get("tau", 1.0)
        theta = control_signals.get("theta", 0.5)

        fused_output = self.fusion(dan_output, van_output, tau=tau, theta=theta)

        alpha = control_signals.get("alpha", 0.1)
        inhibited_output = self.dmn(fused_output, alpha=alpha)

        prev_hidden = torch.zeros_like(inhibited_output) if not hasattr(self, 'prev_hidden') else self.prev_hidden
        decay_rate = control_signals.get("decay_rate", 0.95)
        output = self.forget_gate(prev_hidden, inhibited_output, decay_rate=decay_rate)

        self.prev_hidden = output.detach()

        attention_weights = torch.matmul(
            output, hidden_states.transpose(-2, -1)
        ) / (self.embed_dim ** 0.5)
        attention_weights = F.softmax(attention_weights, dim=-1)

        entropy_stats = self._compute_entropy_stats(attention_weights)

        return L1Output(
            output_ids=torch.argmax(output, dim=-1),
            hidden_states=output,
            attention_weights=attention_weights,
            entropy_stats=entropy_stats,
            van_event=van_event,
            p_harm=p_harm,
            control_signals=control_signals
        )

    def _compute_entropy_stats(self, attention_weights: torch.Tensor) -> Dict[str, float]:
        """
        计算注意力熵统计
        """
        entropy = -torch.sum(
            attention_weights * torch.log(attention_weights + 1e-10),
            dim=-1
        ).mean().item()

        return {
            "mean": entropy,
            "variance": 0.0,
            "current": entropy
        }


class DMNInhibition(nn.Module):
    """
    DMN (Default Mode Network) 抑制模块

    功能:
    - 估计内部噪声 ξ
    - 应用抑制 α · ξ
    - 防止无意义自循环
    """

    def __init__(self, embed_dim: int, noise_dim: int = 512):
        super().__init__()
        self.embed_dim = embed_dim
        self.noise_estimator = nn.Sequential(
            nn.Linear(embed_dim, noise_dim),
            nn.Tanh(),
            nn.Linear(noise_dim, embed_dim)
        )
        self.inhibition_strength = 0.1

    def forward(
        self,
        hidden_states: torch.Tensor,
        alpha: float = 0.1
    ) -> torch.Tensor:
        """
        应用DMN抑制

        Args:
            hidden_states: [batch, seq_len, embed_dim]
            alpha: 抑制系数

        Returns:
            inhibited: 抑制后的hidden_states
        """
        noise = self.noise_estimator(hidden_states)
        inhibited = hidden_states - alpha * noise * self.inhibition_strength
        return inhibited


class ForgetGate(nn.Module):
    """
    遗忘门模块

    功能:
    - 提供指数衰减的KV缓存
    - 防止模型"陷入"过去的无效状态
    """

    def __init__(self, embed_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim

        self.forget_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

        self.decay_rate = 0.95

    def forward(
        self,
        prev_hidden: torch.Tensor,
        current_input: torch.Tensor,
        decay_rate: Optional[float] = None
    ) -> torch.Tensor:
        """
        应用遗忘门

        Args:
            prev_hidden: [batch, seq_len, embed_dim]
            current_input: [batch, seq_len, embed_dim]
            decay_rate: 衰减率

        Returns:
            output: 遗忘后的输出
        """
        combined = torch.cat([prev_hidden, current_input], dim=-1)
        f = self.forget_proj(combined)

        decay = decay_rate or self.decay_rate
        decayed = f * decay

        output = decayed * prev_hidden + (1 - decayed) * current_input

        return output


class SimplifiedL1(nn.Module):
    """
    简化版L1生成层 - 用于快速原型验证
    """

    def __init__(self, embed_dim: int = 512, task_bias_dim: int = 64):
        super().__init__()
        self.embed_dim = embed_dim

        self.dan = SimplifiedDAN(embed_dim, task_bias_dim)
        self.van = SimplifiedVAN()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        task_bias: torch.Tensor,
        tokens: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, bool]:
        """
        简化版前向传播

        Returns:
            output: [batch, seq_len, embed_dim]
            van_event: 是否触发VAN事件
        """
        dan_output, _ = self.dan(query, key, value, task_bias)

        if tokens is not None:
            van_event, _ = self.van(tokens, dan_output)
        else:
            van_event = False

        return dan_output, van_event
