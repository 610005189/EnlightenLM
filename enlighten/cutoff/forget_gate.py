"""
Forget Gate - 遗忘门模块
提供指数衰减的KV缓存，防止模型"陷入"过去状态
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ForgetGate(nn.Module):
    """
    遗忘门模块

    功能:
    - 计算遗忘门值 f ∈ [0, 1]
    - 提供指数衰减的KV缓存
    - 选择性遗忘无效状态
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        hidden_dim: Optional[int] = None,
        decay_rate: float = 0.95
    ):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.decay_rate = decay_rate

        self.forget_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

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
            decay_rate: 衰减率 (覆盖默认值)

        Returns:
            output: 遗忘后的输出
        """
        combined = torch.cat([prev_hidden, current_input], dim=-1)
        f = self.forget_proj(combined)

        decay = decay_rate or self.decay_rate
        decayed = f * decay

        output = decayed * prev_hidden + (1 - decayed) * current_input

        return output


class AdaptiveForgetGate(nn.Module):
    """
    自适应遗忘门
    根据输入动态调整遗忘强度
    """

    def __init__(self, embed_dim: int = 1024):
        super().__init__()
        self.embed_dim = embed_dim

        self.gate_network = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.Sigmoid()
        )

    def forward(
        self,
        prev_hidden: torch.Tensor,
        current_input: torch.Tensor,
        memory: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        自适应遗忘

        Args:
            prev_hidden: 上一个隐藏状态
            current_input: 当前输入
            memory: 可选的记忆向量
        """
        if memory is None:
            memory = prev_hidden.mean(dim=1, keepdim=True)

        combined = torch.cat([prev_hidden, current_input, memory], dim=-1)
        f = self.gate_network(combined)

        output = f * prev_hidden + (1 - f) * current_input

        return output


class ForgetGateWithAttention(nn.Module):
    """
    带注意力机制的遗忘门
    根据注意力权重调整遗忘强度
    """

    def __init__(self, embed_dim: int = 1024):
        super().__init__()
        self.embed_dim = embed_dim

        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )

        self.attention_proj = nn.Linear(embed_dim, 1)

    def forward(
        self,
        prev_hidden: torch.Tensor,
        current_input: torch.Tensor,
        attention_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        带注意力遗忘

        注意力高的位置遗忘强度低
        """
        combined = torch.cat([prev_hidden, current_input], dim=-1)
        f = self.gate(combined)

        attn_scores = self.attention_proj(prev_hidden).squeeze(-1)

        attn_mask = torch.sigmoid(attn_scores)

        f_adjusted = f * (1 + attn_mask.unsqueeze(-1)) / 2

        output = f_adjusted * prev_hidden + (1 - f_adjusted) * current_input

        return output


class HierarchicalForgetGate(nn.Module):
    """
    分层遗忘门
    在多个时间尺度上进行遗忘
    """

    def __init__(self, embed_dim: int = 1024, num_layers: int = 3):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim // (i + 1)),
                nn.Sigmoid()
            )
            for i in range(num_layers)
        ])

        self.decay_rates = [0.95, 0.9, 0.8]

    def forward(
        self,
        prev_hidden: torch.Tensor,
        current_input: torch.Tensor
    ) -> torch.Tensor:
        """
        分层遗忘

        每层使用不同的衰减率
        """
        output = current_input

        for i, (gate, decay) in enumerate(zip(self.gates, self.decay_rates)):
            combined = torch.cat([prev_hidden, output], dim=-1)
            f = gate(combined)

            output = f * decay * prev_hidden + (1 - f) * output

        return output


class MemoryAugmentedForgetGate(nn.Module):
    """
    记忆增强遗忘门
    结合外部记忆进行遗忘决策
    """

    def __init__(self, embed_dim: int = 1024, memory_size: int = 512):
        super().__init__()
        self.embed_dim = embed_dim
        self.memory_size = memory_size

        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.Sigmoid()
        )

        self.memory_proj = nn.Linear(memory_size, embed_dim)

    def forward(
        self,
        prev_hidden: torch.Tensor,
        current_input: torch.Tensor,
        external_memory: torch.Tensor
    ) -> torch.Tensor:
        """
        记忆增强遗忘

        Args:
            prev_hidden: [batch, seq_len, embed_dim]
            current_input: [batch, seq_len, embed_dim]
            external_memory: [batch, memory_size]
        """
        memory_repr = self.memory_proj(external_memory).unsqueeze(1)

        combined = torch.cat([prev_hidden, current_input, memory_repr], dim=-1)
        f = self.gate(combined)

        output = f * prev_hidden + (1 - f) * current_input

        return output
