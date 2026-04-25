"""
DAN (Default Attention Network) - 目标驱动注意力网络
根据任务类型强制引导注意力方向
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class DANAttention(nn.Module):
    """
    DAN (目标驱动注意力网络)

    特点:
    - 根据任务类型强制引导注意力方向
    - 任务偏置 B_DAN 由 L3 下发
    - 目标驱动的主动聚焦
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 12,
        task_bias_dim: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.task_bias_proj = nn.Linear(task_bias_dim, num_heads)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        task_bias: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            query: [batch, seq_len, embed_dim]
            key: [batch, seq_len, embed_dim]
            value: [batch, seq_len, embed_dim]
            task_bias: [batch, task_bias_dim] 任务偏置向量
            attention_mask: [batch, seq_len] or [batch, seq_len, seq_len]

        Returns:
            output: [batch, seq_len, embed_dim]
            attention_weights: [batch, num_heads, seq_len, seq_len]
        """
        batch_size = query.size(0)

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        task_bias = self.task_bias_proj(task_bias)
        task_bias = task_bias.unsqueeze(2).unsqueeze(2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        scores = scores + task_bias

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context = torch.matmul(attention_weights, v)

        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        output = self.out_proj(context)

        return output, attention_weights


class SimplifiedDAN(nn.Module):
    """
    简化版DAN - 用于快速原型验证
    不包含多头注意力，专注于任务偏置机制
    """

    def __init__(self, embed_dim: int, task_bias_dim: int = 128):
        super().__init__()
        self.embed_dim = embed_dim
        self.task_bias_proj = nn.Linear(task_bias_dim, 1)
        self.scale = embed_dim ** 0.5

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        task_bias: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        简化版前向传播

        Args:
            query: [batch, seq_len, embed_dim]
            key: [batch, seq_len, embed_dim]
            value: [batch, seq_len, embed_dim]
            task_bias: [batch, task_bias_dim]

        Returns:
            output: [batch, seq_len, embed_dim]
            attention_weights: [batch, seq_len, seq_len]
        """
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale

        bias = self.task_bias_proj(task_bias)
        bias = torch.sigmoid(bias)
        scores = scores * bias

        attention_weights = F.softmax(scores, dim=-1)

        output = torch.matmul(attention_weights, value)

        return output, attention_weights
