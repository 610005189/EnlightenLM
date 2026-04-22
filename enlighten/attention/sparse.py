"""
Sparse Attention - 稀疏注意力实现
用于长上下文场景，降低计算复杂度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class SparseAttention(nn.Module):
    """
    稀疏注意力模块

    复杂度: O(n · m · d) 其中 m << n
    相比标准注意力的 O(n² · d) 大幅降低
    """

    def __init__(
        self,
        embed_dim: int,
        memory_size: int = 512,
        mode: str = "topk"
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.memory_size = memory_size
        self.mode = mode

        self.importance_scorer = nn.Linear(embed_dim, 1)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        稀疏注意力前向传播

        Args:
            query: [batch, seq_len, embed_dim]
            key: [batch, seq_len, embed_dim]
            value: [batch, seq_len, embed_dim]
            attention_mask: [batch, seq_len]

        Returns:
            output: [batch, seq_len, embed_dim]
            attention_weights: [batch, seq_len, seq_len] (稀疏)
        """
        batch_size, seq_len, _ = query.shape

        importance = self.importance_scorer(key).squeeze(-1)

        if attention_mask is not None:
            importance = importance.masked_fill(attention_mask == 0, float('-inf'))

        if self.mode == "topk":
            return self._topk_attention(query, key, value, importance)
        elif self.mode == "threshold":
            return self._threshold_attention(query, key, value, importance)
        else:
            return self._standard_attention(query, key, value)

    def _topk_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        importance: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Top-K稀疏注意力
        只保留最重要的K个key
        """
        batch_size, seq_len, embed_dim = query.shape

        topk = min(self.memory_size, seq_len)

        _, topk_indices = torch.topk(importance, k=topk, dim=-1)

        batch_indices = torch.arange(batch_size, device=query.device).unsqueeze(1).expand_as(topk_indices)

        key_selected = key[batch_indices, topk_indices]
        value_selected = value[batch_indices, topk_indices]

        scores = torch.matmul(query, key_selected.transpose(-2, -1)) / (embed_dim ** 0.5)

        attention_weights = F.softmax(scores, dim=-1)

        output = torch.matmul(attention_weights, value_selected)

        sparse_weights = torch.zeros_like(importance).unsqueeze(1)
        sparse_weights[batch_indices, :, topk_indices] = attention_weights

        return output, sparse_weights.squeeze(1)

    def _threshold_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        importance: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        阈值稀疏注意力
        只保留重要性大于阈值的key
        """
        batch_size, seq_len, embed_dim = query.shape

        threshold = importance.mean()

        mask = (importance > threshold).float()

        key_masked = key * mask.unsqueeze(-1)
        value_masked = value * mask.unsqueeze(-1)

        scores = torch.matmul(query, key_masked.transpose(-2, -1)) / (embed_dim ** 0.5)

        attention_weights = F.softmax(scores, dim=-1)

        output = torch.matmul(attention_weights, value_masked)

        return output, attention_weights


class FlashSparseAttention(nn.Module):
    """
    Flash-like稀疏注意力
    使用线性近似实现更高效的稀疏注意力
    """

    def __init__(self, embed_dim: int, reduction_ratio: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.reduction_ratio = reduction_ratio

        self.projection = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor
    ) -> torch.Tensor:
        """
        近似稀疏注意力

        使用线性投影近似标准注意力
        """
        batch_size, seq_len, embed_dim = query.shape

        reduced_len = max(1, int(seq_len * self.reduction_ratio))

        projected = self.projection(key)

        pooled = F.adaptive_avg_pool1d(
            projected.transpose(1, 2),
            output_size=reduced_len
        ).transpose(1, 2)

        scores = torch.matmul(query, pooled.transpose(-2, -1)) / (embed_dim ** 0.5)

        attention_weights = F.softmax(scores, dim=-1)

        output = torch.matmul(attention_weights, pooled)

        return output


class ChunkedSparseAttention(nn.Module):
    """
    分块稀疏注意力
    将序列分成多个chunk，每个chunk内部全连接，chunk间稀疏
    """

    def __init__(self, embed_dim: int, chunk_size: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.chunk_size = chunk_size

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor
    ) -> torch.Tensor:
        """
        分块稀疏注意力
        """
        batch_size, seq_len, embed_dim = query.shape

        num_chunks = (seq_len + self.chunk_size - 1) // self.chunk_size

        outputs = []

        for i in range(num_chunks):
            start = i * self.chunk_size
            end = min((i + 1) * self.chunk_size, seq_len)

            q_chunk = query[:, start:end, :]

            k_local = key[:, start:end, :]
            v_local = value[:, start:end, :]

            local_attn = torch.matmul(q_chunk, k_local.transpose(-2, -1)) / (embed_dim ** 0.5)
            local_attn = F.softmax(local_attn, dim=-1)
            local_output = torch.matmul(local_attn, v_local)

            if i > 0:
                k_prev = key[:, :start, :]
                v_prev = value[:, :start, :]

                cross_scores = torch.matmul(q_chunk, k_prev.transpose(-2, -1)) / (embed_dim ** 0.5)
                cross_attn = F.softmax(cross_scores, dim=-1)
                cross_output = torch.matmul(cross_attn, v_prev)

                alpha = 0.1
                local_output = local_output * (1 - alpha) + cross_output * alpha

            outputs.append(local_output)

        return torch.cat(outputs, dim=1)
