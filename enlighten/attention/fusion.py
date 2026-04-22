"""
Attention Fusion - 双流注意力融合
融合DAN和VAN的输出，通过门控机制动态调整
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class AttentionFusion(nn.Module):
    """
    双流注意力融合模块

    融合公式: Attn_fused = g · Attn_DAN + (1-g) · Attn_VAN

    特点:
    - 门控机制动态调整双流权重
    - 支持动态温度和稀疏截断
    - 稳定性标志控制计算模式
    """

    def __init__(self, embed_dim: int = 1024, gate_hidden_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        self.gate_hidden_dim = gate_hidden_dim

        self.gate_predictor = nn.Sequential(
            nn.Linear(embed_dim * 2, gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(gate_hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        attn_dan: torch.Tensor,
        attn_van: torch.Tensor,
        tau: float = 1.0,
        theta: float = 0.5
    ) -> torch.Tensor:
        """
        融合双流注意力

        Args:
            attn_dan: [batch, seq_len, embed_dim] DAN注意力输出
            attn_van: [batch, seq_len, embed_dim] VAN注意力输出
            tau: 温度参数，控制分布锐度
            theta: 稀疏截断阈值

        Returns:
            fused: [batch, seq_len, embed_dim] 融合后的注意力
        """
        dan_mean = attn_dan.mean(dim=1)
        van_mean = attn_van.mean(dim=1)

        combined = torch.cat([dan_mean, van_mean], dim=-1)

        g = self.gate_predictor(combined)

        fused = g * attn_dan + (1 - g) * attn_van

        fused = self.apply_temperature(fused, tau)

        fused = self.apply_sparse_threshold(fused, theta)

        return fused

    def apply_temperature(self, attn: torch.Tensor, tau: float) -> torch.Tensor:
        """
        应用温度参数

        温度控制分布锐度:
        - tau < 1: 分布更锐利
        - tau = 1: 保持不变
        - tau > 1: 分布更平滑
        """
        return F.softmax(attn / tau, dim=-1)

    def apply_sparse_threshold(self, attn: torch.Tensor, theta: float) -> torch.Tensor:
        """
        应用稀疏截断

        只保留大于阈值theta的注意力权重
        """
        mask = (attn > theta).float()
        return attn * mask


class DynamicFusion(nn.Module):
    """
    动态融合模块
    支持多种融合策略
    """

    def __init__(self, embed_dim: int, strategy: str = "gate"):
        super().__init__()
        self.embed_dim = embed_dim
        self.strategy = strategy

        if strategy == "gate":
            self.fusion = AttentionFusion(embed_dim)
        elif strategy == "additive":
            self.weight_dan = nn.Parameter(torch.tensor(0.5))
            self.weight_van = nn.Parameter(torch.tensor(0.5))
        elif strategy == "multiplicative":
            self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        attn_dan: torch.Tensor,
        attn_van: torch.Tensor,
        tau: float = 1.0,
        theta: float = 0.5
    ) -> torch.Tensor:
        """
        根据策略融合双流注意力
        """
        if self.strategy == "gate":
            return self.fusion(attn_dan, attn_van, tau, theta)
        elif self.strategy == "additive":
            w_dan = torch.sigmoid(self.weight_dan)
            w_van = torch.sigmoid(self.weight_van)
            fused = w_dan * attn_dan + w_van * attn_van
            return F.softmax(fused / tau, dim=-1) * (fused > theta).float()
        elif self.strategy == "multiplicative":
            scale = torch.sigmoid(self.scale)
            fused = (attn_dan * attn_van) ** scale
            return F.softmax(fused / tau, dim=-1) * (fused > theta).float()
        else:
            return (attn_dan + attn_van) / 2


class StabilityTracker(nn.Module):
    """
    稳定性追踪器
    用于决定是否只计算单流注意力
    """

    def __init__(self, threshold: float = 0.1):
        super().__init__()
        self.threshold = threshold

    def forward(
        self,
        attn_dan: torch.Tensor,
        attn_van: torch.Tensor
    ) -> bool:
        """
        判断双流是否稳定

        Returns:
            stable: 是否稳定 (True=稳定, False=不稳定)
        """
        dan_mean = attn_dan.mean().item()
        van_mean = attn_van.mean().item()

        diff = abs(dan_mean - van_mean)

        return diff < self.threshold
