"""
DMN (Default Mode Network) 抑制模块
防止无意义自循环，抑制内部噪声
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DMNInhibition(nn.Module):
    """
    DMN抑制模块

    功能:
    - 估计内部噪声 ξ
    - 应用抑制 α · ξ
    - LayerNorm归一化确保扰动不累积
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        noise_dim: int = 512,
        inhibition_strength: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.noise_dim = noise_dim
        self.inhibition_strength = inhibition_strength

        self.noise_estimator = nn.Sequential(
            nn.Linear(embed_dim, noise_dim),
            nn.Tanh(),
            nn.Linear(noise_dim, embed_dim)
        )

        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        alpha: float = 0.1
    ) -> torch.Tensor:
        """
        应用DMN抑制

        Args:
            hidden_states: [batch, seq_len, embed_dim]
            alpha: 抑制系数 ∈ [0, 1]

        Returns:
            inhibited: 抑制后的hidden_states
        """
        noise = self.noise_estimator(hidden_states)

        inhibited = hidden_states - alpha * noise * self.inhibition_strength

        inhibited = self.layer_norm(inhibited)

        return inhibited


class DMNController(nn.Module):
    """
    DMN控制器
    动态调节DMN抑制强度
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        hidden_dim: int = 256
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.alpha_predictor = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        base_alpha: float = 0.1
    ) -> torch.Tensor:
        """
        预测最优抑制系数

        Args:
            hidden_states: [batch, seq_len, embed_dim]
            base_alpha: 基础抑制系数

        Returns:
            alpha: 抑制系数
        """
        pooled = hidden_states.mean(dim=1)

        alpha = self.alpha_predictor(pooled).squeeze(-1)

        alpha = alpha * base_alpha * 10

        return alpha.clamp(0.0, 1.0)


class DMNWithGate(nn.Module):
    """
    带门控的DMN抑制
    更精细地控制抑制强度
    """

    def __init__(self, embed_dim: int = 1024):
        super().__init__()
        self.embed_dim = embed_dim

        self.noise_estimator = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, embed_dim)
        )

        self.gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Sigmoid(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        alpha: float = 0.1
    ) -> torch.Tensor:
        """
        带门控的DMN抑制
        """
        noise = self.noise_estimator(hidden_states)

        gate_value = self.gate(hidden_states).squeeze(-1).unsqueeze(-1)

        inhibited = hidden_states - alpha * noise * gate_value * 0.1

        return inhibited


class TimeVaryingDMN(nn.Module):
    """
    时变DMN抑制
    抑制强度随时间/步数变化
    """

    def __init__(self, embed_dim: int = 1024):
        super().__init__()
        self.embed_dim = embed_dim

        self.noise_estimator = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, embed_dim)
        )

        self.time_encoder = nn.Embedding(1000, embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        step: int,
        base_alpha: float = 0.1
    ) -> torch.Tensor:
        """
        时变DMN抑制

        抑制强度随step增加而增加
        """
        noise = self.noise_estimator(hidden_states)

        time_embedding = self.time_encoder(torch.tensor([step % 1000], device=hidden_states.device))

        time_alpha = torch.sigmoid(time_embedding.mean(dim=1))

        alpha = base_alpha * (1 + 0.1 * step) * time_alpha

        inhibited = hidden_states - alpha.unsqueeze(-1) * noise * 0.1

        return inhibited
