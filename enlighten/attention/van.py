"""
VAN (Ventral Attention Network) - 刺激驱动注意力网络
三级漏斗机制：light → medium → full
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Set, Dict, Any
from dataclasses import dataclass


@dataclass
class VANResult:
    """VAN 检测结果"""
    van_event: bool
    p_harm: float
    event_type: int
    level: str
    sensitive_tokens: Optional[List[int]] = None


class KeywordMatcher:
    """
    关键词匹配器 - VAN Level light
    基于规则的正则匹配，检测敏感词
    """

    def __init__(self, keywords: Optional[List[str]] = None):
        self.keywords = set(keywords) if keywords else set()

    def add_keywords(self, keywords: List[str]) -> None:
        """添加关键词"""
        self.keywords.update(keywords)

    def detect(self, tokens: List[int], token_to_word: Optional[Dict[int, str]] = None) -> Tuple[bool, List[int]]:
        """
        检测敏感词

        Args:
            tokens: token ID 列表
            token_to_word: 可选的 token 到词的映射

        Returns:
            (has_sensitive, sensitive_token_indices)
        """
        if not self.keywords:
            return False, []

        sensitive_indices = []

        if token_to_word:
            for i, tok_id in enumerate(tokens):
                word = token_to_word.get(tok_id, "")
                if word.lower() in self.keywords:
                    sensitive_indices.append(i)

        return len(sensitive_indices) > 0, sensitive_indices


class LightweightMLPClassifier(nn.Module):
    """
    轻量 MLP 分类器 - VAN Level medium/full
    基于隐藏状态预测有害概率
    """

    def __init__(self, vocab_size: int = 50000, embed_dim: int = 768, hidden_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, hidden_states: torch.Tensor) -> float:
        """
        预测有害概率

        Args:
            hidden_states: [batch, seq_len, embed_dim]

        Returns:
            p_harm: 有害概率 ∈ [0, 1]
        """
        pooled = hidden_states.mean(dim=1)
        p_harm = self.classifier(pooled).item()
        return p_harm


class FullAttentionChecker(nn.Module):
    """
    完整注意力检查器 - VAN Level full
    使用完整注意力机制检测异常模式
    """

    def __init__(self, embed_dim: int = 768, num_heads: int = 12):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.attention_threshold = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, hidden_states: torch.Tensor) -> Tuple[float, torch.Tensor]:
        """
        完整注意力检查

        Args:
            hidden_states: [batch, seq_len, embed_dim]

        Returns:
            (p_harm, attention_weights)
        """
        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)

        attended = torch.matmul(attention_weights, v)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

        threshold_output = self.attention_threshold(attended)
        p_harm = threshold_output.mean().item()

        return p_harm, attention_weights


class VANFunnel:
    """
    VAN 三级漏斗协调器

    Level light: 关键词匹配（计算量最小）
    Level medium: 关键词 + 轻量 MLP 分类器
    Level full: 关键词 + MLP + 完整注意力（最精确）

    Args:
        level: 漏斗级别 ("light" | "medium" | "full")
        vocab_size: 词表大小
        embed_dim: 嵌入维度
        van_threshold: VAN 触发阈值
    """

    def __init__(
        self,
        level: str = "medium",
        vocab_size: int = 50000,
        embed_dim: int = 768,
        van_threshold: float = 0.9,
        sensitive_keywords: Optional[List[str]] = None
    ):
        self.level = level
        self.van_threshold = van_threshold

        self.keyword_matcher = KeywordMatcher(sensitive_keywords)

        self.mlp_classifier = None
        self.full_checker = None

        if level in ["medium", "full"]:
            self.mlp_classifier = LightweightMLPClassifier(vocab_size, embed_dim)

        if level == "full":
            self.full_checker = FullAttentionChecker(embed_dim)

    def forward(
        self,
        tokens: torch.Tensor,
        hidden_states: torch.Tensor,
        token_to_word: Optional[Dict[int, str]] = None
    ) -> VANResult:
        """
        VAN 检测前向传播

        Args:
            tokens: [batch, seq_len] token IDs
            hidden_states: [batch, seq_len, embed_dim] 隐藏状态
            token_to_word: 可选的 token 到词的映射

        Returns:
            VANResult: 检测结果
        """
        token_list = tokens[0].tolist() if tokens.dim() > 1 else tokens.tolist()

        keyword_detected, sensitive_indices = self.keyword_matcher.detect(token_list, token_to_word)

        if keyword_detected:
            return VANResult(
                van_event=True,
                p_harm=1.0,
                event_type=1,
                level=self.level,
                sensitive_tokens=sensitive_indices
            )

        p_harm = 0.0

        if self.level in ["medium", "full"] and self.mlp_classifier is not None:
            p_harm = self.mlp_classifier(hidden_states)

            if p_harm >= self.van_threshold:
                return VANResult(
                    van_event=True,
                    p_harm=p_harm,
                    event_type=2,
                    level=self.level
                )

        if self.level == "full" and self.full_checker is not None:
            full_p_harm, _ = self.full_checker(hidden_states)
            p_harm = max(p_harm, full_p_harm)

            if full_p_harm >= self.van_threshold:
                return VANResult(
                    van_event=True,
                    p_harm=full_p_harm,
                    event_type=2,
                    level=self.level
                )

        return VANResult(
            van_event=False,
            p_harm=p_harm,
            event_type=0,
            level=self.level
        )

    def set_level(self, level: str) -> None:
        """
        设置漏斗级别

        Args:
            level: 新的级别 ("light" | "medium" | "full")
        """
        if level not in ["light", "medium", "full"]:
            raise ValueError(f"Unknown VAN level: {level}")

        old_level = self.level
        self.level = level

        if level in ["medium", "full"] and self.mlp_classifier is None:
            self.mlp_classifier = LightweightMLPClassifier()

        if level == "full" and self.full_checker is None:
            self.full_checker = FullAttentionChecker()

        if level == "light":
            self.mlp_classifier = None
            self.full_checker = None
        elif level == "medium":
            self.full_checker = None


# 向后兼容别名
VANAttention = VANFunnel


class SimplifiedVAN(nn.Module):
    """
    简化版VAN - 用于快速原型验证
    基于规则检测，无学习参数
    """

    def __init__(self, sensitive_tokens: List[int] = None):
        super().__init__()
        self.sensitive_tokens = set(sensitive_tokens or [])

    def forward(
        self,
        tokens: torch.Tensor,
        hidden_states: torch.Tensor
    ) -> Tuple[bool, int]:
        """
        简化版前向传播

        Args:
            tokens: [batch, seq_len]
            hidden_states: [batch, seq_len, embed_dim]

        Returns:
            van_event: 是否触发VAN事件
            event_type: 事件类型 (0=无, 1=敏感, 2=严重)
        """
        if self.sensitive_tokens:
            token_set = set(tokens.flatten().tolist())
            if token_set & self.sensitive_tokens:
                return True, 1

        return False, 0


class SaliencyDetector(nn.Module):
    """
    显著性检测器
    用于识别输入中的重要/异常部分
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, 1)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        计算显著性分数

        Args:
            hidden_states: [batch, seq_len, embed_dim]

        Returns:
            saliency_map: [batch, seq_len] 显著性分数
        """
        scores = self.attention(hidden_states).squeeze(-1)
        saliency = torch.sigmoid(scores)
        return saliency


class InterruptMaskGenerator(nn.Module):
    """
    中断掩码生成器
    根据事件类型生成相应的掩码
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.event_classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 3)
        )

    def forward(
        self,
        event_type: int,
        seq_len: int,
        batch_size: int,
        device: torch.device = torch.device('cpu')
    ) -> torch.Tensor:
        """
        生成中断掩码

        Args:
            event_type: 事件类型 (0=无, 1=敏感, 2=严重)
            seq_len: 序列长度
            batch_size: batch大小
            device: 设备

        Returns:
            mask: [batch, seq_len, seq_len]
        """
        mask = torch.ones(batch_size, seq_len, seq_len, device=device)

        if event_type == 1:
            mask = torch.tril(mask)
        elif event_type == 2:
            mask = torch.zeros(batch_size, seq_len, seq_len, device=device)
            mask[:, -1, :] = 1.0

        return mask
