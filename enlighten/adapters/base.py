"""
Adapter Base - 适配器基类
定义 EnlightenLM 与不同推理框架适配的通用接口
"""

import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional, Tuple, List


class AdapterType(Enum):
    """适配器类型"""
    NATIVE = "native"           # HuggingFace 原生
    VLLM = "vllm"              # vLLM 高性能推理
    DEEPSEEK_V3 = "deepseek_v3"  # DeepSeek-V3 原生
    DEEPSEEK_V4 = "deepseek_v4"   # DeepSeek-V4 原生（计划中）


@dataclass
class AdapterConfig:
    """适配器配置"""
    adapter_type: AdapterType
    device: str = "cuda"
    max_memory_gb: int = 16
    enable_attention_sinks: bool = True
    sliding_window: int = 4096


class AdapterBase(ABC):
    """
    适配器基类

    定义 EnlightenLM 与不同推理框架适配的通用接口

    子类需要实现:
    - forward(): 前向传播
    - get_kv_cache(): 获取 KV 缓存
    - update_kv_cache(): 更新 KV 缓存
    - compute_attention(): 计算注意力
    """

    def __init__(self, config: AdapterConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            input_ids: 输入 token IDs [batch, seq_len]
            attention_mask: 注意力掩码 [batch, seq_len]

        Returns:
            logits: 输出 logits [batch, seq_len, vocab_size]
        """
        pass

    @abstractmethod
    def get_kv_cache(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取当前 KV 缓存

        Returns:
            (keys, values): KV 缓存张量对
        """
        pass

    @abstractmethod
    def update_kv_cache(
        self,
        keys: torch.Tensor,
        values: torch.Tensor
    ) -> None:
        """
        更新 KV 缓存

        Args:
            keys: 新的 key 张量
            values: 新的 value 张量
        """
        pass

    @abstractmethod
    def compute_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算注意力

        Args:
            query: Query 张量 [batch, num_heads, seq_len, head_dim]
            key: Key 张量 [batch, num_heads, seq_len, head_dim]
            value: Value 张量 [batch, num_heads, seq_len, head_dim]
            attention_mask: 注意力掩码

        Returns:
            attention_output: 注意力输出
        """
        pass

    def get_adapter_type(self) -> AdapterType:
        """获取适配器类型"""
        return self.config.adapter_type

    def is_available(self) -> bool:
        """检查适配器是否可用"""
        return True

    def get_capabilities(self) -> Dict[str, Any]:
        """
        获取适配器能力

        返回:
            能力字典
        """
        return {
            "type": self.config.adapter_type.value,
            "device": self.config.device,
            "supports_memory_efficient_attention": True,
            "supports_prefix_caching": True,
            "max_batch_size": 1,
            "max_sequence_length": 4096
        }


class MemoryEfficientAttentionMixin:
    """
    内存高效注意力混入类

    提供 Flash Attention 等高效注意力实现
    """

    def _apply_memory_efficient_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        softmax_scale: Optional[float] = None,
        is_causal: bool = True
    ) -> torch.Tensor:
        """
        应用内存高效的注意力机制

        使用 Flash Attention 或其他高效实现

        Args:
            query: [batch, num_heads, seq_len, head_dim]
            key: [batch, num_heads, seq_len, head_dim]
            value: [batch, num_heads, seq_len, head_dim]
            attention_mask: 可选的掩码
            dropout_p: Dropout 概率
            softmax_scale: Softmax 缩放因子
            is_causal: 是否使用因果掩码

        Returns:
            attention_output: [batch, num_heads, seq_len, head_dim]
        """
        try:
            from flash_attn import flash_attn_func
            output = flash_attn_func(
                query.transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                is_causal=is_causal
            )
            return output.transpose(1, 2)
        except ImportError:
            return self._fallback_attention(
                query, key, value, attention_mask, is_causal
            )

    def _fallback_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True
    ) -> torch.Tensor:
        """
        回退到标准注意力实现

        当 Flash Attention 不可用时使用
        """
        seq_len = query.size(2)
        head_dim = query.size(3)

        if is_causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=query.device, dtype=torch.bool),
                diagonal=1
            )
            if attention_mask is not None:
                attention_mask = attention_mask & ~causal_mask
            else:
                attention_mask = ~causal_mask

        scale = head_dim ** -0.5
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale

        if attention_mask is not None:
            scores = scores.masked_fill(~attention_mask, float('-inf'))

        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, value)

        return output
