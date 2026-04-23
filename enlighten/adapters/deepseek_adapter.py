"""
DeepSeek Adapter - DeepSeek 模型适配器
支持 DeepSeek-V3 和未来 DeepSeek-V4 模型的适配
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass

from .base import AdapterBase, AdapterConfig, AdapterType, MemoryEfficientAttentionMixin


@dataclass
class DeepSeekConfig:
    """DeepSeek 适配器配置"""
    model_name: str = "deepseek-ai/DeepSeek-V3"
    api_key_env: str = "DEEPSEEK_API_KEY"
    max_tokens: int = 2048
    temperature: float = 0.7
    use_api: bool = True


class DeepSeekAdapter(AdapterBase, MemoryEfficientAttentionMixin):
    """
    DeepSeek 适配器

    支持 DeepSeek-V3 和未来 DeepSeek-V4 模型的适配

    特性:
    - 支持 API 模式和本地模式
    - 与 EnlightenLM L1/L2/L3 组件无缝集成
    - 稀疏注意力支持
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        config: Optional[AdapterConfig] = None,
        deepseek_config: Optional[DeepSeekConfig] = None
    ):
        if config is None:
            config = AdapterConfig(adapter_type=AdapterType.DEEPSEEK_V3)

        super().__init__(config)

        self.model = model
        self.deepseek_config = deepseek_config or DeepSeekConfig()

        self._init_api_client()

        self.prefix_caching_enabled = True
        self.prefix_hashes = {}

    def _init_api_client(self) -> None:
        """初始化 API 客户端"""
        if self.deepseek_config.use_api:
            from ..api.deepseek_client import DeepSeekAPIClient, DeepSeekConfig

            import os
            api_key = os.environ.get(self.deepseek_config.api_key_env)

            if api_key:
                api_config = DeepSeekConfig(
                    api_key=api_key,
                    model=self.deepseek_config.model_name
                )
                self.api_client = DeepSeekAPIClient(api_config)
            else:
                self.api_client = None
        else:
            self.api_client = None

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
        if self.model is not None:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            return outputs.logits
        else:
            raise ValueError("本地模型未加载，请使用 API 模式")

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> Tuple[str, float]:
        """
        使用 API 生成文本

        Args:
            prompt: 输入提示
            max_tokens: 最大 token 数
            temperature: 温度参数

        Returns:
            (response_text, latency)
        """
        if self.api_client is None:
            return "API 客户端未初始化", 0.0

        return self.api_client.generate(
            prompt=prompt,
            max_tokens=max_tokens or self.deepseek_config.max_tokens,
            temperature=temperature or self.deepseek_config.temperature,
            **kwargs
        )

    def generate_stream(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ):
        """
        流式生成文本

        Args:
            prompt: 输入提示
            max_tokens: 最大 token 数
            temperature: 温度参数

        Yields:
            生成的文本片段
        """
        if self.api_client is None:
            yield "API 客户端未初始化"
            return

        yield from self.api_client.generate_stream(
            prompt=prompt,
            max_tokens=max_tokens or self.deepseek_config.max_tokens,
            temperature=temperature or self.deepseek_config.temperature,
            **kwargs
        )

    def get_kv_cache(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取当前层的 KV 缓存"""
        if self.model is None:
            return torch.zeros(1), torch.zeros(1)

        if hasattr(self.model, "get_kv_cache"):
            return self.model.get_kv_cache()

        return torch.zeros(1), torch.zeros(1)

    def update_kv_cache(
        self,
        keys: torch.Tensor,
        values: torch.Tensor
    ) -> None:
        """更新 KV 缓存"""
        pass

    def compute_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        使用内存高效注意力计算

        Args:
            query: [batch, num_heads, seq_len, head_dim]
            key: [batch, num_heads, seq_len, head_dim]
            value: [batch, num_heads, seq_len, head_dim]
            attention_mask: [batch, seq_len]

        Returns:
            attention_output: [batch, num_heads, seq_len, head_dim]
        """
        return self._apply_memory_efficient_attention(
            query,
            key,
            value,
            attention_mask=attention_mask,
            is_causal=True
        )

    def compute_sparse_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        sparse_indices: List[int],
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        稀疏注意力计算

        Args:
            query: [batch, num_heads, seq_len, head_dim]
            key: [batch, num_heads, seq_len, head_dim]
            value: [batch, num_heads, seq_len, head_dim]
            sparse_indices: 稀疏计算的索引列表
            attention_mask: 注意力掩码

        Returns:
            attention_output: 稀疏注意力输出
        """
        batch_size, num_heads, seq_len, head_dim = query.shape

        sparse_key = key[:, :, sparse_indices, :]
        sparse_value = value[:, :, sparse_indices, :]

        scale = head_dim ** -0.5
        scores = torch.matmul(query, sparse_key.transpose(-2, -1)) * scale

        if attention_mask is not None:
            sparse_mask = attention_mask[:, sparse_indices]
            scores = scores.masked_fill(~sparse_mask.unsqueeze(1), float('-inf'))

        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, sparse_value)

        return output

    def enable_prefix_caching(self) -> None:
        """启用前缀缓存"""
        self.prefix_caching_enabled = True

    def disable_prefix_caching(self) -> None:
        """禁用前缀缓存"""
        self.prefix_caching_enabled = False

    def get_prefix_hash(self, prompt: str) -> Optional[str]:
        """获取提示的前缀哈希"""
        return self.prefix_hashes.get(prompt[:100])

    def cache_prefix(self, prompt: str, kv_cache: Tuple[torch.Tensor, torch.Tensor]) -> None:
        """缓存提示的 KV 缓存"""
        if not self.prefix_caching_enabled:
            return

        prefix = prompt[:100]
        self.prefix_hashes[prefix] = kv_cache

    def clear_prefix_cache(self) -> None:
        """清除前缀缓存"""
        self.prefix_hashes.clear()

    def is_available(self) -> bool:
        """检查适配器是否可用"""
        if self.api_client is not None:
            return self.api_client.is_available()

        if self.model is not None:
            return True

        return False

    def get_capabilities(self) -> Dict[str, Any]:
        """获取 DeepSeek 适配器能力"""
        base_capabilities = super().get_capabilities()

        deepseek_capabilities = {
            "type": "deepseek",
            "model_name": self.deepseek_config.model_name,
            "supports_api": self.deepseek_config.use_api,
            "supports_prefix_caching": self.prefix_caching_enabled,
            "max_tokens": self.deepseek_config.max_tokens,
            "temperature": self.deepseek_config.temperature
        }

        return {**base_capabilities, **deepseek_capabilities}

    def switch_model_version(self, version: str) -> None:
        """
        切换模型版本

        Args:
            version: 模型版本 ("v3" | "v4")
        """
        if version.lower() == "v3":
            self.config = AdapterConfig(adapter_type=AdapterType.DEEPSEEK_V3)
            self.deepseek_config.model_name = "deepseek-ai/DeepSeek-V3"
        elif version.lower() == "v4":
            self.config = AdapterConfig(adapter_type=AdapterType.DEEPSEEK_V4)
            self.deepseek_config.model_name = "deepseek-ai/DeepSeek-V4"
        else:
            raise ValueError(f"不支持的模型版本: {version}")

        self._init_api_client()
