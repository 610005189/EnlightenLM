"""
vLLM Adapter - vLLM 高性能推理适配器
将 EnlightenLM 的 L1/L2/L3 组件集成到 vLLM 流水线
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass

from .base import AdapterBase, AdapterConfig, AdapterType, MemoryEfficientAttentionMixin


@dataclass
class VLLMConfig:
    """vLLM 适配器配置"""
    max_model_len: int = 4096
    max_num_seqs: int = 256
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    block_size: int = 16
    enable_prefix_caching: bool = True


class VLLMKVCacheManager:
    """
    vLLM KV 缓存管理器

    负责管理 EnlightenLM 特有的稀疏 KV 缓存
    集成到 vLLM 的 PagedAttention 机制中
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        max_block: int = 1024,
        block_size: int = 16
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_block = max_block
        self.block_size = block_size

        self.cache = {
            "keys": torch.zeros(max_block, block_size, num_heads, head_dim),
            "values": torch.zeros(max_block, block_size, num_heads, head_dim)
        }

        self.block_mapping = {}
        self.num_allocated_blocks = 0

    def allocate_block(self) -> int:
        """分配一个新的缓存块"""
        if self.num_allocated_blocks >= self.max_block:
            raise RuntimeError("KV 缓存块已耗尽")

        block_id = self.num_allocated_blocks
        self.num_allocated_blocks += 1
        return block_id

    def free_block(self, block_id: int) -> None:
        """释放缓存块"""
        if block_id in self.block_mapping:
            del self.block_mapping[block_id]

    def get_block(
        self,
        block_ids: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取指定块的 KV 缓存"""
        keys = self.cache["keys"][block_ids]
        values = self.cache["values"][block_ids]
        return keys, values

    def update_block(
        self,
        block_id: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        position: int
    ) -> None:
        """更新指定块的 KV 缓存"""
        block_offset = position % self.block_size
        self.cache["keys"][block_id, block_offset] = keys
        self.cache["values"][block_id, block_offset] = values


class VLLMAdapter(AdapterBase, MemoryEfficientAttentionMixin):
    """
    vLLM 适配器

    将 EnlightenLM 的三层架构集成到 vLLM 高性能推理流水线

    特性:
    - PagedAttention 内存管理
    - Flash Attention 加速
    - 稀疏注意力支持
    - 动态批处理
    """

    def __init__(
        self,
        model: nn.Module,
        config: AdapterConfig,
        vllm_config: Optional[VLLMConfig] = None
    ):
        super().__init__(config)

        self.model = model
        self.vllm_config = vllm_config or VLLMConfig()

        self._init_kv_cache_manager()

        self.prefix_caching_enabled = self.vllm_config.enable_prefix_caching
        self.prefix_hashes = {}

    def _init_kv_cache_manager(self) -> None:
        """初始化 KV 缓存管理器"""
        try:
            from transformers import AutoConfig
            hf_config = AutoConfig.from_pretrained(self.model.name_or_path)

            num_layers = getattr(hf_config, "num_hidden_layers", 12)
            num_heads = getattr(hf_config, "num_attention_heads", 12)
            head_dim = getattr(hf_config, "hidden_size", 768) // num_heads

            self.kv_cache_manager = VLLMKVCacheManager(
                num_layers=num_layers,
                num_heads=num_heads,
                head_dim=head_dim,
                max_block=1024,
                block_size=self.vllm_config.block_size
            )
        except Exception:
            self.kv_cache_manager = None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        vLLM 风格的前向传播

        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        return outputs.logits

    def get_kv_cache(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取当前层的 KV 缓存"""
        if self.kv_cache_manager is None:
            return torch.zeros(1), torch.zeros(1)

        return self.kv_cache_manager.cache["keys"][:self.kv_cache_manager.num_allocated_blocks], \
               self.kv_cache_manager.cache["values"][:self.kv_cache_manager.num_allocated_blocks]

    def update_kv_cache(
        self,
        keys: torch.Tensor,
        values: torch.Tensor
    ) -> None:
        """更新 KV 缓存"""
        if self.kv_cache_manager is None:
            return

        block_id = self.kv_cache_manager.allocate_block()
        self.kv_cache_manager.update_block(block_id, keys, values, position=0)

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

        只计算指定索引位置的注意力，用于 EnlightenLM 的 L2 工作记忆

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
        """检查 vLLM 是否可用"""
        try:
            import vllm
            return True
        except ImportError:
            return False

    def get_capabilities(self) -> Dict[str, Any]:
        """获取 vLLM 适配器能力"""
        base_capabilities = super().get_capabilities()

        vllm_capabilities = {
            "type": "vllm",
            "supports_paged_attention": True,
            "supports_flash_attention": True,
            "supports_prefix_caching": self.prefix_caching_enabled,
            "max_model_len": self.vllm_config.max_model_len,
            "max_num_seqs": self.vllm_config.max_num_seqs,
            "gpu_memory_utilization": self.vllm_config.gpu_memory_utilization,
            "block_size": self.vllm_config.block_size
        }

        return {**base_capabilities, **vllm_capabilities}


class VLLMInferenceEngine:
    """
    vLLM 推理引擎

    封装 vLLM 的 LLM 推理接口，提供 EnlightenLM 特有的功能
    """

    def __init__(
        self,
        model_name: str,
        adapter: VLLMAdapter,
        device: str = "cuda"
    ):
        self.model_name = model_name
        self.adapter = adapter
        self.device = device
        self.llm = None

    def initialize(self) -> None:
        """初始化 vLLM 推理引擎"""
        try:
            from vllm import LLM, SamplingParams

            self.llm = LLM(
                model=self.model_name,
                tensor_parallel_size=self.adapter.vllm_config.tensor_parallel_size,
                gpu_memory_utilization=self.adapter.vllm_config.gpu_memory_utilization,
                max_model_len=self.adapter.vllm_config.max_model_len,
                enable_prefix_caching=self.adapter.vllm_config.enable_prefix_caching
            )
            self.SamplingParams = SamplingParams
        except ImportError:
            raise ImportError("vLLM 未安装，请运行: pip install vllm")

    def generate(
        self,
        prompts: List[str],
        temperature: float = 0.7,
        max_tokens: int = 256,
        **kwargs
    ) -> List[str]:
        """
        生成文本

        Args:
            prompts: 提示列表
            temperature: 温度参数
            max_tokens: 最大 token 数
            **kwargs: 其他采样参数

        Returns:
            生成的文本列表
        """
        if self.llm is None:
            self.initialize()

        sampling_params = self.SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

        outputs = self.llm.generate(prompts, sampling_params)

        return [output.outputs[0].text for output in outputs]

    def get_adapter(self) -> VLLMAdapter:
        """获取适配器"""
        return self.adapter
