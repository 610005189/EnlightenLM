"""
配置模式系统
定义 full/balanced/lightweight 三种预设模式
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


class EnlightenMode(Enum):
    """EnlightenLM 运行模式枚举"""
    FULL = "full"
    BALANCED = "balanced"
    LIGHTWEIGHT = "lightweight"
    CUSTOM = "custom"


@dataclass
class VANStreamConfig:
    """VAN 流配置"""
    level: str = "medium"  # "light" | "medium" | "full"


@dataclass
class WorkingMemoryConfig:
    """工作记忆配置"""
    capacity: int = 512
    refresh_interval: int = 32
    use_topk_refresh: bool = True
    eviction_policy: str = "lru"
    attention_size: int = 32


@dataclass
class EntropyMonitorConfig:
    """熵监控配置"""
    window_size: int = 20


@dataclass
class CutoffConfig:
    """截断配置"""
    low_entropy_threshold: float = 0.5
    low_variance_threshold: float = 0.05
    min_duration: int = 5
    van_threshold: float = 0.9
    cooldown_steps: int = 10


@dataclass
class AsyncReviewConfig:
    """异步审核配置"""
    enabled: bool = False
    model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    interval: int = 32


@dataclass
class ModelProviderConfig:
    """模型提供者配置"""
    use_local_model: bool = False
    local_model_name: str = "distilgpt2"
    local_model_path: Optional[str] = None
    api_provider: str = "deepseek"
    api_model: str = "deepseek-chat"
    device: str = "auto"
    local_max_length: int = 1024
    local_temperature: float = 0.7


@dataclass
class ModeConfig:
    """
    模式配置

    Attributes:
        mode: 运行模式
        van_level: VAN 漏斗级别 ("light" | "medium" | "full")
        gate_fusion: 是否启用门控融合
        dmn_noise: 是否启用 DMN 噪声抑制
        working_memory: 工作记忆配置
        entropy_monitor: 熵监控配置
        cutoff: 截断配置
        async_review: 异步审核配置
        model_provider: 模型提供者配置 (本地模型 vs API)
    """
    mode: EnlightenMode = EnlightenMode.BALANCED
    van_level: str = "medium"
    gate_fusion: bool = True
    dmn_noise: bool = False
    working_memory: WorkingMemoryConfig = field(default_factory=WorkingMemoryConfig)
    entropy_monitor: EntropyMonitorConfig = field(default_factory=EntropyMonitorConfig)
    cutoff: CutoffConfig = field(default_factory=CutoffConfig)
    async_review: AsyncReviewConfig = field(default_factory=AsyncReviewConfig)
    model_provider: ModelProviderConfig = field(default_factory=ModelProviderConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModeConfig':
        """从字典创建配置"""
        mode = EnlightenMode(data.get('mode', 'balanced'))

        van_level = data.get('van_level', 'medium')
        gate_fusion = data.get('gate_fusion', True)
        dmn_noise = data.get('dmn_noise', False)

        wm_data = data.get('working_memory', {})
        working_memory = WorkingMemoryConfig(**wm_data) if wm_data else WorkingMemoryConfig()

        em_data = data.get('entropy_monitor', {})
        entropy_monitor = EntropyMonitorConfig(**em_data) if em_data else EntropyMonitorConfig()

        cutoff_data = data.get('cutoff', {})
        cutoff = CutoffConfig(**cutoff_data) if cutoff_data else CutoffConfig()

        ar_data = data.get('async_review', {})
        async_review = AsyncReviewConfig(**ar_data) if ar_data else AsyncReviewConfig()

        mp_data = data.get('model_provider', {})
        model_provider = ModelProviderConfig(**mp_data) if mp_data else ModelProviderConfig()

        return cls(
            mode=mode,
            van_level=van_level,
            gate_fusion=gate_fusion,
            dmn_noise=dmn_noise,
            working_memory=working_memory,
            entropy_monitor=entropy_monitor,
            cutoff=cutoff,
            async_review=async_review,
            model_provider=model_provider
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'mode': self.mode.value,
            'van_level': self.van_level,
            'gate_fusion': self.gate_fusion,
            'dmn_noise': self.dmn_noise,
            'working_memory': {
                'capacity': self.working_memory.capacity,
                'refresh_interval': self.working_memory.refresh_interval,
                'use_topk_refresh': self.working_memory.use_topk_refresh,
                'eviction_policy': self.working_memory.eviction_policy
            },
            'entropy_monitor': {
                'window_size': self.entropy_monitor.window_size
            },
            'cutoff': {
                'low_entropy_threshold': self.cutoff.low_entropy_threshold,
                'low_variance_threshold': self.cutoff.low_variance_threshold,
                'min_duration': self.cutoff.min_duration,
                'van_threshold': self.cutoff.van_threshold,
                'cooldown_steps': self.cutoff.cooldown_steps
            },
            'async_review': {
                'enabled': self.async_review.enabled,
                'model': self.async_review.model,
                'interval': self.async_review.interval
            },
            'model_provider': {
                'use_local_model': self.model_provider.use_local_model,
                'local_model_name': self.model_provider.local_model_name,
                'local_model_path': self.model_provider.local_model_path,
                'api_provider': self.model_provider.api_provider,
                'api_model': self.model_provider.api_model,
                'device': self.model_provider.device,
                'local_max_length': self.model_provider.local_max_length,
                'local_temperature': self.model_provider.local_temperature
            }
        }


MODE_PRESETS: Dict[str, ModeConfig] = {
    "full": ModeConfig(
        mode=EnlightenMode.FULL,
        van_level="full",
        gate_fusion=True,
        dmn_noise=True,
        working_memory=WorkingMemoryConfig(
            capacity=512,
            refresh_interval=32,
            use_topk_refresh=True
        ),
        cutoff=CutoffConfig(
            low_entropy_threshold=0.5,
            low_variance_threshold=0.05,
            min_duration=5,
            van_threshold=0.9,
            cooldown_steps=10
        ),
        async_review=AsyncReviewConfig(
            enabled=True,
            model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            interval=32
        )
    ),

    "balanced": ModeConfig(
        mode=EnlightenMode.BALANCED,
        van_level="medium",
        gate_fusion=True,
        dmn_noise=False,
        working_memory=WorkingMemoryConfig(
            capacity=512,
            refresh_interval=32,
            use_topk_refresh=True
        ),
        cutoff=CutoffConfig(
            low_entropy_threshold=0.5,
            low_variance_threshold=0.05,
            min_duration=5,
            van_threshold=0.9,
            cooldown_steps=10
        ),
        async_review=AsyncReviewConfig(
            enabled=False
        )
    ),

    "lightweight": ModeConfig(
        mode=EnlightenMode.LIGHTWEIGHT,
        van_level="light",
        gate_fusion=False,
        dmn_noise=False,
        working_memory=WorkingMemoryConfig(
            capacity=256,
            refresh_interval=0,
            use_topk_refresh=False
        ),
        cutoff=CutoffConfig(
            low_entropy_threshold=0.5,
            low_variance_threshold=0.05,
            min_duration=5,
            van_threshold=0.9,
            cooldown_steps=5
        ),
        async_review=AsyncReviewConfig(
            enabled=False
        )
    )
}


def get_mode_preset(mode: str) -> ModeConfig:
    """
    获取预设模式配置

    Args:
        mode: 模式名称 ("full" | "balanced" | "lightweight")

    Returns:
        ModeConfig: 模式配置

    Raises:
        ValueError: 当模式名称无效时
    """
    if mode not in MODE_PRESETS:
        raise ValueError(f"Unknown mode: {mode}. Available modes: {list(MODE_PRESETS.keys())}")
    return MODE_PRESETS[mode]


def get_mode_from_env() -> Optional[str]:
    """
    从环境变量获取模式

    Returns:
        str: 模式名称，如果未设置则返回 None
    """
    import os
    return os.environ.get("ENLIGHTEN_MODE")
