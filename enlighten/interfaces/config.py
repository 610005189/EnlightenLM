"""模型配置管理"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import yaml

from enlighten.interfaces.base import ModelProvider


@dataclass
class ModelConfig:
    """通用模型配置"""

    provider: Optional[ModelProvider] = None
    model_path: Optional[str] = None
    device: str = "auto"
    dtype: str = "auto"
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = -1
    stop: List[str] = field(default_factory=list)
    timeout: int = 60
    retry_count: int = 3
    extra_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLaMAModelConfig(ModelConfig):
    """LLaMA 特定配置"""

    base_url: Optional[str] = None
    revision: Optional[str] = None
    trust_remote_code: bool = False
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    use_flash_attention: bool = True
    device_map: Optional[str] = "auto"
    max_seq_len: int = 2048

    def __post_init__(self):
        if self.provider is None:
            self.provider = ModelProvider.LLAMA
        elif isinstance(self.provider, str):
            self.provider = ModelProvider(self.provider)


@dataclass
class OllamaModelConfig(ModelConfig):
    """Ollama 特定配置"""

    base_url: str = "http://localhost:11434"
    stream: bool = True
    options: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.provider is None:
            self.provider = ModelProvider.OLLAMA
        elif isinstance(self.provider, str):
            self.provider = ModelProvider(self.provider)


@dataclass
class MistralModelConfig(ModelConfig):
    """Mistral 特定配置"""

    revision: Optional[str] = None
    trust_remote_code: bool = False
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    use_flash_attention: bool = True
    device_map: Optional[str] = "auto"
    max_seq_len: int = 4096

    def __post_init__(self):
        if self.provider is None:
            self.provider = ModelProvider.MISTRAL
        elif isinstance(self.provider, str):
            self.provider = ModelProvider(self.provider)


@dataclass
class ClaudeModelConfig(ModelConfig):
    """Claude 特定配置

    Anthropic Claude API 配置参数。
    """

    api_key: Optional[str] = None
    base_url: str = "https://api.anthropic.com"
    api_version: str = "2023-06-01"
    stream: bool = True
    max_retries: int = 3
    timeout: int = 120

    def __post_init__(self):
        if self.provider is None:
            self.provider = ModelProvider.CLAUDE
        elif isinstance(self.provider, str):
            self.provider = ModelProvider(self.provider)


class ModelConfigManager:
    """模型配置管理器"""

    def __init__(self, config_path: Optional[str] = None):
        self._configs: Dict[str, ModelConfig] = {}
        self._active_config: Optional[str] = None
        self._listeners: List[Callable] = []

        if config_path:
            self.load_from_file(config_path)

    def load_from_file(self, path: str) -> None:
        """从文件加载配置"""
        path = Path(path)
        if path.suffix in [".yaml", ".yml"]:
            with open(path) as f:
                data = yaml.safe_load(f)
        elif path.suffix == ".json":
            import json

            with open(path) as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")

        self._parse_config_data(data)

    def _parse_config_data(self, data: Dict[str, Any]) -> None:
        """解析配置数据"""
        if not isinstance(data, dict):
            return

        for name, config_data in data.items():
            if isinstance(config_data, dict):
                provider_str = config_data.get("provider", "custom")
                provider = (
                    ModelProvider(provider_str) if isinstance(provider_str, str) else provider_str
                )

                config_classes = {
                    ModelProvider.LLAMA: LLaMAModelConfig,
                    ModelProvider.OLLAMA: OllamaModelConfig,
                }

                config_class = config_classes.get(provider, ModelConfig)
                self._configs[name] = config_class(**config_data)

    def register_config(self, name: str, config: ModelConfig) -> None:
        """注册配置"""
        self._configs[name] = config

    def get_config(self, name: str) -> Optional[ModelConfig]:
        """获取配置"""
        return self._configs.get(name)

    def set_active_config(self, name: str) -> None:
        """设置活跃配置"""
        if name not in self._configs:
            raise KeyError(f"Config not found: {name}")

        old_config = self._active_config
        self._active_config = name

        for listener in self._listeners:
            listener(old_config, name, self._configs[name])

    def add_listener(self, listener: Callable) -> None:
        """添加配置变更监听器"""
        self._listeners.append(listener)

    def reload(self) -> None:
        """重新加载配置"""
        pass
