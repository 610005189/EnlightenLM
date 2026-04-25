"""模型工厂"""

from typing import Any, Dict, Type, Union

from enlighten.interfaces.base import BaseModelInterface, ModelProvider
from enlighten.interfaces.config import ModelConfig


class ModelFactoryError(Exception):
    """模型工厂错误"""

    pass


class ModelFactory:
    """模型工厂"""

    _registry: Dict[ModelProvider, Type[BaseModelInterface]] = {}

    @classmethod
    def register(cls, provider: ModelProvider) -> callable:
        """注册模型后端"""

        def decorator(model_class: Type[BaseModelInterface]):
            cls._registry[provider] = model_class
            return model_class

        return decorator

    @classmethod
    def create(cls, config: Union[ModelConfig, Dict[str, Any]], **kwargs) -> BaseModelInterface:
        """创建模型实例"""
        if isinstance(config, dict):
            config = cls._dict_to_config(config)

        model_class = cls._registry.get(config.provider)
        if model_class is None:
            raise ModelFactoryError(f"Unsupported provider: {config.provider}")

        model_config = vars(config) if isinstance(config, ModelConfig) else config
        return model_class(model_path=config.model_path, device=config.device, config=model_config)

    @classmethod
    def _dict_to_config(cls, data: Dict[str, Any]) -> ModelConfig:
        """将字典转换为 ModelConfig"""
        from enlighten.interfaces.config import (
            ClaudeModelConfig,
            LLaMAModelConfig,
            MistralModelConfig,
            OllamaModelConfig,
        )

        provider = ModelProvider(data.get("provider", "custom"))

        config_classes = {
            ModelProvider.LLAMA: LLaMAModelConfig,
            ModelProvider.MISTRAL: MistralModelConfig,
            ModelProvider.OLLAMA: OllamaModelConfig,
            ModelProvider.CLAUDE: ClaudeModelConfig,
        }

        config_class = config_classes.get(provider, ModelConfig)
        return config_class(**data)

    @classmethod
    def get_registered_providers(cls) -> list:
        """获取已注册的提供者列表"""
        return list(cls._registry.keys())
