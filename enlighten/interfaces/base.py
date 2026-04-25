"""模型接口基础定义"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Union

import torch


class ModelProvider(Enum):
    """模型提供者枚举"""
    OLLAMA = "ollama"
    DEEPSEEK = "deepseek"
    HUGGINGFACE = "huggingface"
    VLLM = "vllm"
    OPENAI = "openai"
    LLAMA = "llama"
    MISTRAL = "mistral"
    CLAUDE = "claude"
    CUSTOM = "custom"


class ModelState(Enum):
    """模型状态枚举"""
    UNINITIALIZED = "uninitialized"
    LOADING = "loading"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    UNLOADED = "unloaded"


@dataclass
class ModelCapability:
    """模型能力描述"""
    supports_streaming: bool = False
    supports_function_calling: bool = False
    supports_vision: bool = False
    supports_json_mode: bool = False
    max_context_length: int = 4096
    max_new_tokens: int = 2048
    supports_batch_inference: bool = True
    supports_prefix_caching: bool = False


@dataclass
class ModelMetadata:
    """模型元数据"""
    name: str
    provider: ModelProvider
    model_path: Optional[str] = None
    revision: Optional[str] = None
    quantize: Optional[str] = None
    context_length: int = 4096
    vocab_size: int = 0
    hidden_size: int = 0
    num_layers: int = 0
    num_heads: int = 0
    head_dim: int = 0
    capabilities: ModelCapability = field(default_factory=ModelCapability)


class BaseModelInterface(ABC):
    """通用模型接口抽象基类"""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.model_path = model_path
        self.device = device or self._get_default_device()
        self.config = config or {}
        self._state = ModelState.UNINITIALIZED
        self._metadata: Optional[ModelMetadata] = None
        self._model: Any = None

    @abstractmethod
    def load(self, **kwargs) -> bool:
        pass

    @abstractmethod
    def unload(self) -> None:
        pass

    def reload(self, **kwargs) -> bool:
        self.unload()
        return self.load(**kwargs)

    @abstractmethod
    def generate(
        self,
        prompt: Union[str, List[int]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None,
        **kwargs
    ) -> str:
        pass

    @abstractmethod
    def generate_stream(
        self,
        prompt: Union[str, List[int]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None,
        **kwargs
    ) -> Iterator[str]:
        pass

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        prompt = self._format_chat_messages(messages)
        return self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )

    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> Iterator[str]:
        prompt = self._format_chat_messages(messages)
        yield from self.generate_stream(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )

    def is_available(self) -> bool:
        return self._state == ModelState.READY

    def get_state(self) -> ModelState:
        return self._state

    def get_metadata(self) -> Optional[ModelMetadata]:
        return self._metadata

    @abstractmethod
    def get_capabilities(self) -> ModelCapability:
        pass

    @abstractmethod
    def get_device(self) -> str:
        pass

    @abstractmethod
    def get_memory_usage(self) -> Dict[str, float]:
        pass

    def cleanup(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _get_default_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda:0"
        elif torch.backends.mps.is_available():
            return "mps:0"
        return "cpu"

    def _format_chat_messages(self, messages: List[Dict[str, str]]) -> str:
        formatted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                formatted.append(f"System: {content}")
            elif role == "user":
                formatted.append(f"User: {content}")
            elif role == "assistant":
                formatted.append(f"Assistant: {content}")
        formatted.append("Assistant:")
        return "\n".join(formatted)

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload()
        return False