# 通用模型接口设计文档

## 1. 概述

### 1.1 设计目标

设计一套统一的模型接口（`BaseModelInterface`），抽象不同模型后端的差异，为 EnlightenLM 提供标准化的模型加载、推理和释放接口。

### 1.2 设计原则

- **可扩展性**：新模型后端只需实现接口即可接入
- **一致性**：所有模型后端提供统一的API调用方式
- **解耦性**：业务逻辑与具体模型实现分离
- **资源管理**：统一的资源加载和释放机制

### 1.3 支持的模型后端

| 后端类型 | 说明 | 实现类 |
|---------|------|--------|
| Ollama | 本地 LLM 推理服务 | `OllamaModel` |
| DeepSeek | DeepSeek API | `DeepSeekModel` |
| HuggingFace | HF Transformers 本地模型 | `HuggingFaceModel` |
| vLLM | vLLM 高性能推理 | `VLLMModel` |
| OpenAI | OpenAI 兼容 API | `OpenAIModel` |

---

## 2. 接口层次结构

```
BaseModelInterface (抽象基类)
├── OllamaModel
├── DeepSeekModel
├── HuggingFaceModel
├── VLLMModel
└── OpenAIModel
```

---

## 3. 核心接口定义

### 3.1 BaseModelInterface (抽象基类)

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any, AsyncIterator, Dict, Iterator, List, Optional,
    Tuple, Union
)
import torch


class ModelProvider(Enum):
    """模型提供者枚举"""
    OLLAMA = "ollama"
    DEEPSEEK = "deepseek"
    HUGGINGFACE = "huggingface"
    VLLM = "vllm"
    OPENAI = "openai"
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
    """
    通用模型接口抽象基类

    定义模型加载、推理和释放的标准接口。所有模型后端需实现此接口。

    方法分类:
    - 生命周期管理: load, unload, reload
    - 推理接口: generate, generate_stream, chat
    - 状态查询: is_available, get_state, get_metadata
    - 资源管理: get_device, get_memory_usage, cleanup
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化模型接口

        Args:
            model_path: 模型路径或名称
            device: 运行设备 ("cuda", "cpu", "mps")
            config: 模型特定配置
        """
        self.model_path = model_path
        self.device = device or self._get_default_device()
        self.config = config or {}
        self._state = ModelState.UNINITIALIZED
        self._metadata: Optional[ModelMetadata] = None
        self._model: Any = None

    # ==================== 生命周期管理 ====================

    @abstractmethod
    def load(self, **kwargs) -> bool:
        """
        加载模型

        根据模型后端类型执行相应的加载逻辑:
        - 本地模型: 加载权重到指定设备
        - API 模型: 初始化客户端连接

        Args:
            **kwargs: 后端特定参数

        Returns:
            bool: 加载是否成功

        Raises:
            ModelLoadError: 加载失败时抛出
        """
        pass

    @abstractmethod
    def unload(self) -> None:
        """
        卸载模型

        释放模型占用的资源，包括:
        - GPU 显存
        - CPU 内存
        - API 连接
        """
        pass

    def reload(self, **kwargs) -> bool:
        """
        重新加载模型

        先卸载再加载，用于模型更新或配置变更。

        Args:
            **kwargs: 传递给 load() 的参数

        Returns:
            bool: 重新加载是否成功
        """
        self.unload()
        return self.load(**kwargs)

    # ==================== 推理接口 ====================

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
        """
        生成文本（非流式）

        Args:
            prompt: 输入提示（字符串或 token IDs）
            max_tokens: 最大生成 token 数
            temperature: 采样温度 (0.0-2.0)
            top_p: Nucleus 采样概率
            stop: 停止词列表
            **kwargs: 后端特定参数

        Returns:
            str: 生成的文本
        """
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
        """
        生成文本（流式）

        Args:
            prompt: 输入提示
            max_tokens: 最大生成 token 数
            temperature: 采样温度
            top_p: Nucleus 采样概率
            stop: 停止词列表
            **kwargs: 后端特定参数

        Yields:
            str: 生成的文本片段
        """
        pass

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        聊天模式推理

        默认实现将 messages 转换为 prompt 调用 generate()。
        子类可重写以提供更高效的聊天实现。

        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}]
            max_tokens: 最大生成 token 数
            temperature: 采样温度
            **kwargs: 后端特定参数

        Returns:
            str: 生成的回复
        """
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
        """
        聊天模式推理（流式）

        默认实现将 messages 转换为 prompt 调用 generate_stream()。

        Args:
            messages: 消息列表
            max_tokens: 最大生成 token 数
            temperature: 采样温度
            **kwargs: 后端特定参数

        Yields:
            str: 生成的回复片段
        """
        prompt = self._format_chat_messages(messages)
        yield from self.generate_stream(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )

    # ==================== 状态查询 ====================

    def is_available(self) -> bool:
        """
        检查模型是否可用

        Returns:
            bool: 模型是否已加载且可用
        """
        return self._state == ModelState.READY

    def get_state(self) -> ModelState:
        """
        获取模型当前状态

        Returns:
            ModelState: 当前状态
        """
        return self._state

    def get_metadata(self) -> Optional[ModelMetadata]:
        """
        获取模型元数据

        Returns:
            ModelMetadata: 模型元数据，加载前返回 None
        """
        return self._metadata

    @abstractmethod
    def get_capabilities(self) -> ModelCapability:
        """
        获取模型能力

        Returns:
            ModelCapability: 模型支持的能力
        """
        pass

    # ==================== 资源管理 ====================

    @abstractmethod
    def get_device(self) -> str:
        """
        获取模型运行设备

        Returns:
            str: 设备标识 ("cuda:0", "cpu", "mps:0")
        """
        pass

    @abstractmethod
    def get_memory_usage(self) -> Dict[str, float]:
        """
        获取内存使用情况

        Returns:
            Dict[str, float]: 内存使用统计 (单位: MB)
                - allocated: 已分配显存
                - reserved: 已预留显存
                - system_mem: 系统内存使用
        """
        pass

    def cleanup(self) -> None:
        """
        清理运行时资源

        不同于 unload()，cleanup() 只清理临时资源，
        模型保持加载状态。
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ==================== 工具方法 ====================

    def _get_default_device(self) -> str:
        """
        获取默认设备

        Returns:
            str: 默认设备
        """
        if torch.cuda.is_available():
            return "cuda:0"
        elif torch.backends.mps.is_available():
            return "mps:0"
        return "cpu"

    def _format_chat_messages(self, messages: List[Dict[str, str]]) -> str:
        """
        格式化聊天消息为 prompt

        子类可根据模型特性重写此方法。

        Args:
            messages: 消息列表

        Returns:
            str: 格式化的 prompt
        """
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
        """上下文管理器入口"""
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.unload()
        return False
```

---

## 4. 配置管理机制

### 4.1 ModelConfig 配置类

```python
@dataclass
class ModelConfig:
    """通用模型配置"""
    provider: ModelProvider
    model_path: str

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
class OllamaModelConfig(ModelConfig):
    """Ollama 特定配置"""
    provider: ModelProvider = ModelProvider.OLLAMA
    base_url: str = "http://localhost:11434"
    stream: bool = True
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HuggingFaceModelConfig(ModelConfig):
    """HuggingFace 特定配置"""
    provider: ModelProvider = ModelProvider.HUGGINGFACE
    revision: Optional[str] = None
    trust_remote_code: bool = False
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    use_flash_attention: bool = True
    device_map: Optional[str] = "auto"
```

### 4.2 ModelConfigManager 配置管理器

```python
class ModelConfigManager:
    """
    模型配置管理器

    负责:
    - 从 YAML/JSON 加载配置
    - 配置验证和合并
    - 运行时配置热更新
    - 环境变量覆盖
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置管理器

        Args:
            config_path: 配置文件路径
        """
        self._configs: Dict[str, ModelConfig] = {}
        self._active_config: Optional[str] = None
        self._listeners: List[Callable] = []

        if config_path:
            self.load_from_file(config_path)

    def load_from_file(self, path: str) -> None:
        """从文件加载配置"""
        import yaml
        import json
        from pathlib import Path

        path = Path(path)
        if path.suffix in [".yaml", ".yml"]:
            with open(path) as f:
                data = yaml.safe_load(f)
        elif path.suffix == ".json":
            with open(path) as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")

        self._parse_config_data(data)

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
```

---

## 5. 模型工厂机制

### 5.1 ModelFactory 模型工厂

```python
class ModelFactory:
    """
    模型工厂

    根据配置创建相应的模型实例，支持:
    - 自动后端检测
    - 插件式扩展
    - 配置验证
    """

    _registry: Dict[ModelProvider, Type[BaseModelInterface]] = {}

    @classmethod
    def register(
        cls,
        provider: ModelProvider
    ) -> Callable:
        """
        注册模型后端

        用法:
        ```python
        @ModelFactory.register(ModelProvider.CUSTOM)
        class CustomModel(BaseModelInterface):
            ...
        ```
        """
        def decorator(model_class: Type[BaseModelInterface]):
            cls._registry[provider] = model_class
            return model_class
        return decorator

    @classmethod
    def create(
        cls,
        config: Union[ModelConfig, Dict[str, Any]],
        **kwargs
    ) -> BaseModelInterface:
        """
        创建模型实例

        Args:
            config: 模型配置
            **kwargs: 额外参数

        Returns:
            BaseModelInterface: 模型实例
        """
        if isinstance(config, dict):
            config = cls._dict_to_config(config)

        model_class = cls._registry.get(config.provider)
        if model_class is None:
            raise ValueError(f"Unsupported provider: {config.provider}")

        return model_class(
            model_path=config.model_path,
            device=config.device,
            config=vars(config)
        )

    @classmethod
    def _dict_to_config(cls, data: Dict[str, Any]) -> ModelConfig:
        """将字典转换为 ModelConfig"""
        provider = ModelProvider(data.get("provider", "custom"))

        config_classes = {
            ModelProvider.OLLAMA: OllamaModelConfig,
            ModelProvider.HUGGINGFACE: HuggingFaceModelConfig,
        }

        config_class = config_classes.get(provider, ModelConfig)
        return config_class(**data)
```

---

## 6. 具体实现示例

### 6.1 OllamaModel 实现

```python
@ModelFactory.register(ModelProvider.OLLAMA)
class OllamaModel(BaseModelInterface):
    """Ollama 模型实现"""

    def __init__(
        self,
        model_path: str = "qwen2.5:14b",
        base_url: str = "http://localhost:11434",
        **kwargs
    ):
        super().__init__(model_path=model_path, **kwargs)

        self.base_url = base_url
        self._client = None

    def load(self, **kwargs) -> bool:
        """加载 Ollama 模型"""
        self._state = ModelState.LOADING

        try:
            import httpx

            self._client = httpx.Client(
                base_url=self.base_url,
                timeout=self.config.get("timeout", 60)
            )

            response = self._client.get("/api/tags")
            response.raise_for_status()

            self._metadata = ModelMetadata(
                name=self.model_path,
                provider=ModelProvider.OLLAMA,
                capabilities=ModelCapability(
                    supports_streaming=True,
                    max_context_length=8192
                )
            )

            self._state = ModelState.READY
            return True

        except Exception as e:
            self._state = ModelState.ERROR
            raise ModelLoadError(f"Ollama load failed: {e}") from e

    def unload(self) -> None:
        """卸载 Ollama 模型"""
        if self._client:
            self._client.close()
            self._client = None

        self._model = None
        self._metadata = None
        self._state = ModelState.UNLOADED

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """生成文本"""
        if not self.is_available():
            raise RuntimeError("Model not loaded")

        data = {
            "model": self.model_path,
            "prompt": prompt,
            "temperature": temperature or self.config.get("temperature", 0.7),
            "max_tokens": max_tokens or self.config.get("max_tokens", 2048),
            "stream": False
        }

        response = self._client.post("/api/generate", json=data)
        response.raise_for_status()

        result = ""
        for line in response.iter_lines():
            if line:
                import json
                chunk = json.loads(line)
                if "response" in chunk:
                    result += chunk["response"]
                if chunk.get("done"):
                    break

        return result

    def generate_stream(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> Iterator[str]:
        """流式生成文本"""
        if not self.is_available():
            raise RuntimeError("Model not loaded")

        data = {
            "model": self.model_path,
            "prompt": prompt,
            "temperature": temperature or self.config.get("temperature", 0.7),
            "max_tokens": max_tokens or self.config.get("max_tokens", 2048),
            "stream": True
        }

        response = self._client.post("/api/generate", json=data)
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                import json
                chunk = json.loads(line)
                if "response" in chunk:
                    yield chunk["response"]

    def get_device(self) -> str:
        """Ollama 运行在服务端，这里返回 CPU"""
        return "cpu"

    def get_memory_usage(self) -> Dict[str, float]:
        """获取内存使用"""
        return {
            "allocated": 0.0,
            "reserved": 0.0,
            "system_mem": 0.0
        }

    def get_capabilities(self) -> ModelCapability:
        """获取模型能力"""
        return ModelCapability(
            supports_streaming=True,
            max_context_length=8192,
            max_new_tokens=2048
        )


class ModelLoadError(Exception):
    """模型加载错误"""
    pass
```

### 6.2 HuggingFaceModel 实现骨架

```python
@ModelFactory.register(ModelProvider.HUGGINGFACE)
class HuggingFaceModel(BaseModelInterface):
    """HuggingFace Transformers 模型实现"""

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        **kwargs
    ):
        super().__init__(model_path=model_path, device=device, **kwargs)

        self._pipeline = None
        self._tokenizer = None
        self._model = None

    def load(self, **kwargs) -> bool:
        """加载 HuggingFace 模型"""
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self._state = ModelState.LOADING

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=self.config.get("trust_remote_code", False)
            )

            load_in_8bit = self.config.get("load_in_8bit", False)
            load_in_4bit = self.config.get("load_in_4bit", False)

            if load_in_8bit or load_in_4bit:
                from transformers import BitsAndBytesConfig

                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=load_in_8bit,
                    load_in_4bit=load_in_4bit
                )
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    quantization_config=quantization_config,
                    device_map=self.device,
                    trust_remote_code=self.config.get("trust_remote_code", False)
                )
            else:
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    device_map=self.device,
                    trust_remote_code=self.config.get("trust_remote_code", False)
                )

            self._metadata = self._extract_metadata()
            self._state = ModelState.READY
            return True

        except Exception as e:
            self._state = ModelState.ERROR
            raise ModelLoadError(f"HF load failed: {e}") from e

    def generate(self, prompt: str, **kwargs) -> str:
        """生成文本"""
        if not self.is_available():
            raise RuntimeError("Model not loaded")

        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        outputs = self._model.generate(**inputs, **kwargs)
        return self._tokenizer.decode(outputs[0], skip_special_tokens=True)

    def get_device(self) -> str:
        """获取模型设备"""
        return str(self._model.device) if self._model else self.device

    def get_memory_usage(self) -> Dict[str, float]:
        """获取内存使用"""
        import torch

        if not torch.cuda.is_available():
            return {"allocated": 0, "reserved": 0, "system_mem": 0}

        return {
            "allocated": torch.cuda.memory_allocated() / 1024**2,
            "reserved": torch.cuda.memory_reserved() / 1024**2,
            "system_mem": 0
        }

    def get_capabilities(self) -> ModelCapability:
        """获取模型能力"""
        return ModelCapability(
            supports_streaming=False,
            supports_batch_inference=True,
            max_context_length=self._metadata.context_length if self._metadata else 4096
        )

    def _extract_metadata(self) -> ModelMetadata:
        """提取模型元数据"""
        config = self._model.config
        return ModelMetadata(
            name=self.model_path,
            provider=ModelProvider.HUGGINGFACE,
            context_length=getattr(config, "max_position_embeddings", 4096),
            vocab_size=getattr(config, "vocab_size", 0),
            hidden_size=getattr(config, "hidden_size", 0),
            num_layers=getattr(config, "num_hidden_layers", 0),
            num_heads=getattr(config, "num_attention_heads", 0),
            capabilities=self.get_capabilities()
        )
```

---

## 7. 使用示例

### 7.1 基础用法

```python
from enlighten.interfaces import ModelFactory, ModelConfig, ModelProvider

config = ModelConfig(
    provider=ModelProvider.OLLAMA,
    model_path="qwen2.5:14b",
    base_url="http://localhost:11434"
)

model = ModelFactory.create(config)

with model:
    response = model.generate("你好，请介绍一下自己")
    print(response)

    for chunk in model.generate_stream("写一个故事"):
        print(chunk, end="")
```

### 7.2 配置热更新

```python
manager = ModelConfigManager("configs/models.yaml")

manager.add_listener(lambda old, new, cfg: print(f"Config changed: {old} -> {new}"))

manager.set_active_config("ollama_qwen")
model = ModelFactory.create(manager.get_config("ollama_qwen"))
```

### 7.3 自定义后端注册

```python
from enlighten.interfaces import BaseModelInterface, ModelFactory, ModelProvider

@ModelFactory.register(ModelProvider.CUSTOM)
class CustomModel(BaseModelInterface):
    def load(self) -> bool:
        # 实现加载逻辑
        pass

    def generate(self, prompt: str, **kwargs) -> str:
        # 实现生成逻辑
        pass

    # ... 实现其他抽象方法
```

---

## 8. 文件结构

```
enlighten/
├── interfaces/
│   ├── __init__.py
│   ├── base.py              # BaseModelInterface 定义
│   ├── config.py            # ModelConfig 相关类
│   ├── factory.py           # ModelFactory
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── ollama.py        # OllamaModel
│   │   ├── deepseek.py     # DeepSeekModel
│   │   ├── huggingface.py  # HuggingFaceModel
│   │   ├── vllm.py          # VLLMModel
│   │   └── openai.py        # OpenAIModel
│   └── errors.py            # 自定义异常
└── adapters/                 # 底层适配器（保持现有结构）
    ├── base.py
    └── ...
```

---

## 9. 与现有代码的集成

### 9.1 兼容现有 AdapterBase

`BaseModelInterface` 与现有的 `AdapterBase` 是不同层次的抽象：

| 层次 | 类 | 职责 |
|------|-----|------|
| 上层 | BaseModelInterface | 模型加载、推理、API 统一 |
| 下层 | AdapterBase | 底层前向传播、注意力计算 |

两者可以结合使用：

```python
class HybridModel(BaseModelInterface):
    """混合模型：上层 API + 下层适配器"""

    def __init__(self, adapter: AdapterBase, **kwargs):
        super().__init__(**kwargs)
        self.adapter = adapter

    def generate(self, prompt: str, **kwargs) -> str:
        # 使用 adapter 进行推理
        pass
```

### 9.2 迁移计划

1. **Phase 1**: 创建 `enlighten/interfaces/` 目录，实现基础接口
2. **Phase 2**: 迁移 OllamaAPIClient 到 OllamaModel
3. **Phase 3**: 实现 HuggingFaceModel 和 VLLMModel
4. **Phase 4**: 更新 EnlightenLM 主代码使用新接口

---

## 10. 后续扩展

- [ ] 支持异步推理接口 (`agenerate`, `achat`)
- [ ] 支持分布式推理
- [ ] 支持模型版本管理
- [ ] 支持模型量化配置
- [ ] 添加性能监控和指标收集
