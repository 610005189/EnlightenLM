"""Ollama 模型实现

与现有 OllamaAPIClient 集成的 Ollama 模型支持。
"""

from typing import Any, Dict, Iterator, List, Optional, Union

import httpx

from enlighten.interfaces.base import (
    BaseModelInterface,
    ModelCapability,
    ModelMetadata,
    ModelProvider,
    ModelState,
)
from enlighten.interfaces.config import OllamaModelConfig
from enlighten.interfaces.errors import ModelLoadError
from enlighten.interfaces.factory import ModelFactory


@ModelFactory.register(ModelProvider.OLLAMA)
class OllamaModel(BaseModelInterface):
    """Ollama 模型实现

    与本地 Ollama 服务交互，提供模型推理功能。
    兼容现有 OllamaAPIClient 的使用方式。
    """

    def __init__(
        self,
        model_path: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__(model_path=model_path, device="cpu", config=config or {})

        self.base_url = base_url
        self._client: Optional[httpx.Client] = None
        self._stream = self.config.get("stream", True)

    def load(self, **kwargs) -> bool:
        """加载 Ollama 模型"""
        self._state = ModelState.LOADING

        try:
            self._client = httpx.Client(
                base_url=self.base_url,
                timeout=self.config.get("timeout", 60)
            )

            response = self._client.get("/api/tags")
            response.raise_for_status()

            available_models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in available_models]

            if self.model_path not in model_names:
                pass

            self._metadata = ModelMetadata(
                name=self.model_path,
                provider=ModelProvider.OLLAMA,
                model_path=self.model_path,
                capabilities=ModelCapability(
                    supports_streaming=True,
                    max_context_length=8192,
                    max_new_tokens=self.config.get("max_tokens", 2048)
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
        prompt: Union[str, List[int]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None,
        **kwargs
    ) -> str:
        """生成文本（非流式）"""
        if not self.is_available():
            raise RuntimeError("Model not loaded")

        max_tokens = max_tokens or self.config.get("max_tokens", 2048)
        temperature = temperature or self.config.get("temperature", 0.7)
        top_p = top_p or self.config.get("top_p", 1.0)

        data = {
            "model": self.model_path,
            "prompt": prompt if isinstance(prompt, str) else self._decode_tokens(prompt),
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": False
        }

        if stop:
            data["stop"] = [stop] if isinstance(stop, str) else stop

        options = self.config.get("options", {})
        if options:
            data["options"] = options

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
        prompt: Union[str, List[int]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None,
        **kwargs
    ) -> Iterator[str]:
        """生成文本（流式）"""
        if not self.is_available():
            raise RuntimeError("Model not loaded")

        max_tokens = max_tokens or self.config.get("max_tokens", 2048)
        temperature = temperature or self.config.get("temperature", 0.7)
        top_p = top_p or self.config.get("top_p", 1.0)

        data = {
            "model": self.model_path,
            "prompt": prompt if isinstance(prompt, str) else self._decode_tokens(prompt),
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": True
        }

        if stop:
            data["stop"] = [stop] if isinstance(stop, str) else stop

        options = self.config.get("options", {})
        if options:
            data["options"] = options

        response = self._client.post("/api/generate", json=data)
        response.raise_for_status()

        stop_list = [stop] if isinstance(stop, str) else (stop or [])

        for line in response.iter_lines():
            if line:
                import json
                chunk = json.loads(line)
                if "response" in chunk:
                    text_chunk = chunk["response"]

                    for stop_word in stop_list:
                        if stop_word in text_chunk:
                            text_chunk = text_chunk.split(stop_word)[0]
                            return

                    if text_chunk:
                        yield text_chunk

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """聊天模式推理"""
        if not self.is_available():
            raise RuntimeError("Model not loaded")

        max_tokens = max_tokens or self.config.get("max_tokens", 2048)
        temperature = temperature or self.config.get("temperature", 0.7)

        data = {
            "model": self.model_path,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }

        response = self._client.post("/api/chat", json=data)
        response.raise_for_status()

        result = ""
        for line in response.iter_lines():
            if line:
                import json
                chunk = json.loads(line)
                if "message" in chunk and "content" in chunk["message"]:
                    result += chunk["message"]["content"]
                if chunk.get("done"):
                    break

        return result

    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> Iterator[str]:
        """聊天模式推理（流式）"""
        if not self.is_available():
            raise RuntimeError("Model not loaded")

        max_tokens = max_tokens or self.config.get("max_tokens", 2048)
        temperature = temperature or self.config.get("temperature", 0.7)

        data = {
            "model": self.model_path,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True
        }

        response = self._client.post("/api/chat", json=data)
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                import json
                chunk = json.loads(line)
                if "message" in chunk and "content" in chunk["message"]:
                    yield chunk["message"]["content"]

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
            max_new_tokens=self.config.get("max_tokens", 2048)
        )

    def _decode_tokens(self, tokens: List[int]) -> str:
        """将 token IDs 解码为字符串"""
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            return tokenizer.decode(tokens)
        except Exception:
            return str(tokens)

    def is_ollama_available(self) -> bool:
        """检查 Ollama 服务是否可用"""
        try:
            response = self._client.get("/api/tags")
            return response.status_code == 200
        except Exception:
            return False

    def get_available_models(self) -> List[str]:
        """获取可用模型列表"""
        try:
            response = self._client.get("/api/tags")
            response.raise_for_status()
            data = response.json()
            return [m.get("name", "") for m in data.get("models", [])]
        except Exception:
            return []