"""Claude 模型实现

Anthropic Claude API 模型支持。
"""

import json
from typing import Any, Dict, Iterator, List, Optional, Union

import httpx

from enlighten.interfaces.base import (
    BaseModelInterface,
    ModelCapability,
    ModelMetadata,
    ModelProvider,
    ModelState,
)
from enlighten.interfaces.config import ClaudeModelConfig
from enlighten.interfaces.errors import ModelLoadError
from enlighten.interfaces.factory import ModelFactory


@ModelFactory.register(ModelProvider.CLAUDE)
class ClaudeModel(BaseModelInterface):
    """Claude 模型实现

    通过 Anthropic API 与 Claude 交互，支持:
    - Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Haiku 等
    - 流式和非流式推理
    - 聊天模式
    """

    def __init__(
        self,
        model_path: str = "claude-3-5-sonnet-20241022",
        api_key: Optional[str] = None,
        base_url: str = "https://api.anthropic.com",
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__(model_path=model_path, device="api", config=config or {})

        self.api_key = api_key or self.config.get("api_key")
        self.base_url = base_url or self.config.get("base_url", "https://api.anthropic.com")
        self.api_version = self.config.get("api_version", "2023-06-01")
        self._stream = self.config.get("stream", True)

        self._client: Optional[httpx.Client] = None
        self._available_models: Optional[List[str]] = None

    def load(self, **kwargs) -> bool:
        """加载 Claude 模型（初始化 API 客户端）"""
        self._state = ModelState.LOADING

        try:
            self._client = httpx.Client(
                base_url=self.base_url,
                timeout=self.config.get("timeout", 120),
                headers=self._get_headers()
            )

            self._metadata = ModelMetadata(
                name=self.model_path,
                provider=ModelProvider.CLAUDE,
                model_path=self.model_path,
                capabilities=self.get_capabilities()
            )

            self._state = ModelState.READY
            return True

        except Exception as e:
            self._state = ModelState.ERROR
            raise ModelLoadError(f"Claude load failed: {e}") from e

    def _get_headers(self) -> Dict[str, str]:
        """获取请求头"""
        headers = {
            "x-api-key": self.api_key or "",
            "anthropic-version": self.api_version,
            "content-type": "application/json"
        }
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    def unload(self) -> None:
        """卸载 Claude 模型（关闭 API 客户端）"""
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
        temperature = temperature if temperature is not None else self.config.get("temperature", 0.7)
        top_p = top_p or self.config.get("top_p", 1.0)

        prompt_text = prompt if isinstance(prompt, str) else self._decode_tokens(prompt)

        messages = [{"role": "user", "content": prompt_text}]
        data = self._build_request_data(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_sequences=stop,
            stream=False,
            **kwargs
        )

        response = self._client.post("/v1/messages", json=data)
        response.raise_for_status()

        result = response.json()
        return result.get("content", [{}])[0].get("text", "")

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
        temperature = temperature if temperature is not None else self.config.get("temperature", 0.7)
        top_p = top_p or self.config.get("top_p", 1.0)

        prompt_text = prompt if isinstance(prompt, str) else self._decode_tokens(prompt)

        messages = [{"role": "user", "content": prompt_text}]
        data = self._build_request_data(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_sequences=stop,
            stream=True,
            **kwargs
        )

        stop_list = [stop] if isinstance(stop, str) else (stop or [])

        with self._client.stream("POST", "/v1/messages", json=data) as response:
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    line = line.strip()
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break

                        try:
                            chunk = json.loads(data_str)
                            if chunk.get("type") == "content_block_delta":
                                delta = chunk.get("delta", {})
                                if delta.get("type") == "text_delta":
                                    text_chunk = delta.get("text", "")

                                    for stop_word in stop_list:
                                        if stop_word in text_chunk:
                                            text_chunk = text_chunk.split(stop_word)[0]
                                            if text_chunk:
                                                yield text_chunk
                                            return

                                    if text_chunk:
                                        yield text_chunk
                        except json.JSONDecodeError:
                            continue

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None,
        **kwargs
    ) -> str:
        """聊天模式推理"""
        if not self.is_available():
            raise RuntimeError("Model not loaded")

        max_tokens = max_tokens or self.config.get("max_tokens", 2048)
        temperature = temperature if temperature is not None else self.config.get("temperature", 0.7)
        top_p = top_p or self.config.get("top_p", 1.0)

        anthropic_messages = self._convert_messages_format(messages)

        data = self._build_request_data(
            messages=anthropic_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_sequences=stop,
            stream=False,
            **kwargs
        )

        response = self._client.post("/v1/messages", json=data)
        response.raise_for_status()

        result = response.json()
        return result.get("content", [{}])[0].get("text", "")

    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None,
        **kwargs
    ) -> Iterator[str]:
        """聊天模式推理（流式）"""
        if not self.is_available():
            raise RuntimeError("Model not loaded")

        max_tokens = max_tokens or self.config.get("max_tokens", 2048)
        temperature = temperature if temperature is not None else self.config.get("temperature", 0.7)
        top_p = top_p or self.config.get("top_p", 1.0)

        anthropic_messages = self._convert_messages_format(messages)

        data = self._build_request_data(
            messages=anthropic_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_sequences=stop,
            stream=True,
            **kwargs
        )

        stop_list = [stop] if isinstance(stop, str) else (stop or [])

        with self._client.stream("POST", "/v1/messages", json=data) as response:
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    line = line.strip()
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break

                        try:
                            chunk = json.loads(data_str)
                            if chunk.get("type") == "content_block_delta":
                                delta = chunk.get("delta", {})
                                if delta.get("type") == "text_delta":
                                    text_chunk = delta.get("text", "")

                                    for stop_word in stop_list:
                                        if stop_word in text_chunk:
                                            text_chunk = text_chunk.split(stop_word)[0]
                                            if text_chunk:
                                                yield text_chunk
                                            return

                                    if text_chunk:
                                        yield text_chunk
                        except json.JSONDecodeError:
                            continue

    def _build_request_data(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop_sequences: Optional[List[str]] = None,
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """构建请求数据"""
        data = {
            "model": self.model_path,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": stream
        }

        if temperature > 0:
            data["temperature"] = temperature

        if top_p < 1.0:
            data["top_p"] = top_p

        if stop_sequences:
            data["stop_sequences"] = stop_sequences

        system = kwargs.get("system")
        if system:
            data["system"] = system

        if kwargs.get("tools"):
            data["tools"] = kwargs["tools"]

        return data

    def _convert_messages_format(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """转换消息格式以适配 Claude API

        Claude API 要求消息格式为 [{"role": "user", "content": "..."}]
        """
        converted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                continue

            if role == "assistant":
                role = "assistant"
            elif role == "user":
                role = "user"
            else:
                role = "user"

            converted.append({"role": role, "content": content})

        return converted

    def _decode_tokens(self, tokens: List[int]) -> str:
        """将 token IDs 解码为字符串"""
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            return tokenizer.decode(tokens)
        except Exception:
            return str(tokens)

    def get_device(self) -> str:
        """Claude API 运行在服务端，返回 api"""
        return "api"

    def get_memory_usage(self) -> Dict[str, float]:
        """获取内存使用（API 模式返回 0）"""
        return {
            "allocated": 0.0,
            "reserved": 0.0,
            "system_mem": 0.0
        }

    def get_capabilities(self) -> ModelCapability:
        """获取模型能力"""
        return ModelCapability(
            supports_streaming=True,
            supports_function_calling=self._supports_function_calling(),
            supports_vision=self._supports_vision(),
            supports_json_mode=True,
            max_context_length=self._get_context_length(),
            max_new_tokens=self.config.get("max_tokens", 2048)
        )

    def _supports_function_calling(self) -> bool:
        """检查是否支持 function calling"""
        return "claude-3" in self.model_path or "claude-3-5" in self.model_path

    def _supports_vision(self) -> bool:
        """检查是否支持视觉"""
        return "claude-3" in self.model_path or "claude-3-5" in self.model_path

    def _get_context_length(self) -> int:
        """获取上下文长度"""
        if "claude-3-opus" in self.model_path:
            return 200000
        elif "claude-3-sonnet" in self.model_path:
            return 200000
        elif "claude-3-5" in self.model_path:
            return 200000
        elif "claude-3-haiku" in self.model_path:
            return 200000
        return 100000

    def is_available(self) -> bool:
        """检查模型是否可用"""
        return self._state == ModelState.READY and self._client is not None

    def get_available_models(self) -> List[str]:
        """获取可用模型列表

        注意: Claude API 不提供模型列表接口，返回空列表
        """
        return [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-sonnet-20240620",
            "claude-3-haiku-20240307"
        ]
