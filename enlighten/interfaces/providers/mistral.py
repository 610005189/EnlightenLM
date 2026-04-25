"""Mistral 模型实现

基于 HuggingFace Transformers 的 Mistral 模型支持。
"""

from typing import Any, Dict, Iterator, List, Optional, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from enlighten.interfaces.base import (
    BaseModelInterface,
    ModelCapability,
    ModelMetadata,
    ModelProvider,
    ModelState,
)
from enlighten.interfaces.config import MistralModelConfig
from enlighten.interfaces.errors import ModelLoadError
from enlighten.interfaces.factory import ModelFactory


@ModelFactory.register(ModelProvider.MISTRAL)
class MistralModel(BaseModelInterface):
    """Mistral 模型实现

    基于 HuggingFace Transformers 实现，支持本地 Mistral 模型推理。
    Mistral 是一个高效的开源模型，支持注意力机制的优化。
    """

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(model_path=model_path, device=device, config=config or {})

        self._tokenizer = None
        self._model = None

        self._load_in_8bit = self.config.get("load_in_8bit", False)
        self._load_in_4bit = self.config.get("load_in_4bit", False)
        self._trust_remote_code = self.config.get("trust_remote_code", False)
        self._use_flash_attention = self.config.get("use_flash_attention", True)
        self._max_seq_len = self.config.get("max_seq_len", 4096)
        self._revision = self.config.get("revision", None)

    def load(self, **kwargs) -> bool:
        """加载 Mistral 模型"""
        self._state = ModelState.LOADING

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=self._trust_remote_code, revision=self._revision
            )

            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            device_map = self.config.get("device_map", "auto")

            if self._load_in_8bit or self._load_in_4bit:
                from transformers import BitsAndBytesConfig

                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=self._load_in_8bit, load_in_4bit=self._load_in_4bit
                )
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    quantization_config=quantization_config,
                    device_map=device_map,
                    trust_remote_code=self._trust_remote_code,
                    revision=self._revision,
                    torch_dtype=self._get_dtype(),
                )
            else:
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    device_map=device_map,
                    trust_remote_code=self._trust_remote_code,
                    revision=self._revision,
                    torch_dtype=self._get_dtype(),
                    attn_implementation="flash_attention_2"
                    if self._use_flash_attention
                    else "eager",
                )

            self._model.eval()

            self._metadata = self._extract_metadata()
            self._state = ModelState.READY
            return True

        except Exception as e:
            self._state = ModelState.ERROR
            raise ModelLoadError(f"Mistral load failed: {e}") from e

    def _get_dtype(self) -> torch.dtype:
        """获取数据类型"""
        dtype_str = self.config.get("dtype", "auto")
        if dtype_str == "auto":
            if torch.cuda.is_available():
                return torch.float16
            return torch.float32
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "int8": torch.int8,
        }
        return dtype_map.get(dtype_str, torch.float16)

    def _extract_metadata(self) -> ModelMetadata:
        """提取模型元数据"""
        config = self._model.config
        return ModelMetadata(
            name=self.model_path,
            provider=ModelProvider.MISTRAL,
            model_path=self.model_path,
            revision=self._revision,
            context_length=getattr(config, "max_position_embeddings", self._max_seq_len),
            vocab_size=getattr(config, "vocab_size", 0),
            hidden_size=getattr(config, "hidden_size", 0),
            num_layers=getattr(config, "num_hidden_layers", 0),
            num_heads=getattr(config, "num_attention_heads", 0),
            head_dim=getattr(config, "head_dim", 0),
            capabilities=self.get_capabilities(),
        )

    def unload(self) -> None:
        """卸载 Mistral 模型"""
        if self._model is not None:
            del self._model
            self._model = None

        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None

        self._metadata = None
        self._state = ModelState.UNLOADED
        self.cleanup()

    def generate(
        self,
        prompt: Union[str, List[int]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ) -> str:
        """生成文本（非流式）"""
        if not self.is_available():
            raise RuntimeError("Model not loaded")

        max_tokens = max_tokens or self.config.get("max_tokens", 2048)
        temperature = temperature or self.config.get("temperature", 0.7)
        top_p = top_p or self.config.get("top_p", 1.0)

        if isinstance(prompt, str):
            inputs = self._tokenizer(prompt, return_tensors="pt")
        else:
            inputs = {"input_ids": torch.tensor([prompt])}

        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        generation_kwargs = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": temperature > 0,
            **kwargs,
        }

        with torch.no_grad():
            outputs = self._model.generate(**inputs, **generation_kwargs)

        generated_text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)

        if isinstance(prompt, str):
            generated_text = generated_text[len(prompt) :]

        if stop:
            stop_list = [stop] if isinstance(stop, str) else stop
            for stop_word in stop_list:
                if stop_word in generated_text:
                    generated_text = generated_text.split(stop_word)[0]

        return generated_text

    def generate_stream(
        self,
        prompt: Union[str, List[int]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ) -> Iterator[str]:
        """生成文本（流式）"""
        if not self.is_available():
            raise RuntimeError("Model not loaded")

        max_tokens = max_tokens or self.config.get("max_tokens", 2048)
        temperature = temperature or self.config.get("temperature", 0.7)

        if isinstance(prompt, str):
            inputs = self._tokenizer(prompt, return_tensors="pt")
        else:
            inputs = {"input_ids": torch.tensor([prompt])}

        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        stop_list = [stop] if isinstance(stop, str) else (stop or [])
        prompt_len = inputs["input_ids"].shape[1]

        generation_kwargs = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p or self.config.get("top_p", 1.0),
            "do_sample": temperature > 0,
            **kwargs,
        }

        from transformers import TextIteratorStreamer
        from threading import Thread

        streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True)

        generation_kwargs["streamer"] = streamer

        def generate_in_thread():
            with torch.no_grad():
                self._model.generate(**inputs, **generation_kwargs)

        thread = Thread(target=generate_in_thread)
        thread.start()

        for text_chunk in streamer:
            should_stop = False
            for stop_word in stop_list:
                if stop_word in text_chunk:
                    text_chunk = text_chunk.split(stop_word)[0]
                    should_stop = True
                    break

            if text_chunk:
                yield text_chunk

            if should_stop or len(text_chunk) == 0:
                thread.join()
                return

        thread.join()

    def get_device(self) -> str:
        """获取模型运行设备"""
        if self._model is not None:
            if hasattr(self._model, "device"):
                return str(self._model.device)
            if hasattr(self._model, "hf_device_map"):
                devices = set(self._model.hf_device_map.values())
                return f"cuda:{min(devices)}" if devices else "cpu"
        return self.device

    def get_memory_usage(self) -> Dict[str, float]:
        """获取内存使用情况"""
        if not torch.cuda.is_available():
            return {"allocated": 0.0, "reserved": 0.0, "system_mem": 0.0}

        return {
            "allocated": torch.cuda.memory_allocated() / 1024**2,
            "reserved": torch.cuda.memory_reserved() / 1024**2,
            "system_mem": 0.0,
        }

    def get_capabilities(self) -> ModelCapability:
        """获取模型能力"""
        return ModelCapability(
            supports_streaming=True,
            supports_batch_inference=True,
            supports_prefix_caching=self._use_flash_attention,
            max_context_length=self._metadata.context_length
            if self._metadata
            else self._max_seq_len,
            max_new_tokens=self.config.get("max_tokens", 2048),
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> str:
        """聊天模式推理"""
        if not self.is_available():
            raise RuntimeError("Model not loaded")

        prompt = self._format_mistral_chat_prompt(messages)

        return self.generate(
            prompt=prompt, max_tokens=max_tokens, temperature=temperature, **kwargs
        )

    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> Iterator[str]:
        """聊天模式推理（流式）"""
        if not self.is_available():
            raise RuntimeError("Model not loaded")

        prompt = self._format_mistral_chat_prompt(messages)

        yield from self.generate_stream(
            prompt=prompt, max_tokens=max_tokens, temperature=temperature, **kwargs
        )

    def _format_mistral_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        """格式化聊天消息为 Mistral 格式的 prompt

        Mistral-Instruct 模型使用类似以下的模板:
        [INST] user message [/INST] assistant response
        """
        if not messages:
            return ""

        formatted_parts = []

        for i, msg in enumerate(messages):
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                formatted_parts.append(f"[INST] {content} [/INST]")
            elif role == "user":
                if formatted_parts:
                    formatted_parts.append(f"[INST] {content} [/INST]")
                else:
                    formatted_parts.append(f"[INST] {content} [/INST]")
            elif role == "assistant":
                formatted_parts.append(f"{content}")

        if not formatted_parts[-1].endswith("[/INST]"):
            formatted_parts.append("")
        else:
            formatted_parts.append("")

        return " ".join(formatted_parts)
