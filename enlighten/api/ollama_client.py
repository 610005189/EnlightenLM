"""Ollama API 客户端

用于与本地 Ollama 服务交互，提供模型推理功能。
"""

import httpx
import json
from typing import Optional, Dict, Any


class OllamaConfig:
    """Ollama 配置类"""
    def __init__(self,
                 base_url: str = "http://localhost:11434",
                 model: str = "qwen2.5:14b",
                 temperature: float = 0.7,
                 max_tokens: int = 1024):
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens


class OllamaAPIClient:
    """Ollama API 客户端"""
    def __init__(self, config: Optional[OllamaConfig] = None):
        self.config = config or OllamaConfig()
        self.client = httpx.Client(base_url=self.config.base_url, timeout=30.0)

    def is_available(self) -> bool:
        """检查 Ollama 服务是否可用"""
        try:
            response = self.client.get("/api/tags")
            return response.status_code == 200
        except Exception:
            return False

    def generate(self, prompt: str, **kwargs) -> str:
        """生成文本"""
        data = {
            "model": kwargs.get("model", self.config.model),
            "prompt": prompt,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens)
        }

        try:
            response = self.client.post("/api/generate", json=data)
            response.raise_for_status()
            
            # 解析流式响应
            result = ""
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if "response" in data:
                        result += data["response"]
                    if data.get("done", False):
                        break
            return result
        except Exception as e:
            raise RuntimeError(f"Ollama API 调用失败: {str(e)}")

    def chat(self, messages: list, **kwargs) -> str:
        """聊天模式"""
        data = {
            "model": kwargs.get("model", self.config.model),
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens)
        }

        try:
            response = self.client.post("/api/chat", json=data)
            response.raise_for_status()
            
            # 解析流式响应
            result = ""
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if "message" in data and "content" in data["message"]:
                        result += data["message"]["content"]
                    if data.get("done", False):
                        break
            return result
        except Exception as e:
            raise RuntimeError(f"Ollama API 调用失败: {str(e)}")

    def get_models(self) -> list:
        """获取可用模型列表"""
        try:
            response = self.client.get("/api/tags")
            response.raise_for_status()
            data = response.json()
            return data.get("models", [])
        except Exception as e:
            raise RuntimeError(f"获取模型列表失败: {str(e)}")

    def close(self):
        """关闭客户端"""
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
