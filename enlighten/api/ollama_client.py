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
                 model: str = "qwen2.5:7b",
                 temperature: float = 0.7,
                 max_tokens: int = 1024,
                 retry_endpoints: bool = True):
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retry_endpoints = retry_endpoints


class OllamaAPIClient:
    """Ollama API 客户端"""
    
    # 标准API端点配置
    API_ENDPOINTS = [
        ("/api/chat", "chat"),
        ("/api/generate", "generate")
    ]
    
    def __init__(self, config: Optional[OllamaConfig] = None):
        self.config = config or OllamaConfig()
        self.client = httpx.Client(base_url=self.config.base_url, timeout=120.0)

    def is_available(self) -> bool:
        """检查 Ollama 服务是否可用"""
        try:
            response = self.client.get("/api/tags")
            return response.status_code == 200
        except Exception:
            return False

    def _parse_response(self, response: httpx.Response, endpoint: str) -> str:
        """解析API响应"""
        data = response.json()
        if endpoint == "/api/chat":
            return data.get("message", {}).get("content", "")
        elif endpoint == "/api/generate":
            return data.get("response", "")
        return ""

    def _classify_error(self, error: Exception) -> str:
        """分类错误类型"""
        if isinstance(error, httpx.TimeoutException):
            return "请求超时"
        elif isinstance(error, httpx.ConnectError):
            return "连接失败"
        elif isinstance(error, httpx.HTTPStatusError):
            status_code = error.response.status_code
            if status_code == 404:
                return "端点不存在"
            elif status_code == 500:
                return "服务器内部错误"
            elif status_code == 503:
                return "服务不可用"
            return f"HTTP错误({status_code})"
        return f"未知错误({str(error)})"

    def generate(self, prompt: str, **kwargs) -> str:
        """生成文本"""
        model = kwargs.get("model", self.config.model)
        temperature = kwargs.get("temperature", self.config.temperature)
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)
        
        # 构建端点列表
        endpoints = []
        if self.config.retry_endpoints:
            endpoints = self.API_ENDPOINTS
        else:
            endpoints = [self.API_ENDPOINTS[0]]  # 只使用第一个端点
        
        # 准备请求数据
        chat_data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        generate_data = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        endpoint_data = {
            "/api/chat": chat_data,
            "/api/generate": generate_data
        }
        
        last_error = None
        tried_endpoints = []
        
        for endpoint, _ in endpoints:
            try:
                data = endpoint_data[endpoint]
                response = self.client.post(endpoint, json=data)
                response.raise_for_status()
                return self._parse_response(response, endpoint)
            except Exception as e:
                last_error = e
                error_type = self._classify_error(e)
                tried_endpoints.append(f"{endpoint}: {error_type}")
        
        # 所有端点都失败
        tried_str = "; ".join(tried_endpoints)
        raise RuntimeError(f"Ollama API调用失败，尝试了{len(tried_endpoints)}个端点: {tried_str}")

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
