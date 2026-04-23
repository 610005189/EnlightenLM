"""
DashScope API Client - 阿里云百炼 API 客户端
支持千问等模型的 API 调用
"""

import os
import requests
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class DashScopeConfig:
    """DashScope API 配置"""
    api_key: str
    base_url: str = "https://dashscope.aliyuncs.com"
    model: str = "qwen-turbo"
    max_tokens: int = 2048
    temperature: float = 0.7


class DashScopeAPIClient:
    """
    阿里云百炼（DashScope）API 客户端

    支持千问等模型的 API 调用
    """

    def __init__(self, config: Optional[DashScopeConfig] = None):
        if config is None:
            api_key = os.environ.get("DASHSCOPE_API_KEY")
            if not api_key:
                raise ValueError("DashScope API key 未设置，请设置环境变量 DASHSCOPE_API_KEY")

            config = DashScopeConfig(api_key=api_key)

        self.config = config

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> Tuple[str, float]:
        """
        生成文本

        Args:
            prompt: 输入提示
            max_tokens: 最大 token 数
            temperature: 温度参数
            **kwargs: 其他参数

        Returns:
            (response_text, latency)
        """
        import time
        start_time = time.time()

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.config.model,
            "input": {
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            },
            "parameters": {
                "max_tokens": max_tokens or self.config.max_tokens,
                "temperature": temperature or self.config.temperature,
                **kwargs
            }
        }

        try:
            response = requests.post(
                f"{self.config.base_url}/api/v1/services/aigc/text-generation/generation",
                headers=headers,
                json=payload,
                timeout=60
            )

            response.raise_for_status()

            result = response.json()

            latency = time.time() - start_time

            if "output" in result and "text" in result["output"]:
                content = result["output"]["text"]
                return content, latency
            elif "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0].get("message", {}).get("content", "")
                return content, latency
            else:
                return f"API响应格式异常: {result}", latency

        except requests.exceptions.Timeout:
            return "API请求超时", time.time() - start_time
        except requests.exceptions.RequestException as e:
            return f"API请求失败: {str(e)}", time.time() - start_time
        except Exception as e:
            return f"异常: {str(e)}", time.time() - start_time

    def generate_with_system(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> Tuple[str, float]:
        """
        带系统提示的生成

        Args:
            system_prompt: 系统提示
            user_prompt: 用户提示
            max_tokens: 最大 token 数
            temperature: 温度参数

        Returns:
            (response_text, latency)
        """
        import time
        start_time = time.time()

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.config.model,
            "input": {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            },
            "parameters": {
                "max_tokens": max_tokens or self.config.max_tokens,
                "temperature": temperature or self.config.temperature,
                **kwargs
            }
        }

        try:
            response = requests.post(
                f"{self.config.base_url}/api/v1/services/aigc/text-generation/generation",
                headers=headers,
                json=payload,
                timeout=60
            )

            response.raise_for_status()

            result = response.json()

            latency = time.time() - start_time

            if "output" in result and "text" in result["output"]:
                content = result["output"]["text"]
                return content, latency
            else:
                return f"API响应格式异常: {result}", latency

        except Exception as e:
            return f"异常: {str(e)}", time.time() - start_time

    def is_available(self) -> bool:
        """检查 API 是否可用"""
        try:
            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
            }

            response = requests.get(
                f"{self.config.base_url}/api/v1/models",
                headers=headers,
                timeout=10
            )

            return response.status_code in [200, 401]
        except Exception:
            return False

    def get_model_list(self) -> Dict[str, Any]:
        """获取可用模型列表"""
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
        }

        try:
            response = requests.get(
                f"{self.config.base_url}/api/v1/models",
                headers=headers,
                timeout=10
            )

            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}", "message": response.text}
        except Exception as e:
            return {"error": str(e)}


def create_dashscope_client() -> DashScopeAPIClient:
    """创建 DashScope API 客户端"""
    return DashScopeAPIClient()


if __name__ == "__main__":
    print("测试 DashScope API 客户端...")

    client = DashScopeAPIClient()

    print(f"API 可用: {client.is_available()}")

    print("\n测试生成...")
    response, latency = client.generate("你好，请介绍一下自己")

    print(f"响应: {response}")
    print(f"延迟: {latency:.2f}秒")
