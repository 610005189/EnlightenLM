"""
DeepSeek API Client - DeepSeek API 客户端
支持 DeepSeek-V3 等模型的 API 调用
"""

import os
import requests
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

# 加载环境变量
from dotenv import load_dotenv
from pathlib import Path

# 从项目根目录加载.env文件
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


@dataclass
class DeepSeekConfig:
    """DeepSeek API 配置"""
    api_key: str
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"
    max_tokens: int = 2048
    temperature: float = 0.7


class DeepSeekAPIClient:
    """
    DeepSeek API 客户端

    支持 DeepSeek-V3 等模型的 API 调用
    """

    def __init__(self, config: Optional[DeepSeekConfig] = None):
        if config is None:
            api_key = os.environ.get("DEEPSEEK_API_KEY")
            if not api_key:
                raise ValueError("DeepSeek API key 未设置，请设置环境变量 DEEPSEEK_API_KEY")

            config = DeepSeekConfig(api_key=api_key)

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
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens or self.config.max_tokens,
            "temperature": temperature or self.config.temperature,
            **kwargs
        }

        try:
            response = requests.post(
                f"{self.config.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )

            response.raise_for_status()

            result = response.json()

            latency = time.time() - start_time

            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                return content, latency
            else:
                return f"API错误: {result}", latency

        except requests.exceptions.Timeout:
            return "API请求超时", time.time() - start_time
        except requests.exceptions.RequestException as e:
            return f"API请求失败: {str(e)}", time.time() - start_time
        except Exception as e:
            return f"异常: {str(e)}", time.time() - start_time

    def generate_stream(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ):
        """
        流式生成文本

        Args:
            prompt: 输入提示
            max_tokens: 最大 token 数
            temperature: 温度参数
            **kwargs: 其他参数

        Yields:
            生成的文本片段
        """
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens or self.config.max_tokens,
            "temperature": temperature or self.config.temperature,
            "stream": True,
            **kwargs
        }

        try:
            response = requests.post(
                f"{self.config.base_url}/chat/completions",
                headers=headers,
                json=payload,
                stream=True,
                timeout=60
            )

            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    line_text = line.decode('utf-8')
                    if line_text.startswith('data: '):
                        data = line_text[6:]
                        if data == '[DONE]':
                            break
                        import json
                        try:
                            chunk = json.loads(data)
                            if "choices" in chunk and len(chunk["choices"]) > 0:
                                delta = chunk["choices"][0].get("delta", {})
                                if "content" in delta:
                                    yield delta["content"]
                        except json.JSONDecodeError:
                            continue

        except Exception as e:
            yield f"异常: {str(e)}"

    def is_available(self) -> bool:
        """检查 API 是否可用"""
        try:
            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
            }

            response = requests.get(
                f"{self.config.base_url}/models",
                headers=headers,
                timeout=10
            )

            return response.status_code == 200
        except Exception:
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
        }

        try:
            response = requests.get(
                f"{self.config.base_url}/models",
                headers=headers,
                timeout=10
            )

            response.raise_for_status()

            return response.json()
        except Exception as e:
            return {"error": str(e)}


def create_deepseek_client() -> DeepSeekAPIClient:
    """创建 DeepSeek API 客户端"""
    return DeepSeekAPIClient()


if __name__ == "__main__":
    print("测试 DeepSeek API 客户端...")

    client = DeepSeekAPIClient()

    print(f"API 可用: {client.is_available()}")

    print("\n测试生成...")
    response, latency = client.generate("你好，请介绍一下自己")

    print(f"响应: {response}")
    print(f"延迟: {latency:.2f}秒")
