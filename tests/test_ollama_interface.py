"""Ollama 模型接口测试"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from enlighten.interfaces.base import ModelProvider, ModelState
from enlighten.interfaces.config import OllamaModelConfig
from enlighten.interfaces.factory import ModelFactory
from enlighten.interfaces.errors import ModelLoadError
from enlighten.interfaces.providers.ollama import OllamaModel


class TestOllamaModelConfig:
    """测试 Ollama 模型配置"""

    def test_default_config(self):
        """测试默认配置"""
        config = OllamaModelConfig(
            provider=ModelProvider.OLLAMA,
            model_path="llama3.2"
        )

        assert config.provider == ModelProvider.OLLAMA
        assert config.model_path == "llama3.2"
        assert config.base_url == "http://localhost:11434"
        assert config.stream == True
        assert config.max_tokens == 2048
        assert config.temperature == 0.7

    def test_custom_config(self):
        """测试自定义配置"""
        config = OllamaModelConfig(
            provider=ModelProvider.OLLAMA,
            model_path="qwen2.5:14b",
            base_url="http://192.168.1.100:11434",
            timeout=120
        )

        assert config.base_url == "http://192.168.1.100:11434"
        assert config.timeout == 120


class TestOllamaModelInterface:
    """测试 Ollama 模型接口"""

    def test_model_factory_registration(self):
        """测试模型工厂注册"""
        providers = ModelFactory.get_registered_providers()
        assert ModelProvider.OLLAMA in providers

    def test_create_ollama_model(self):
        """测试通过工厂创建 Ollama 模型"""
        config = OllamaModelConfig(
            provider=ModelProvider.OLLAMA,
            model_path="llama3.2"
        )

        model = ModelFactory.create(config)
        assert model.model_path == "llama3.2"
        assert model.base_url == "http://localhost:11434"
        assert model._state == ModelState.UNINITIALIZED

    def test_create_from_dict(self):
        """测试从字典创建模型"""
        config_dict = {
            "provider": "ollama",
            "model_path": "llama3.2",
            "base_url": "http://localhost:11434"
        }

        model = ModelFactory.create(config_dict)
        assert model.model_path == "llama3.2"


class TestOllamaModelLoading:
    """测试 Ollama 模型加载"""

    @patch("httpx.Client")
    def test_load_success(self, mock_client_class):
        """测试成功加载模型"""
        from enlighten.interfaces.providers.ollama import OllamaModel

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "llama3.2", "size": 1234567890}
            ]
        }
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        model = OllamaModel(
            model_path="llama3.2",
            base_url="http://localhost:11434"
        )

        result = model.load()

        assert result == True
        assert model._state == ModelState.READY
        assert model._metadata is not None
        assert model._metadata.name == "llama3.2"

    @patch("httpx.Client")
    def test_load_failure(self, mock_client_class):
        """测试加载失败"""
        from enlighten.interfaces.providers.ollama import OllamaModel

        mock_client_class.side_effect = Exception("Connection refused")

        model = OllamaModel(
            model_path="llama3.2",
            base_url="http://localhost:11434"
        )

        with pytest.raises(ModelLoadError):
            model.load()

        assert model._state == ModelState.ERROR


class TestOllamaModelGeneration:
    """测试 Ollama 模型推理"""

    def test_generate_without_loaded_model(self):
        """测试未加载模型时生成应该失败"""
        from enlighten.interfaces.providers.ollama import OllamaModel

        model = OllamaModel(
            model_path="llama3.2",
            base_url="http://localhost:11434"
        )

        with pytest.raises(RuntimeError, match="Model not loaded"):
            model.generate("Hello")

    @patch("httpx.Client")
    def test_generate_success(self, mock_client_class):
        """测试成功生成"""
        from enlighten.interfaces.providers.ollama import OllamaModel

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = [
            b'{"response": "Hello", "done": false}',
            b'{"response": " World", "done": true}'
        ]
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        model = OllamaModel(
            model_path="llama3.2",
            base_url="http://localhost:11434"
        )
        model.load()

        result = model.generate("Hi")

        assert result == "Hello World"

    @patch("httpx.Client")
    def test_generate_stream(self, mock_client_class):
        """测试流式生成"""
        from enlighten.interfaces.providers.ollama import OllamaModel

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = [
            b'{"response": "Hello", "done": false}',
            b'{"response": " World", "done": true}'
        ]
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        model = OllamaModel(
            model_path="llama3.2",
            base_url="http://localhost:11434"
        )
        model.load()

        chunks = list(model.generate_stream("Hi"))

        assert chunks == ["Hello", " World"]


class TestOllamaModelChat:
    """测试 Ollama 聊天功能"""

    @patch("httpx.Client")
    def test_chat_success(self, mock_client_class):
        """测试成功聊天"""
        from enlighten.interfaces.providers.ollama import OllamaModel

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = [
            b'{"message": {"content": "Hello!"}, "done": false}',
            b'{"message": {"content": " How can I help?"}, "done": true}'
        ]
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        model = OllamaModel(
            model_path="llama3.2",
            base_url="http://localhost:11434"
        )
        model.load()

        messages = [
            {"role": "user", "content": "Hi"}
        ]

        result = model.chat(messages)

        assert result == "Hello! How can I help?"


class TestOllamaModelState:
    """测试 Ollama 模型状态管理"""

    def test_initial_state(self):
        """测试初始状态"""
        from enlighten.interfaces.providers.ollama import OllamaModel

        model = OllamaModel(model_path="llama3.2")

        assert model._state == ModelState.UNINITIALIZED
        assert model.is_available() == False

    @patch("httpx.Client")
    def test_ready_state_after_load(self, mock_client_class):
        """测试加载后状态为 READY"""
        from enlighten.interfaces.providers.ollama import OllamaModel

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": []}
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        model = OllamaModel(model_path="llama3.2")
        model.load()

        assert model._state == ModelState.READY
        assert model.is_available() == True

    @patch("httpx.Client")
    def test_unload(self, mock_client_class):
        """测试卸载模型"""
        from enlighten.interfaces.providers.ollama import OllamaModel

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": []}
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        model = OllamaModel(model_path="llama3.2")
        model.load()
        model.unload()

        assert model._state == ModelState.UNLOADED
        assert model.is_available() == False
        mock_client.close.assert_called_once()


class TestOllamaModelContextManager:
    """测试 Ollama 模型上下文管理器"""

    @patch("httpx.Client")
    def test_context_manager(self, mock_client_class):
        """测试上下文管理器"""
        from enlighten.interfaces.providers.ollama import OllamaModel

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": []}
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        with OllamaModel(model_path="llama3.2") as model:
            assert model.is_available() == True

        assert model._state == ModelState.UNLOADED


class TestOllamaModelHelperMethods:
    """测试 Ollama 模型辅助方法"""

    def test_get_device(self):
        """测试获取设备"""
        from enlighten.interfaces.providers.ollama import OllamaModel

        model = OllamaModel(model_path="llama3.2")
        assert model.get_device() == "cpu"

    def test_get_memory_usage(self):
        """测试获取内存使用"""
        from enlighten.interfaces.providers.ollama import OllamaModel

        model = OllamaModel(model_path="llama3.2")
        memory = model.get_memory_usage()

        assert "allocated" in memory
        assert "reserved" in memory
        assert "system_mem" in memory

    def test_get_capabilities(self):
        """测试获取能力"""
        from enlighten.interfaces.providers.ollama import OllamaModel

        model = OllamaModel(model_path="llama3.2")
        capabilities = model.get_capabilities()

        assert capabilities.supports_streaming == True
        assert capabilities.max_context_length == 8192


if __name__ == "__main__":
    pytest.main([__file__, "-v"])