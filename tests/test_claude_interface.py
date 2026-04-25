"""Claude 模型接口测试"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from enlighten.interfaces.base import ModelProvider, ModelState, ModelCapability, ModelMetadata
from enlighten.interfaces.config import ClaudeModelConfig
from enlighten.interfaces.factory import ModelFactory
from enlighten.interfaces.errors import ModelLoadError
from enlighten.interfaces.providers.claude import ClaudeModel


class TestClaudeModelConfig:
    """测试 Claude 模型配置"""

    def test_default_config(self):
        """测试默认配置"""
        config = ClaudeModelConfig(
            provider=ModelProvider.CLAUDE,
            model_path="claude-3-5-sonnet-20241022"
        )

        assert config.provider == ModelProvider.CLAUDE
        assert config.model_path == "claude-3-5-sonnet-20241022"
        assert config.base_url == "https://api.anthropic.com"
        assert config.api_version == "2023-06-01"
        assert config.stream == True
        assert config.max_tokens == 2048
        assert config.temperature == 0.7

    def test_custom_config(self):
        """测试自定义配置"""
        config = ClaudeModelConfig(
            provider=ModelProvider.CLAUDE,
            model_path="claude-3-opus-20240229",
            api_key="sk-test-key",
            base_url="https://custom.anthropic.com",
            timeout=180,
            max_retries=5
        )

        assert config.api_key == "sk-test-key"
        assert config.base_url == "https://custom.anthropic.com"
        assert config.timeout == 180
        assert config.max_retries == 5


class TestClaudeModelInterface:
    """测试 Claude 模型接口"""

    def test_model_factory_registration(self):
        """测试模型工厂注册"""
        providers = ModelFactory.get_registered_providers()
        assert ModelProvider.CLAUDE in providers

    def test_create_claude_model(self):
        """测试通过工厂创建 Claude 模型"""
        config = ClaudeModelConfig(
            provider=ModelProvider.CLAUDE,
            model_path="claude-3-5-sonnet-20241022",
            api_key="sk-test-key"
        )

        model = ModelFactory.create(config)
        assert model.model_path == "claude-3-5-sonnet-20241022"
        assert model._state == ModelState.UNINITIALIZED

    def test_create_from_dict(self):
        """测试从字典创建模型"""
        config_dict = {
            "provider": "claude",
            "model_path": "claude-3-5-sonnet-20241022",
            "api_key": "sk-test-key"
        }

        model = ModelFactory.create(config_dict)
        assert model.model_path == "claude-3-5-sonnet-20241022"
        assert model.api_key == "sk-test-key"


class TestClaudeModelLoading:
    """测试 Claude 模型加载"""

    @patch("enlighten.interfaces.providers.claude.httpx.Client")
    def test_load_success(self, mock_client_class):
        """测试成功加载模型"""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        model = ClaudeModel(
            model_path="claude-3-5-sonnet-20241022",
            api_key="sk-test-key"
        )

        result = model.load()

        assert result == True
        assert model._state == ModelState.READY
        assert model._client is not None
        assert model._metadata is not None
        assert model._metadata.provider == ModelProvider.CLAUDE
        assert model._metadata.name == "claude-3-5-sonnet-20241022"

    @patch("enlighten.interfaces.providers.claude.httpx.Client")
    def test_load_failure(self, mock_client_class):
        """测试加载失败"""
        mock_client_class.side_effect = Exception("Connection refused")

        model = ClaudeModel(
            model_path="claude-3-5-sonnet-20241022",
            api_key="sk-test-key"
        )

        with pytest.raises(ModelLoadError):
            model.load()

        assert model._state == ModelState.ERROR


class TestClaudeModelUnload:
    """测试 Claude 模型卸载"""

    @patch("enlighten.interfaces.providers.claude.httpx.Client")
    def test_unload(self, mock_client_class):
        """测试卸载模型"""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        model = ClaudeModel(
            model_path="claude-3-5-sonnet-20241022",
            api_key="sk-test-key"
        )
        model.load()
        model.unload()

        mock_client.close.assert_called_once()
        assert model._client is None
        assert model._metadata is None
        assert model._state == ModelState.UNLOADED


class TestClaudeModelGeneration:
    """测试 Claude 模型推理"""

    def setup_method(self):
        """设置测试环境"""
        self.mock_client = MagicMock()

    @patch("enlighten.interfaces.providers.claude.httpx.Client")
    def test_generate_without_loaded_model(self, mock_client_class):
        """测试未加载模型时生成应该失败"""
        model = ClaudeModel(
            model_path="claude-3-5-sonnet-20241022",
            api_key="sk-test-key"
        )

        with pytest.raises(RuntimeError, match="Model not loaded"):
            model.generate("Hello")

    @patch("enlighten.interfaces.providers.claude.httpx.Client")
    def test_generate_success(self, mock_client_class):
        """测试成功生成文本"""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "content": [{"type": "text", "text": "Hello! How can I help you today?"}]
        }
        mock_client.post.return_value = mock_response

        model = ClaudeModel(
            model_path="claude-3-5-sonnet-20241022",
            api_key="sk-test-key"
        )
        model.load()

        result = model.generate("Hello")

        assert isinstance(result, str)
        assert "Hello" in result or len(result) > 0
        mock_client.post.assert_called_once()

    @patch("enlighten.interfaces.providers.claude.httpx.Client")
    def test_generate_with_temperature(self, mock_client_class):
        """测试带温度参数的生成"""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "content": [{"type": "text", "text": "Generated text"}]
        }
        mock_client.post.return_value = mock_response

        model = ClaudeModel(
            model_path="claude-3-5-sonnet-20241022",
            api_key="sk-test-key",
            config={"temperature": 0.5}
        )
        model.load()

        result = model.generate("Hello", temperature=0.8)

        assert isinstance(result, str)


class TestClaudeModelChat:
    """测试 Claude 模型聊天"""

    @patch("enlighten.interfaces.providers.claude.httpx.Client")
    def test_chat_success(self, mock_client_class):
        """测试成功聊天"""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "content": [{"type": "text", "text": "I'm doing well! How can I assist you today?"}]
        }
        mock_client.post.return_value = mock_response

        model = ClaudeModel(
            model_path="claude-3-5-sonnet-20241022",
            api_key="sk-test-key"
        )
        model.load()

        messages = [
            {"role": "user", "content": "Hello, how are you?"}
        ]

        result = model.chat(messages)

        assert isinstance(result, str)
        assert len(result) > 0

    @patch("enlighten.interfaces.providers.claude.httpx.Client")
    def test_chat_with_system_message(self, mock_client_class):
        """测试带系统消息的聊天"""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "content": [{"type": "text", "text": "Understood! I'm ready to help."}]
        }
        mock_client.post.return_value = mock_response

        model = ClaudeModel(
            model_path="claude-3-5-sonnet-20241022",
            api_key="sk-test-key"
        )
        model.load()

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ]

        result = model.chat(messages)

        assert isinstance(result, str)


class TestClaudeModelStreaming:
    """测试 Claude 模型流式推理"""

    @patch("enlighten.interfaces.providers.claude.httpx.Client")
    def test_generate_stream(self, mock_client_class):
        """测试流式生成"""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_stream_response = MagicMock()
        mock_stream_response.iter_lines.return_value = iter([
            'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hello"}}',
            'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": " World"}}',
            'data: [DONE]'
        ])

        mock_client.stream.return_value.__enter__.return_value = mock_stream_response

        model = ClaudeModel(
            model_path="claude-3-5-sonnet-20241022",
            api_key="sk-test-key"
        )
        model.load()

        chunks = list(model.generate_stream("Hello"))

        assert len(chunks) == 2
        assert "Hello" in chunks[0]
        assert "World" in chunks[1]


class TestClaudeModelState:
    """测试 Claude 模型状态管理"""

    def test_initial_state(self):
        """测试初始状态"""
        model = ClaudeModel(
            model_path="claude-3-5-sonnet-20241022",
            api_key="sk-test-key"
        )

        assert model._state == ModelState.UNINITIALIZED
        assert model.is_available() == False

    @patch("enlighten.interfaces.providers.claude.httpx.Client")
    def test_ready_state_after_load(self, mock_client_class):
        """测试加载后状态为 READY"""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        model = ClaudeModel(
            model_path="claude-3-5-sonnet-20241022",
            api_key="sk-test-key"
        )
        model.load()

        assert model._state == ModelState.READY
        assert model.is_available() == True


class TestClaudeModelCapabilities:
    """测试 Claude 模型能力"""

    def test_default_capabilities(self):
        """测试默认能力"""
        model = ClaudeModel(
            model_path="claude-3-5-sonnet-20241022",
            api_key="sk-test-key"
        )

        capabilities = model.get_capabilities()

        assert capabilities.supports_streaming == True
        assert capabilities.supports_function_calling == True
        assert capabilities.supports_vision == True
        assert capabilities.supports_json_mode == True
        assert capabilities.max_context_length == 200000
        assert capabilities.max_new_tokens == 2048

    def test_haiku_capabilities(self):
        """测试 Haiku 模型能力"""
        model = ClaudeModel(
            model_path="claude-3-haiku-20240307",
            api_key="sk-test-key"
        )

        capabilities = model.get_capabilities()

        assert capabilities.supports_streaming == True
        assert capabilities.supports_function_calling == True
        assert capabilities.supports_vision == True
        assert capabilities.max_context_length == 200000

    def test_context_length_detection(self):
        """测试上下文长度检测"""
        model_opus = ClaudeModel(
            model_path="claude-3-opus-20240229",
            api_key="sk-test-key"
        )
        assert model_opus._get_context_length() == 200000

        model_sonnet = ClaudeModel(
            model_path="claude-3-sonnet-20240229",
            api_key="sk-test-key"
        )
        assert model_sonnet._get_context_length() == 200000

        model_haiku = ClaudeModel(
            model_path="claude-3-haiku-20240307",
            api_key="sk-test-key"
        )
        assert model_haiku._get_context_length() == 200000

        model_unknown = ClaudeModel(
            model_path="unknown-model",
            api_key="sk-test-key"
        )
        assert model_unknown._get_context_length() == 100000


class TestClaudeModelDevice:
    """测试 Claude 模型设备"""

    def test_get_device(self):
        """测试获取设备"""
        model = ClaudeModel(
            model_path="claude-3-5-sonnet-20241022",
            api_key="sk-test-key"
        )

        assert model.get_device() == "api"


class TestClaudeModelMemory:
    """测试 Claude 模型内存"""

    def test_get_memory_usage(self):
        """测试获取内存使用"""
        model = ClaudeModel(
            model_path="claude-3-5-sonnet-20241022",
            api_key="sk-test-key"
        )

        memory = model.get_memory_usage()

        assert memory["allocated"] == 0.0
        assert memory["reserved"] == 0.0
        assert memory["system_mem"] == 0.0


class TestClaudeModelContextManager:
    """测试 Claude 模型上下文管理器"""

    @patch("enlighten.interfaces.providers.claude.httpx.Client")
    def test_context_manager(self, mock_client_class):
        """测试上下文管理器"""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        with ClaudeModel(
            model_path="claude-3-5-sonnet-20241022",
            api_key="sk-test-key"
        ) as model:
            assert model.is_available() == True

        mock_client.close.assert_called_once()


class TestClaudeModelAvailableModels:
    """测试 Claude 可用模型列表"""

    def test_get_available_models(self):
        """测试获取可用模型列表"""
        model = ClaudeModel(
            model_path="claude-3-5-sonnet-20241022",
            api_key="sk-test-key"
        )

        models = model.get_available_models()

        assert len(models) > 0
        assert "claude-3-5-sonnet-20241022" in models
        assert "claude-3-opus-20240229" in models


class TestClaudeModelMessageConversion:
    """测试 Claude 消息格式转换"""

    def test_convert_user_message(self):
        """测试转换用户消息"""
        model = ClaudeModel(
            model_path="claude-3-5-sonnet-20241022",
            api_key="sk-test-key"
        )

        messages = [{"role": "user", "content": "Hello"}]
        converted = model._convert_messages_format(messages)

        assert len(converted) == 1
        assert converted[0]["role"] == "user"
        assert converted[0]["content"] == "Hello"

    def test_convert_assistant_message(self):
        """测试转换助手消息"""
        model = ClaudeModel(
            model_path="claude-3-5-sonnet-20241022",
            api_key="sk-test-key"
        )

        messages = [{"role": "assistant", "content": "Hi there!"}]
        converted = model._convert_messages_format(messages)

        assert len(converted) == 1
        assert converted[0]["role"] == "assistant"
        assert converted[0]["content"] == "Hi there!"

    def test_convert_system_message_ignored(self):
        """测试转换时忽略系统消息"""
        model = ClaudeModel(
            model_path="claude-3-5-sonnet-20241022",
            api_key="sk-test-key"
        )

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"}
        ]
        converted = model._convert_messages_format(messages)

        assert len(converted) == 1
        assert converted[0]["role"] == "user"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
