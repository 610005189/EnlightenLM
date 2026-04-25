"""LLaMA 模型接口测试"""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock

from enlighten.interfaces.base import ModelProvider, ModelState, ModelCapability, ModelMetadata
from enlighten.interfaces.config import LLaMAModelConfig
from enlighten.interfaces.factory import ModelFactory
from enlighten.interfaces.errors import ModelLoadError
from enlighten.interfaces.providers.llama import LLaMAModel


class TestLLaMAModelConfig:
    """测试 LLaMA 模型配置"""

    def test_default_config(self):
        """测试默认配置"""
        config = LLaMAModelConfig(
            provider=ModelProvider.LLAMA,
            model_path="meta-llama/Llama-3.2-1B"
        )

        assert config.provider == ModelProvider.LLAMA
        assert config.model_path == "meta-llama/Llama-3.2-1B"
        assert config.device == "auto"
        assert config.dtype == "auto"
        assert config.max_tokens == 2048
        assert config.temperature == 0.7
        assert config.load_in_8bit == False
        assert config.load_in_4bit == False
        assert config.use_flash_attention == True

    def test_quantized_config(self):
        """测试量化配置"""
        config = LLaMAModelConfig(
            provider=ModelProvider.LLAMA,
            model_path="meta-llama/Llama-3.2-1B",
            load_in_8bit=True
        )

        assert config.load_in_8bit == True
        assert config.load_in_4bit == False

    def test_custom_config(self):
        """测试自定义配置"""
        config = LLaMAModelConfig(
            provider=ModelProvider.LLAMA,
            model_path="meta-llama/Llama-3.2-1B",
            device="cuda:0",
            max_seq_len=4096,
            trust_remote_code=True
        )

        assert config.device == "cuda:0"
        assert config.max_seq_len == 4096
        assert config.trust_remote_code == True


class TestLLaMAModelInterface:
    """测试 LLaMA 模型接口"""

    def test_model_factory_registration(self):
        """测试模型工厂注册"""
        providers = ModelFactory.get_registered_providers()
        assert ModelProvider.LLAMA in providers
        assert ModelProvider.OLLAMA in providers

    def test_create_llama_model(self):
        """测试通过工厂创建 LLaMA 模型"""
        config = LLaMAModelConfig(
            provider=ModelProvider.LLAMA,
            model_path="meta-llama/Llama-3.2-1B"
        )

        model = ModelFactory.create(config)
        assert model.model_path == "meta-llama/Llama-3.2-1B"
        assert model._state == ModelState.UNINITIALIZED

    def test_create_from_dict(self):
        """测试从字典创建模型"""
        config_dict = {
            "provider": "llama",
            "model_path": "meta-llama/Llama-3.2-1B",
            "device": "cpu"
        }

        model = ModelFactory.create(config_dict)
        assert model.model_path == "meta-llama/Llama-3.2-1B"
        assert model.device == "cpu"


class TestLLaMAModelLoading:
    """测试 LLaMA 模型加载"""

    @patch("enlighten.interfaces.providers.llama.AutoTokenizer")
    @patch("enlighten.interfaces.providers.llama.AutoModelForCausalLM")
    def test_load_success(self, mock_model_class, mock_tokenizer_class):
        """测试成功加载模型"""
        from enlighten.interfaces.providers.llama import LLaMAModel

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "</s>"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model.config = MagicMock()
        mock_model.config.max_position_embeddings = 2048
        mock_model.config.vocab_size = 32000
        mock_model.config.hidden_size = 2048
        mock_model.config.num_hidden_layers = 22
        mock_model.config.num_attention_heads = 16
        mock_model.config.head_dim = 0
        mock_model.eval = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model

        model = LLaMAModel(
            model_path="meta-llama/Llama-3.2-1B",
            device="cpu"
        )

        result = model.load()

        assert result == True
        assert model._state == ModelState.READY
        assert model._tokenizer is not None
        assert model._model is not None
        assert model._metadata is not None
        assert model._metadata.provider == ModelProvider.LLAMA

    @patch("enlighten.interfaces.providers.llama.AutoTokenizer")
    def test_load_failure(self, mock_tokenizer_class):
        """测试加载失败"""
        from enlighten.interfaces.providers.llama import LLaMAModel

        mock_tokenizer_class.from_pretrained.side_effect = Exception("Model not found")

        model = LLaMAModel(
            model_path="invalid/model/path",
            device="cpu"
        )

        with pytest.raises(ModelLoadError):
            model.load()

        assert model._state == ModelState.ERROR


class TestLLaMAModelGeneration:
    """测试 LLaMA 模型推理"""

    def setup_method(self):
        """设置测试环境"""
        self.mock_model = MagicMock()
        self.mock_model.device = "cpu"
        self.mock_model.config = MagicMock()
        self.mock_model.config.max_position_embeddings = 2048
        self.mock_model.config.vocab_size = 32000
        self.mock_model.config.hidden_size = 2048
        self.mock_model.config.num_hidden_layers = 22
        self.mock_model.config.num_attention_heads = 16
        self.mock_model.config.head_dim = 0

        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer.pad_token = None
        self.mock_tokenizer.eos_token = "</s>"
        self.mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}

    @patch("enlighten.interfaces.providers.llama.AutoTokenizer")
    @patch("enlighten.interfaces.providers.llama.AutoModelForCausalLM")
    def test_generate_without_loaded_model(self, mock_model_class, mock_tokenizer_class):
        """测试未加载模型时生成应该失败"""
        from enlighten.interfaces.providers.llama import LLaMAModel

        model = LLaMAModel(
            model_path="meta-llama/Llama-3.2-1B",
            device="cpu"
        )

        with pytest.raises(RuntimeError, match="Model not loaded"):
            model.generate("Hello")

    @patch("enlighten.interfaces.providers.llama.AutoTokenizer")
    @patch("enlighten.interfaces.providers.llama.AutoModelForCausalLM")
    def test_generate_with_loaded_model(self, mock_model_class, mock_tokenizer_class):
        """测试已加载模型的生成"""
        import torch
        from enlighten.interfaces.providers.llama import LLaMAModel

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "</s>"
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_tokenizer.decode.return_value = "Generated text"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model.config = MagicMock()
        mock_model.config.max_position_embeddings = 2048
        mock_model.config.vocab_size = 32000
        mock_model.config.hidden_size = 2048
        mock_model.config.num_hidden_layers = 22
        mock_model.config.num_attention_heads = 16
        mock_model.config.head_dim = 0
        mock_model.eval = MagicMock()

        mock_output = MagicMock()
        mock_output.__getitem__ = MagicMock(return_value=torch.tensor([[1, 2, 3, 4, 5]]))
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])

        mock_model_class.from_pretrained.return_value = mock_model

        model = LLaMAModel(
            model_path="meta-llama/Llama-3.2-1B",
            device="cpu"
        )
        model.load()

        result = model.generate("Hello")

        assert isinstance(result, str)


class TestLLaMAModelState:
    """测试 LLaMA 模型状态管理"""

    def test_initial_state(self):
        """测试初始状态"""
        from enlighten.interfaces.providers.llama import LLaMAModel

        model = LLaMAModel(
            model_path="meta-llama/Llama-3.2-1B"
        )

        assert model._state == ModelState.UNINITIALIZED
        assert model.is_available() == False

    @patch("enlighten.interfaces.providers.llama.AutoTokenizer")
    @patch("enlighten.interfaces.providers.llama.AutoModelForCausalLM")
    def test_ready_state_after_load(self, mock_model_class, mock_tokenizer_class):
        """测试加载后状态为 READY"""
        from enlighten.interfaces.providers.llama import LLaMAModel

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "</s>"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model.config = MagicMock()
        mock_model.config.max_position_embeddings = 2048
        mock_model.config.vocab_size = 32000
        mock_model.config.hidden_size = 2048
        mock_model.config.num_hidden_layers = 22
        mock_model.config.num_attention_heads = 16
        mock_model.config.head_dim = 0
        mock_model.eval = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model

        model = LLaMAModel(
            model_path="meta-llama/Llama-3.2-1B"
        )
        model.load()

        assert model._state == ModelState.READY
        assert model.is_available() == True


class TestLLaMAModelCapabilities:
    """测试 LLaMA 模型能力"""

    def test_default_capabilities(self):
        """测试默认能力"""
        from enlighten.interfaces.providers.llama import LLaMAModel

        model = LLaMAModel(
            model_path="meta-llama/Llama-3.2-1B"
        )

        capabilities = model.get_capabilities()

        assert capabilities.supports_streaming == True
        assert capabilities.supports_batch_inference == True
        assert capabilities.max_context_length == 2048
        assert capabilities.max_new_tokens == 2048

    def test_custom_capabilities(self):
        """测试自定义能力"""
        from enlighten.interfaces.providers.llama import LLaMAModel

        model = LLaMAModel(
            model_path="meta-llama/Llama-3.2-1B",
            config={"max_tokens": 4096, "use_flash_attention": True}
        )

        capabilities = model.get_capabilities()

        assert capabilities.supports_prefix_caching == True
        assert capabilities.max_new_tokens == 4096


class TestLLaMAModelContextManager:
    """测试 LLaMA 模型上下文管理器"""

    @patch("enlighten.interfaces.providers.llama.AutoTokenizer")
    @patch("enlighten.interfaces.providers.llama.AutoModelForCausalLM")
    def test_context_manager(self, mock_model_class, mock_tokenizer_class):
        """测试上下文管理器"""
        from enlighten.interfaces.providers.llama import LLaMAModel

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "</s>"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model.config = MagicMock()
        mock_model.config.max_position_embeddings = 2048
        mock_model.config.vocab_size = 32000
        mock_model.config.hidden_size = 2048
        mock_model.config.num_hidden_layers = 22
        mock_model.config.num_attention_heads = 16
        mock_model.config.head_dim = 0
        mock_model.eval = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model

        with LLaMAModel(model_path="meta-llama/Llama-3.2-1B") as model:
            assert model.is_available() == True

        assert model._state == ModelState.UNLOADED


if __name__ == "__main__":
    pytest.main([__file__, "-v"])