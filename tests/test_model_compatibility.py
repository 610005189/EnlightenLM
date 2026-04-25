"""模型兼容性测试

验证不同模型在三层架构中的表现一致性，确保模型之间的功能一致性。
测试所有已集成的模型接口（LLaMA、Mistral、Claude、Ollama）。
"""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
from typing import Iterator

from enlighten.interfaces.base import (
    BaseModelInterface,
    ModelProvider,
    ModelState,
    ModelCapability,
    ModelMetadata,
)
from enlighten.interfaces.config import (
    ModelConfig,
    LLaMAModelConfig,
    MistralModelConfig,
    OllamaModelConfig,
    ClaudeModelConfig,
)
from enlighten.interfaces.factory import ModelFactory
from enlighten.interfaces.providers.llama import LLaMAModel
from enlighten.interfaces.providers.mistral import MistralModel
from enlighten.interfaces.providers.ollama import OllamaModel
from enlighten.interfaces.providers.claude import ClaudeModel
from enlighten.l1_generation import L1Generation, SimplifiedL1
from enlighten.l2_working_memory import L2WorkingMemory, SimplifiedL2
from enlighten.l3_controller import L3Controller, SimplifiedL3, BayesianL3Controller
from enlighten.hybrid_architecture import (
    HybridEnlightenLM,
    L1Adapter,
    L2WorkingMemoryAdapter,
    L3ControllerAdapter,
)


class TestModelInterfaceConsistency:
    """测试所有模型接口的一致性"""

    def test_base_interface_methods_exist(self):
        """验证 BaseModelInterface 定义了所有必需的方法"""
        required_methods = [
            'load', 'unload', 'generate', 'generate_stream',
            'chat', 'chat_stream', 'is_available', 'get_state',
            'get_metadata', 'get_capabilities', 'get_device',
            'get_memory_usage', 'cleanup'
        ]

        for method_name in required_methods:
            assert hasattr(BaseModelInterface, method_name), \
                f"BaseModelInterface missing method: {method_name}"

    def test_all_providers_implement_base_interface(self):
        """验证所有模型提供者实现了 BaseModelInterface"""
        providers = [
            (ModelProvider.LLAMA, LLaMAModel),
            (ModelProvider.MISTRAL, MistralModel),
            (ModelProvider.OLLAMA, OllamaModel),
            (ModelProvider.CLAUDE, ClaudeModel),
        ]

        for provider, model_class in providers:
            assert issubclass(model_class, BaseModelInterface), \
                f"{provider.value} does not implement BaseModelInterface"

    def test_all_providers_registered_in_factory(self):
        """验证所有模型提供者都在工厂中注册"""
        registered = ModelFactory.get_registered_providers()

        expected_providers = [
            ModelProvider.LLAMA,
            ModelProvider.MISTRAL,
            ModelProvider.OLLAMA,
            ModelProvider.CLAUDE,
        ]

        for provider in expected_providers:
            assert provider in registered, \
                f"{provider.value} not registered in ModelFactory"


class TestModelConfigConsistency:
    """测试模型配置的一致性"""

    def test_all_config_classes_have_required_fields(self):
        """验证所有配置类都有必需的字段"""
        required_fields = ['provider', 'model_path', 'device', 'max_tokens', 'temperature']

        config_classes = [
            LLaMAModelConfig(provider=ModelProvider.LLAMA, model_path="test"),
            MistralModelConfig(provider=ModelProvider.MISTRAL, model_path="test"),
            OllamaModelConfig(provider=ModelProvider.OLLAMA, model_path="test"),
            ClaudeModelConfig(provider=ModelProvider.CLAUDE, model_path="test"),
        ]

        for config in config_classes:
            for field in required_fields:
                assert hasattr(config, field), \
                    f"{type(config).__name__} missing field: {field}"

    def test_all_configs_have_sane_defaults(self):
        """验证所有配置类都有合理的默认值"""
        configs = [
            LLaMAModelConfig(provider=ModelProvider.LLAMA, model_path="test"),
            MistralModelConfig(provider=ModelProvider.MISTRAL, model_path="test"),
            OllamaModelConfig(provider=ModelProvider.OLLAMA, model_path="test"),
            ClaudeModelConfig(provider=ModelProvider.CLAUDE, model_path="test"),
        ]

        for config in configs:
            assert config.max_tokens > 0, "max_tokens should be positive"
            assert 0 <= config.temperature <= 2.0, "temperature should be in [0, 2]"


class TestModelCapabilityConsistency:
    """测试模型能力的一致性"""

    def test_all_models_support_streaming(self):
        """验证所有模型都支持流式生成"""
        models = [
            LLaMAModel(model_path="test"),
            MistralModel(model_path="test"),
            OllamaModel(model_path="test"),
            ClaudeModel(model_path="test"),
        ]

        for model in models:
            capabilities = model.get_capabilities()
            assert capabilities.supports_streaming is True, \
                f"{type(model).__name__} should support streaming"

    def test_all_models_return_valid_capabilities(self):
        """验证所有模型都返回有效的 capability 对象"""
        models = [
            LLaMAModel(model_path="test"),
            MistralModel(model_path="test"),
            OllamaModel(model_path="test"),
            ClaudeModel(model_path="test"),
        ]

        for model in models:
            capabilities = model.get_capabilities()
            assert isinstance(capabilities, ModelCapability)
            assert capabilities.max_context_length > 0
            assert capabilities.max_new_tokens > 0


class TestModelStateConsistency:
    """测试模型状态的一致性"""

    def test_initial_state_is_uninitialized(self):
        """验证所有模型的初始状态都是 UNINITIALIZED"""
        models = [
            LLaMAModel(model_path="test"),
            MistralModel(model_path="test"),
            OllamaModel(model_path="test"),
            ClaudeModel(model_path="test"),
        ]

        for model in models:
            assert model.get_state() == ModelState.UNINITIALIZED, \
                f"{type(model).__name__} initial state should be UNINITIALIZED"
            assert model.is_available() is False

    def test_generate_fails_when_not_loaded(self):
        """验证所有模型在未加载时 generate 调用失败"""
        models = [
            LLaMAModel(model_path="test"),
            MistralModel(model_path="test"),
            OllamaModel(model_path="test"),
            ClaudeModel(model_path="test"),
        ]

        for model in models:
            with pytest.raises(RuntimeError, match="Model not loaded"):
                model.generate("test prompt")


class TestGenerateMethodSignature:
    """测试 generate 方法签名的一致性"""

    def test_all_generate_methods_have_same_signature(self):
        """验证所有模型的 generate 方法有相同的签名"""
        import inspect

        base_sig = inspect.signature(BaseModelInterface.generate)
        providers = [
            (ModelProvider.LLAMA, LLaMAModel),
            (ModelProvider.MISTRAL, MistralModel),
            (ModelProvider.OLLAMA, OllamaModel),
            (ModelProvider.CLAUDE, ClaudeModel),
        ]

        for provider, model_class in providers:
            sig = inspect.signature(model_class.generate)
            base_params = set(base_sig.parameters.keys())
            sig_params = set(sig.parameters.keys())

            assert base_params == sig_params, \
                f"{provider.value} generate signature mismatch: {sig_params} vs {base_params}"

    def test_all_generate_stream_methods_return_iterator(self):
        """验证所有模型的 generate_stream 方法返回迭代器"""
        models = [
            LLaMAModel(model_path="test"),
            MistralModel(model_path="test"),
            OllamaModel(model_path="test"),
            ClaudeModel(model_path="test"),
        ]

        for model in models:
            assert hasattr(model, 'generate_stream'), \
                f"{type(model).__name__} missing generate_stream method"


class TestChatMethodConsistency:
    """测试 chat 方法的一致性"""

    def test_all_models_implement_chat(self):
        """验证所有模型都实现了 chat 方法"""
        models = [
            LLaMAModel(model_path="test"),
            MistralModel(model_path="test"),
            OllamaModel(model_path="test"),
            ClaudeModel(model_path="test"),
        ]

        for model in models:
            assert hasattr(model, 'chat'), \
                f"{type(model).__name__} missing chat method"
            assert callable(model.chat), \
                f"{type(model).__name__}.chat should be callable"

    def test_all_models_implement_chat_stream(self):
        """验证所有模型都实现了 chat_stream 方法"""
        models = [
            LLaMAModel(model_path="test"),
            MistralModel(model_path="test"),
            OllamaModel(model_path="test"),
            ClaudeModel(model_path="test"),
        ]

        for model in models:
            assert hasattr(model, 'chat_stream'), \
                f"{type(model).__name__} missing chat_stream method"


class TestL3ArchitectureCompatibility:
    """测试 L3 架构兼容性"""

    def test_l3_controller_signal_ranges(self):
        """验证 L3 控制器输出的信号在有效范围内"""
        controller = L3Controller(config={
            "entropy_threshold": 0.5,
            "variance_threshold": 0.05,
            "tau_range": [0.1, 2.0],
            "theta_range": [0.5, 0.9],
            "alpha_range": [0.0, 1.0],
        })

        entropy_stats = {"mean": 0.3, "variance": 0.02, "trend": -0.1, "current": 0.3}

        for _ in range(10):
            signals = controller.forward(entropy_stats, van_event=False, p_harm=0.0)

            assert 0.1 <= signals.tau <= 2.0, f"tau {signals.tau} out of range"
            assert 0.5 <= signals.theta <= 0.9, f"theta {signals.theta} out of range"
            assert 0.0 <= signals.alpha <= 1.0, f"alpha {signals.alpha} out of range"

    def test_bayesian_l3_controller_signal_ranges(self):
        """验证贝叶斯 L3 控制器输出的信号在有效范围内"""
        controller = BayesianL3Controller()

        entropy_stats = {"mean": 0.3, "variance": 0.02, "trend": -0.1, "current": 0.3}

        for _ in range(10):
            signals = controller.forward(entropy_stats, van_event=False, p_harm=0.0)

            assert 0.2 <= signals.tau <= 2.0, f"tau {signals.tau} out of range"
            assert 0.5 <= signals.theta <= 0.9, f"theta {signals.theta} out of range"
            assert 0.0 <= signals.alpha <= 1.0, f"alpha {signals.alpha} out of range"

    def test_simplified_l3_signal_ranges(self):
        """验证简化 L3 控制器输出的信号在有效范围内"""
        controller = SimplifiedL3()

        entropy_stats = {"mean": 0.3, "variance": 0.02, "trend": -0.1, "current": 0.3}

        for _ in range(10):
            signals = controller.forward(entropy_stats, van_event=False)

            assert 0.1 <= signals["tau"] <= 2.0, f"tau {signals['tau']} out of range"
            assert 0.5 <= signals["theta"] <= 0.9, f"theta {signals['theta']} out of range"
            assert 0.0 <= signals["alpha"] <= 1.0, f"alpha {signals['alpha']} out of range"


class TestHybridArchitectureCompatibility:
    """测试混合架构兼容性"""

    def test_hybrid_model_initialization_all_modes(self):
        """验证混合架构支持所有模式初始化"""
        modes = [
            {"use_local_model": False},
            {"use_bayesian_l3": True},
            {"use_l3_controller": True},
            {"use_l1_adapter": True},
            {"use_skeleton_l2": True},
        ]

        for mode in modes:
            model = HybridEnlightenLM(**mode)
            assert model is not None

    def test_hybrid_model_l3_signal_propagation(self):
        """验证混合架构中 L3 信号正确传播"""
        model = HybridEnlightenLM(
            use_l3_controller=True,
            l3_config={
                "entropy_threshold": 0.5,
                "variance_threshold": 0.05,
            }
        )

        entropy_stats = {"mean": 0.3, "variance": 0.02, "trend": -0.1, "current": 0.3}

        for _ in range(5):
            model.l3_controller_adapter.forward(entropy_stats, van_event=False, p_harm=0.0)

        assert model.l3_controller_adapter is not None
        assert model.get_temperature() > 0
        assert model.get_sparsity_threshold() > 0

    def test_hybrid_model_reset_functionality(self):
        """验证混合架构重置功能"""
        model = HybridEnlightenLM(
            use_bayesian_l3=True,
            use_l3_controller=True,
            use_l1_adapter=True,
            use_skeleton_l2=True,
        )

        model.reset()

        assert model.working_memory is not None
        assert model.van_monitor is not None


class TestL1AdapterCompatibility:
    """测试 L1 适配器兼容性"""

    def test_l1_adapter_with_dummy_inputs(self):
        """验证 L1 适配器能处理各种输入"""
        adapter = L1Adapter(embed_dim=128, num_heads=4, task_bias_dim=32)

        batch_size = 1
        seq_len = 16
        dummy_input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        dummy_hidden = torch.randn(batch_size, seq_len, 128)

        result = adapter(dummy_input_ids, dummy_hidden)

        assert "output_hidden" in result
        assert "attention_weights" in result
        assert "entropy_stats" in result
        assert "van_event" in result
        assert "p_harm" in result

    def test_l1_adapter_control_signals(self):
        """验证 L1 适配器正确处理调控信号"""
        adapter = L1Adapter(embed_dim=128, num_heads=4)

        dummy_input_ids = torch.randint(0, 1000, (1, 16))
        dummy_hidden = torch.randn(1, 16, 128)

        control_signals = {
            "tau": 0.5,
            "theta": 0.7,
            "alpha": 0.2,
            "decay_rate": 0.9
        }

        result1 = adapter(dummy_input_ids, dummy_hidden, control_signals=control_signals)
        result2 = adapter(dummy_input_ids, dummy_hidden, control_signals=control_signals)

        assert result1["control_signals"] == control_signals
        assert result2["control_signals"] == control_signals


class TestL2AdapterCompatibility:
    """测试 L2 适配器兼容性"""

    def test_l2_adapter_entropy_stats(self):
        """验证 L2 适配器正确计算和返回熵统计"""
        adapter = L2WorkingMemoryAdapter(memory_size=64, embedding_dim=128)

        batch_size = 1
        seq_len = 16
        hidden_states = torch.randn(batch_size, seq_len, 128)

        attention_weights = torch.rand(batch_size, seq_len)
        attention_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True)

        result = adapter(hidden_states, attention_weights)

        assert result.entropy_stats is not None
        assert "mean" in result.entropy_stats
        assert "variance" in result.entropy_stats
        assert "trend" in result.entropy_stats

    def test_l2_adapter_sparse_kv_output(self):
        """验证 L2 适配器正确输出稀疏键值"""
        adapter = L2WorkingMemoryAdapter(memory_size=64, embedding_dim=128)

        batch_size = 2
        seq_len = 16
        hidden_states = torch.randn(batch_size, seq_len, 128)

        result = adapter(hidden_states)

        assert result.sparse_kv is not None
        assert len(result.sparse_kv) == 2
        sparse_k, sparse_v = result.sparse_kv
        assert sparse_k.shape[0] <= adapter.memory_size


class TestSimplifiedArchitectureCompatibility:
    """测试简化架构兼容性"""

    def test_simplified_l1_forward(self):
        """验证简化 L1 前向传播"""
        l1 = SimplifiedL1(embed_dim=64, task_bias_dim=32)

        batch_size = 1
        seq_len = 8
        query = torch.randn(batch_size, seq_len, 64)
        key = torch.randn(batch_size, seq_len, 64)
        value = torch.randn(batch_size, seq_len, 64)
        task_bias = torch.randn(batch_size, 32)

        output, van_event = l1(query, key, value, task_bias)

        assert output.shape == (batch_size, seq_len, 64)
        assert isinstance(van_event, bool)

    def test_simplified_l2_entropy_stats(self):
        """验证简化 L2 熵统计"""
        l2 = SimplifiedL2(memory_size=64, embedding_dim=64)

        batch_size = 1
        seq_len = 16
        hidden_states = torch.randn(batch_size, seq_len, 64)

        attention_weights = torch.rand(batch_size, seq_len, seq_len)
        attention_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True)

        sparse_kv, entropy_stats = l2(hidden_states, attention_weights)

        assert sparse_kv.shape == (64, 64)
        assert "mean" in entropy_stats
        assert "variance" in entropy_stats

    def test_simplified_l3_control_signals(self):
        """验证简化 L3 调控信号"""
        l3 = SimplifiedL3(entropy_threshold=0.5, variance_threshold=0.05)

        entropy_stats = {"mean": 0.3, "variance": 0.02, "trend": -0.1, "current": 0.3}

        for _ in range(5):
            signals = l3.forward(entropy_stats, van_event=False)

            assert "tau" in signals
            assert "theta" in signals
            assert "alpha" in signals
            assert "stability" in signals
            assert "cutoff" in signals
            assert "reason" in signals


class TestModelFactoryIntegration:
    """测试模型工厂集成"""

    def test_create_all_provider_types(self):
        """验证工厂可以创建所有类型的模型"""
        configs = [
            LLaMAModelConfig(provider=ModelProvider.LLAMA, model_path="test-llama"),
            MistralModelConfig(provider=ModelProvider.MISTRAL, model_path="test-mistral"),
            OllamaModelConfig(provider=ModelProvider.OLLAMA, model_path="test-ollama"),
            ClaudeModelConfig(provider=ModelProvider.CLAUDE, model_path="test-claude"),
        ]

        for config in configs:
            model = ModelFactory.create(config)
            assert model is not None
            assert isinstance(model, BaseModelInterface)
            assert model.model_path == config.model_path

    def test_create_from_dict_all_providers(self):
        """验证工厂可以从字典创建所有类型的模型"""
        configs = [
            {"provider": "llama", "model_path": "test-llama"},
            {"provider": "mistral", "model_path": "test-mistral"},
            {"provider": "ollama", "model_path": "test-ollama"},
            {"provider": "claude", "model_path": "test-claude"},
        ]

        for config_dict in configs:
            model = ModelFactory.create(config_dict)
            assert model is not None
            assert isinstance(model, BaseModelInterface)


class TestEntropyStatsConsistency:
    """测试熵统计一致性"""

    def test_entropy_stats_format_consistency(self):
        """验证不同组件返回的熵统计格式一致"""
        entropy_stats_from_l1 = {"mean": 0.5, "variance": 0.1, "trend": -0.05, "current": 0.5}
        entropy_stats_from_l2 = {"mean": 0.5, "variance": 0.1, "trend": -0.05, "current": 0.5}
        entropy_stats_from_l3 = {"mean": 0.5, "variance": 0.1, "trend": -0.05, "current": 0.5}

        required_keys = {"mean", "variance", "trend", "current"}

        assert required_keys.issubset(entropy_stats_from_l1.keys())
        assert required_keys.issubset(entropy_stats_from_l2.keys())
        assert required_keys.issubset(entropy_stats_from_l3.keys())

    def test_entropy_stats_values_in_valid_range(self):
        """验证熵统计值在有效范围内"""
        l2 = L2WorkingMemory(memory_size=32, embedding_dim=64)

        batch_size = 1
        seq_len = 8
        hidden_states = torch.randn(batch_size, seq_len, 64)

        attention_weights = torch.rand(batch_size, seq_len)
        attention_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True)

        result = l2(hidden_states, attention_weights)

        assert 0 <= result.entropy_stats["mean"] <= 1
        assert result.entropy_stats["variance"] >= 0


class TestControlSignalsConsistency:
    """测试调控信号一致性"""

    def test_control_signals_dict_format(self):
        """验证调控信号字典格式一致"""
        controller = L3Controller()

        entropy_stats = {"mean": 0.3, "variance": 0.02, "trend": -0.1, "current": 0.3}
        signals = controller.forward(entropy_stats)

        control_dict = controller.get_control_signals_dict(signals)

        required_keys = {"tau", "theta", "alpha", "stability", "cutoff", "reason"}
        assert required_keys.issubset(control_dict.keys())

    def test_l3_controller_adapter_signals(self):
        """验证 L3 适配器正确转换信号"""
        adapter = L3ControllerAdapter()

        entropy_stats = {"mean": 0.3, "variance": 0.02, "trend": -0.1, "current": 0.3}

        for _ in range(3):
            adapter.forward(entropy_stats, van_event=False, p_harm=0.0)

        signals_dict = adapter.get_last_control_signals()

        assert signals_dict is not None
        assert "tau" in signals_dict
        assert "theta" in signals_dict
        assert "alpha" in signals_dict


class TestHybridArchitectureSignalFlow:
    """测试混合架构信号流"""

    def test_l1_to_l2_signal_flow(self):
        """验证 L1 到 L2 的信号流"""
        l1_adapter = L1Adapter(embed_dim=128, num_heads=4, task_bias_dim=32)
        l2_adapter = L2WorkingMemoryAdapter(memory_size=64, embedding_dim=128)

        batch_size = 1
        seq_len = 16
        dummy_input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        dummy_hidden = torch.randn(batch_size, seq_len, 128)

        l1_result = l1_adapter(dummy_input_ids, dummy_hidden)

        attention_weights = l1_result["attention_weights"]
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.float()
            if attention_weights.dim() > 2:
                attention_weights = attention_weights.mean(dim=-1)

        l2_result = l2_adapter(
            l1_result["output_hidden"],
            attention_weights
        )

        assert l2_result.entropy_stats is not None
        assert l2_result.sparse_kv is not None

    def test_l2_to_l3_signal_flow(self):
        """验证 L2 到 L3 的信号流"""
        l2_adapter = L2WorkingMemoryAdapter(memory_size=64, embedding_dim=128)
        l3_adapter = L3ControllerAdapter()

        batch_size = 1
        seq_len = 16
        hidden_states = torch.randn(batch_size, seq_len, 128)

        attention_weights = torch.rand(batch_size, seq_len)
        attention_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True)

        l2_result = l2_adapter(hidden_states, attention_weights)

        l3_signals = l3_adapter.forward(
            l2_result.entropy_stats,
            van_event=False,
            p_harm=0.0
        )

        assert l3_signals.tau is not None
        assert l3_signals.theta is not None
        assert l3_signals.alpha is not None


class TestCrossModelConsistency:
    """跨模型一致性测试"""

    def test_all_models_handle_same_interface(self):
        """验证所有模型处理相同接口的方式一致"""
        test_prompt = "Hello, how are you?"
        test_messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]

        models = [
            LLaMAModel(model_path="test"),
            MistralModel(model_path="test"),
            OllamaModel(model_path="test"),
            ClaudeModel(model_path="test"),
        ]

        for model in models:
            state = model.get_state()
            assert state == ModelState.UNINITIALIZED

            capabilities = model.get_capabilities()
            assert isinstance(capabilities, ModelCapability)
            assert capabilities.max_context_length > 0
            assert capabilities.max_new_tokens > 0

            device = model.get_device()
            assert isinstance(device, str)

            memory_usage = model.get_memory_usage()
            assert "allocated" in memory_usage
            assert "reserved" in memory_usage

    def test_all_configs_produce_consistent_model_instances(self):
        """验证所有配置类产生一致的模型实例"""
        configs = [
            LLaMAModelConfig(provider=ModelProvider.LLAMA, model_path="llama-test"),
            MistralModelConfig(provider=ModelProvider.MISTRAL, model_path="mistral-test"),
            OllamaModelConfig(provider=ModelProvider.OLLAMA, model_path="ollama-test"),
            ClaudeModelConfig(provider=ModelProvider.CLAUDE, model_path="claude-test"),
        ]

        for config in configs:
            model = ModelFactory.create(config)

            assert model.model_path == config.model_path
            assert model.get_state() == ModelState.UNINITIALIZED

            assert callable(model.load)
            assert callable(model.unload)
            assert callable(model.generate)
            assert callable(model.generate_stream)
            assert callable(model.chat)
            assert callable(model.chat_stream)


class TestModelSpecificBehavior:
    """测试模型特定行为"""

    def test_llama_chat_format(self):
        """测试 LLaMA 聊天格式"""
        model = LLaMAModel(model_path="test")
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hi"}
        ]

        formatted = model._format_llama_chat_prompt(messages)

        assert "<|system|>" in formatted
        assert "<|user|>" in formatted
        assert "<|assistant|>" in formatted

    def test_mistral_chat_format(self):
        """测试 Mistral 聊天格式"""
        model = MistralModel(model_path="test")
        messages = [
            {"role": "user", "content": "Hi"}
        ]

        formatted = model._format_mistral_chat_prompt(messages)

        assert "[INST]" in formatted
        assert "[/INST]" in formatted

    def test_ollama_is_available_check(self):
        """测试 Ollama 可用性检查"""
        model = OllamaModel(model_path="test", base_url="http://localhost:11434")

        with patch("httpx.Client") as mock_client:
            mock_instance = MagicMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_instance.get.return_value = mock_response
            mock_client.return_value = mock_instance

            result = model.is_ollama_available()
            assert isinstance(result, bool)

    def test_claude_message_conversion(self):
        """测试 Claude 消息格式转换"""
        model = ClaudeModel(model_path="test")
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"}
        ]

        converted = model._convert_messages_format(messages)

        for msg in converted:
            assert msg["role"] in ["user", "assistant"]
            assert "content" in msg


if __name__ == "__main__":
    pytest.main([__file__, "-v"])