"""
DeepSeek 适配器集成测试
测试 DeepSeek 适配器与 EnlightenLM 的集成
"""

import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enlighten.adapters import DeepSeekAdapter, AdapterConfig, AdapterType
from enlighten.adapters.deepseek_adapter import DeepSeekConfig


def test_deepseek_adapter_init():
    """测试 DeepSeek 适配器初始化"""
    print("=== 测试 DeepSeek 适配器初始化 ===")

    # 测试默认配置
    adapter = DeepSeekAdapter()
    assert adapter is not None
    assert adapter.config.adapter_type == AdapterType.DEEPSEEK_V3
    assert adapter.deepseek_config.model_name == "deepseek-ai/DeepSeek-V3"
    print("✅ 默认配置初始化成功")

    # 测试 V4 配置
    config = AdapterConfig(adapter_type=AdapterType.DEEPSEEK_V4)
    adapter_v4 = DeepSeekAdapter(config=config)
    assert adapter_v4.config.adapter_type == AdapterType.DEEPSEEK_V4
    assert adapter_v4.deepseek_config.model_name == "deepseek-ai/DeepSeek-V3"  # 默认为 V3，需要手动切换
    print("✅ V4 配置初始化成功")

    # 测试模型版本切换
    adapter.switch_model_version("v4")
    assert adapter.config.adapter_type == AdapterType.DEEPSEEK_V4
    assert adapter.deepseek_config.model_name == "deepseek-ai/DeepSeek-V4"
    print("✅ 模型版本切换成功")

    adapter.switch_model_version("v3")
    assert adapter.config.adapter_type == AdapterType.DEEPSEEK_V3
    assert adapter.deepseek_config.model_name == "deepseek-ai/DeepSeek-V3"
    print("✅ 模型版本切换回 V3 成功")


def test_deepseek_adapter_attention():
    """测试 DeepSeek 适配器的注意力计算"""
    print("\n=== 测试 DeepSeek 适配器注意力计算 ===")

    adapter = DeepSeekAdapter()

    # 测试内存高效注意力
    batch_size = 1
    num_heads = 8
    seq_len = 64
    head_dim = 64

    query = torch.randn(batch_size, num_heads, seq_len, head_dim)
    key = torch.randn(batch_size, num_heads, seq_len, head_dim)
    value = torch.randn(batch_size, num_heads, seq_len, head_dim)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

    output = adapter.compute_attention(query, key, value, attention_mask)
    assert output.shape == (batch_size, num_heads, seq_len, head_dim)
    print("✅ 内存高效注意力计算成功")

    # 测试稀疏注意力
    sparse_indices = [0, 10, 20, 30, 40, 50, 60, 63]
    sparse_output = adapter.compute_sparse_attention(
        query, key, value, sparse_indices, attention_mask
    )
    assert sparse_output.shape == (batch_size, num_heads, seq_len, head_dim)
    print("✅ 稀疏注意力计算成功")


def test_deepseek_adapter_prefix_caching():
    """测试 DeepSeek 适配器的前缀缓存"""
    print("\n=== 测试 DeepSeek 适配器前缀缓存 ===")

    adapter = DeepSeekAdapter()

    # 测试前缀缓存启用/禁用
    assert adapter.prefix_caching_enabled is True
    adapter.disable_prefix_caching()
    assert adapter.prefix_caching_enabled is False
    adapter.enable_prefix_caching()
    assert adapter.prefix_caching_enabled is True
    print("✅ 前缀缓存状态切换成功")

    # 测试缓存操作
    test_prompt = "Hello, how are you?"
    test_kv = (torch.randn(1, 1, 10, 64), torch.randn(1, 1, 10, 64))

    adapter.cache_prefix(test_prompt, test_kv)
    cached = adapter.get_prefix_hash(test_prompt)
    assert cached is not None
    print("✅ 前缀缓存成功")

    adapter.clear_prefix_cache()
    cached_after_clear = adapter.get_prefix_hash(test_prompt)
    assert cached_after_clear is None
    print("✅ 前缀缓存清除成功")


def test_deepseek_adapter_api_mode():
    """测试 DeepSeek 适配器的 API 模式"""
    print("\n=== 测试 DeepSeek 适配器 API 模式 ===")

    # 测试 API 模式初始化
    adapter = DeepSeekAdapter()

    # 检查 API 客户端是否创建
    if adapter.api_client:
        print("✅ API 客户端创建成功")
    else:
        print("⚠️ API 客户端未创建（API key 未设置）")

    # 测试 API 可用性检查
    is_available = adapter.is_available()
    print(f"API 可用性: {is_available}")


def test_deepseek_adapter_capabilities():
    """测试 DeepSeek 适配器的能力报告"""
    print("\n=== 测试 DeepSeek 适配器能力报告 ===")

    adapter = DeepSeekAdapter()
    capabilities = adapter.get_capabilities()

    assert "type" in capabilities
    assert capabilities["type"] == "deepseek"
    assert "model_name" in capabilities
    assert "supports_api" in capabilities
    assert "supports_prefix_caching" in capabilities
    assert "max_tokens" in capabilities
    assert "temperature" in capabilities

    print(f"✅ 能力报告: {capabilities}")


def test_deepseek_adapter_with_custom_config():
    """测试带自定义配置的 DeepSeek 适配器"""
    print("\n=== 测试自定义配置的 DeepSeek 适配器 ===")

    custom_config = DeepSeekConfig(
        model_name="deepseek-ai/DeepSeek-V3.1",
        api_key_env="CUSTOM_DEEPSEEK_API_KEY",
        max_tokens=4096,
        temperature=0.5,
        use_api=False
    )

    adapter = DeepSeekAdapter(deepseek_config=custom_config)
    assert adapter.deepseek_config.model_name == "deepseek-ai/DeepSeek-V3.1"
    assert adapter.deepseek_config.api_key_env == "CUSTOM_DEEPSEEK_API_KEY"
    assert adapter.deepseek_config.max_tokens == 4096
    assert adapter.deepseek_config.temperature == 0.5
    assert adapter.deepseek_config.use_api is False

    print("✅ 自定义配置初始化成功")


def main():
    """主测试函数"""
    print("开始 DeepSeek 适配器集成测试...\n")

    try:
        test_deepseek_adapter_init()
        test_deepseek_adapter_attention()
        test_deepseek_adapter_prefix_caching()
        test_deepseek_adapter_api_mode()
        test_deepseek_adapter_capabilities()
        test_deepseek_adapter_with_custom_config()

        print("\n=== 测试完成 ===")
        print("✅ 所有 DeepSeek 适配器测试通过！")
        return True
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
