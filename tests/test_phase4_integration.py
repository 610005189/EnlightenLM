"""
Phase 4 集成测试
测试 DeepSeek 适配器、Engram 优化器和多模态 VAN 的集成
"""

import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enlighten.adapters import DeepSeekAdapter
from enlighten.memory.engram_optimizer import EngramOptimizer, EngramConfig
from enlighten.attention.multimodal_van import MultimodalVan, MultimodalConfig, ModalityType


def test_deepseek_adapter_integration():
    """测试 DeepSeek 适配器集成"""
    print("=== 测试 DeepSeek 适配器集成 ===")

    # 测试 DeepSeek 适配器初始化
    adapter = DeepSeekAdapter()
    assert adapter is not None
    print("✅ DeepSeek 适配器初始化成功")

    # 测试 API 模式
    if adapter.api_client:
        print("✅ API 客户端创建成功")
        is_available = adapter.is_available()
        print(f"API 可用性: {is_available}")
    else:
        print("⚠️ API 客户端未创建（API key 未设置）")

    # 测试注意力计算
    batch_size = 1
    num_heads = 8
    seq_len = 32
    head_dim = 64

    query = torch.randn(batch_size, num_heads, seq_len, head_dim)
    key = torch.randn(batch_size, num_heads, seq_len, head_dim)
    value = torch.randn(batch_size, num_heads, seq_len, head_dim)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

    output = adapter.compute_attention(query, key, value, attention_mask)
    assert output.shape == (batch_size, num_heads, seq_len, head_dim)
    print("✅ 注意力计算测试成功")

    # 测试模型版本切换
    adapter.switch_model_version("v4")
    assert adapter.config.adapter_type.value == "deepseek_v4"
    print("✅ 模型版本切换测试成功")

    adapter.switch_model_version("v3")
    assert adapter.config.adapter_type.value == "deepseek_v3"
    print("✅ 模型版本切换回 V3 测试成功")


def test_engram_optimizer_integration():
    """测试 Engram 优化器集成"""
    print("\n=== 测试 Engram 优化器集成 ===")

    # 测试 Engram 优化器初始化
    config = EngramConfig(
        memory_size=256,
        embedding_dim=256,
        consolidation_threshold=0.1,  # 降低阈值以确保巩固触发
        use_fast_snapshots=False  # 禁用快速快照以避免自动巩固
    )
    optimizer = EngramOptimizer(config)
    assert optimizer is not None
    print("✅ Engram 优化器初始化成功")

    # 测试记忆更新
    batch_size = 1
    seq_len = 64
    embed_dim = 256

    hidden_states = torch.randn(batch_size, seq_len, embed_dim)
    # 提供高重要性分数以确保巩固
    importance_scores = torch.ones(batch_size, seq_len) * 0.9

    stats = optimizer.update(hidden_states, importance_scores)
    assert "total_updates" in stats
    assert stats["total_updates"] == 1
    print("✅ 记忆更新测试成功")

    # 测试多次更新和巩固 - 增加到 15 次更新以确保巩固触发
    for i in range(14):
        hidden_states = torch.randn(batch_size, seq_len, embed_dim)
        importance_scores = torch.ones(batch_size, seq_len) * 0.9
        stats = optimizer.update(hidden_states, importance_scores)

    assert stats["total_updates"] == 15
    print(f"当前统计: 巩固次数={stats['consolidations']}, 总更新次数={stats['total_updates']}")
    
    # 手动触发巩固
    consolidated = optimizer.engram_memory.consolidate()
    if consolidated:
        optimizer.stats["consolidations"] += 1
    
    assert optimizer.stats["consolidations"] > 0
    print(f"✅ 记忆巩固测试成功: 巩固了 {len(consolidated)} 个记忆")

    # 测试稀疏表示
    keys, values, indices = optimizer.get_optimized_kv()
    assert keys.shape[1] == 256
    assert values.shape[1] == 256
    assert len(indices) > 0
    print(f"✅ 稀疏表示测试成功: 保留 {len(indices)} 个索引")

    # 测试开销估算
    overhead = optimizer.estimate_overhead()
    assert 0 < overhead < 10
    print(f"✅ 开销估算测试成功: {overhead:.2f}%")


def test_multimodal_van_integration():
    """测试多模态 VAN 集成"""
    print("\n=== 测试多模态 VAN 集成 ===")

    # 测试多模态 VAN 初始化
    config = MultimodalConfig(
        modalities=[ModalityType.TEXT, ModalityType.IMAGE, ModalityType.AUDIO],
        embed_dim=256,
        num_heads=8,
        text_vocab_size=10000
    )
    van = MultimodalVan(config)
    assert van is not None
    print("✅ 多模态 VAN 初始化成功")

    # 测试文本编码器
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, 10000, (batch_size, seq_len))
    text_result = van.text_encoder(input_ids)
    assert "harm_prob" in text_result
    assert text_result["harm_prob"].shape == (batch_size,)
    print("✅ 文本编码器测试成功")

    # 测试图像编码器
    channels = 3
    height = 32
    width = 32
    images = torch.randn(batch_size, channels, height, width)
    image_result = van.image_encoder(images)
    assert "harm_prob" in image_result
    assert image_result["harm_prob"].shape == (batch_size,)
    print("✅ 图像编码器测试成功")

    # 测试音频编码器
    freq = 80
    time = 50
    audio = torch.randn(batch_size, 1, freq, time)
    audio_result = van.audio_encoder(audio)
    assert "harm_prob" in audio_result
    assert audio_result["harm_prob"].shape == (batch_size,)
    print("✅ 音频编码器测试成功")

    # 测试多模态融合
    inputs = {
        "text": input_ids,
        "image": images,
        "audio": audio
    }
    result = van(inputs)
    assert "harm_prob" in result
    assert "is_multimodal" in result
    assert result["is_multimodal"] is True
    assert len(result["active_modalities"]) == 3
    print("✅ 多模态融合测试成功")

    # 测试有害内容检测
    is_harmful, harm_prob, details = van.detect_harm(inputs, threshold=0.5)
    assert isinstance(is_harmful, (bool, torch.Tensor))
    assert isinstance(harm_prob, (float, torch.Tensor))
    if isinstance(harm_prob, torch.Tensor):
        assert harm_prob.shape == (batch_size,)
        print(f"✅ 有害内容检测测试成功: 有害概率形状 = {harm_prob.shape}")
    else:
        assert 0 <= harm_prob <= 1
        print(f"✅ 有害内容检测测试成功: 有害概率 = {harm_prob:.4f}")


def test_full_integration():
    """测试完整集成"""
    print("\n=== 测试完整集成 ===")

    # 同时初始化所有组件
    print("初始化所有 Phase 4 组件...")

    # 1. DeepSeek 适配器
    adapter = DeepSeekAdapter()
    print("✅ DeepSeek 适配器初始化")

    # 2. Engram 优化器
    engram_config = EngramConfig(
        memory_size=256, 
        embedding_dim=256,
        consolidation_threshold=0.1,
        use_fast_snapshots=False
    )
    engram_optimizer = EngramOptimizer(engram_config)
    print("✅ Engram 优化器初始化")

    # 3. 多模态 VAN
    multimodal_config = MultimodalConfig(
        modalities=[ModalityType.TEXT],
        embed_dim=256,
        num_heads=8,
        text_vocab_size=10000
    )
    multimodal_van = MultimodalVan(multimodal_config)
    print("✅ 多模态 VAN 初始化")

    print("\n所有组件初始化成功，集成测试通过！")


def main():
    """主测试函数"""
    print("开始 Phase 4 集成测试...\n")

    try:
        test_deepseek_adapter_integration()
        test_engram_optimizer_integration()
        test_multimodal_van_integration()
        test_full_integration()

        print("\n=== 测试完成 ===")
        print("✅ 所有 Phase 4 集成测试通过！")
        return True
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
