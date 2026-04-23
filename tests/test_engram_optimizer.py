"""
Engram 优化器集成测试
测试 Engram 记忆优化器的功能
"""

import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enlighten.memory.engram_optimizer import EngramOptimizer, EngramConfig


def test_engram_optimizer_init():
    """测试 Engram 优化器初始化"""
    print("=== 测试 Engram 优化器初始化 ===")

    # 测试默认配置
    config = EngramConfig()
    optimizer = EngramOptimizer(config)
    assert optimizer is not None
    assert optimizer.config.memory_size == 512
    assert optimizer.config.embedding_dim == 1024
    print("✅ 默认配置初始化成功")

    # 测试自定义配置
    custom_config = EngramConfig(
        memory_size=256,
        embedding_dim=512,
        consolidation_threshold=0.6,
        decay_rate=0.9,
        activation_threshold=0.4,
        use_fast_snapshots=True,
        max_fast_snapshots=3,
        enable_compression=True,
        compression_ratio=0.5
    )
    custom_optimizer = EngramOptimizer(custom_config)
    assert custom_optimizer.config.memory_size == 256
    assert custom_optimizer.config.embedding_dim == 512
    print("✅ 自定义配置初始化成功")


def test_engram_optimizer_update():
    """测试 Engram 优化器的记忆更新"""
    print("\n=== 测试 Engram 优化器记忆更新 ===")

    config = EngramConfig(memory_size=128, embedding_dim=64)
    optimizer = EngramOptimizer(config)

    # 生成测试数据
    batch_size = 1
    seq_len = 64
    embed_dim = 64

    hidden_states = torch.randn(batch_size, seq_len, embed_dim)
    importance_scores = torch.randn(batch_size, seq_len)

    # 测试第一次更新
    stats = optimizer.update(hidden_states, importance_scores)
    assert "total_updates" in stats
    assert stats["total_updates"] == 1
    assert "consolidated_count" in stats
    print("✅ 第一次记忆更新成功")

    # 测试多次更新
    for i in range(5):
        hidden_states = torch.randn(batch_size, seq_len, embed_dim)
        stats = optimizer.update(hidden_states)
    assert stats["total_updates"] == 6
    print("✅ 多次记忆更新成功")

    # 检查统计信息
    assert stats["step_count"] == 6
    assert "active_count" in stats
    assert "memory_utilization" in stats
    print(f"✅ 统计信息: {stats}")


def test_engram_optimizer_consolidation():
    """测试 Engram 优化器的记忆巩固"""
    print("\n=== 测试 Engram 优化器记忆巩固 ===")

    config = EngramConfig(
        memory_size=64,
        embedding_dim=32,
        consolidation_threshold=0.3
    )
    optimizer = EngramOptimizer(config)
    optimizer.consolidation_interval = 2  # 设置巩固间隔

    batch_size = 1
    seq_len = 32
    embed_dim = 32

    # 生成高激活的输入
    hidden_states = torch.randn(batch_size, seq_len, embed_dim)
    importance_scores = torch.randn(batch_size, seq_len) + 1.0  # 提高重要性

    # 触发巩固
    for i in range(10):
        stats = optimizer.update(hidden_states, importance_scores)

    assert stats["consolidations"] > 0
    assert stats["consolidated_count"] > 0
    print(f"✅ 记忆巩固成功: 巩固了 {stats['consolidated_count']} 个记忆")


def test_engram_optimizer_sparse_representation():
    """测试 Engram 优化器的稀疏表示"""
    print("\n=== 测试 Engram 优化器稀疏表示 ===")

    config = EngramConfig(memory_size=64, embedding_dim=32)
    optimizer = EngramOptimizer(config)

    batch_size = 1
    seq_len = 32
    embed_dim = 32

    # 生成测试数据
    hidden_states = torch.randn(batch_size, seq_len, embed_dim)
    importance_scores = torch.randn(batch_size, seq_len)

    # 更新记忆
    for i in range(5):
        optimizer.update(hidden_states, importance_scores)

    # 获取稀疏表示
    keys, values, indices = optimizer.get_optimized_kv()
    assert keys.shape[1] == 32  # embedding_dim
    assert values.shape[1] == 32
    assert len(indices) > 0
    print(f"✅ 稀疏表示成功: 保留 {len(indices)} 个索引")


def test_engram_optimizer_overhead_estimation():
    """测试 Engram 优化器的开销估算"""
    print("\n=== 测试 Engram 优化器开销估算 ===")

    config = EngramConfig()
    optimizer = EngramOptimizer(config)

    batch_size = 1
    seq_len = 64
    embed_dim = 1024

    # 生成测试数据
    hidden_states = torch.randn(batch_size, seq_len, embed_dim)

    # 初始开销
    initial_overhead = optimizer.estimate_overhead()
    print(f"初始开销: {initial_overhead:.2f}%")

    # 更新后开销
    for i in range(10):
        optimizer.update(hidden_states)

    updated_overhead = optimizer.estimate_overhead()
    print(f"更新后开销: {updated_overhead:.2f}%")

    # 开销应该在合理范围内
    assert 0 < updated_overhead < 10
    print("✅ 开销估算成功")


def test_engram_optimizer_snapshot():
    """测试 Engram 优化器的快照功能"""
    print("\n=== 测试 Engram 优化器快照功能 ===")

    config = EngramConfig(memory_size=64, embedding_dim=32, use_fast_snapshots=True)
    optimizer = EngramOptimizer(config)

    batch_size = 1
    seq_len = 32
    embed_dim = 32

    # 生成测试数据
    hidden_states = torch.randn(batch_size, seq_len, embed_dim)

    # 更新并创建快照
    for i in range(5):
        optimizer.update(hidden_states)

    # 手动创建快照
    snapshot = optimizer.engram_memory.snapshot()
    assert "memory" in snapshot
    assert "consolidated_indices" in snapshot
    assert "active_indices" in snapshot
    assert "cell_activations" in snapshot
    print("✅ 快照创建成功")

    # 验证快照内容
    assert snapshot["memory"].shape == (64, 32)
    assert isinstance(snapshot["consolidated_indices"], list)
    assert isinstance(snapshot["active_indices"], list)
    assert len(snapshot["cell_activations"]) == 64
    print("✅ 快照内容验证成功")


def test_engram_optimizer_prune():
    """测试 Engram 优化器的记忆修剪"""
    print("\n=== 测试 Engram 优化器记忆修剪 ===")

    config = EngramConfig(memory_size=64, embedding_dim=32)
    optimizer = EngramOptimizer(config)

    batch_size = 1
    seq_len = 32
    embed_dim = 32

    # 生成测试数据
    hidden_states = torch.randn(batch_size, seq_len, embed_dim)

    # 更新多次
    for i in range(20):
        optimizer.update(hidden_states)

    # 执行修剪
    current_time = float(optimizer.step_count)
    pruned = optimizer.engram_memory.prune(current_time)
    print(f"✅ 记忆修剪成功: 修剪了 {pruned} 个记忆")


def test_engram_optimizer_compression():
    """测试 Engram 优化器的记忆压缩"""
    print("\n=== 测试 Engram 优化器记忆压缩 ===")

    config = EngramConfig(
        memory_size=64,
        embedding_dim=32,
        enable_compression=True,
        compression_ratio=0.5
    )
    optimizer = EngramOptimizer(config)

    batch_size = 1
    seq_len = 32
    embed_dim = 32

    # 生成测试数据
    hidden_states = torch.randn(batch_size, seq_len, embed_dim)

    # 更新多次
    for i in range(10):
        optimizer.update(hidden_states)

    # 执行压缩
    compressed_memory, keep_indices = optimizer.engram_memory.compress()
    expected_size = int(64 * 0.5)
    assert compressed_memory.shape[0] <= expected_size
    assert len(keep_indices) <= expected_size
    print(f"✅ 记忆压缩成功: 保留 {len(keep_indices)} 个索引")


def main():
    """主测试函数"""
    print("开始 Engram 优化器集成测试...\n")

    try:
        test_engram_optimizer_init()
        test_engram_optimizer_update()
        test_engram_optimizer_consolidation()
        test_engram_optimizer_sparse_representation()
        test_engram_optimizer_overhead_estimation()
        test_engram_optimizer_snapshot()
        test_engram_optimizer_prune()
        test_engram_optimizer_compression()

        print("\n=== 测试完成 ===")
        print("✅ 所有 Engram 优化器测试通过！")
        return True
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
