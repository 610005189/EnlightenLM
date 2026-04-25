"""
L2骨架代码集成测试套件
验证L2WorkingMemoryAdapter和稀疏注意力的功能

测试分类:
- L2适配器初始化试验 (5组)
- 稀疏注意力选择试验 (5组)
- L1→L2→L3数据流试验 (5组)
- 熵统计和截断判断试验 (5组)
"""

import pytest
import torch
import numpy as np
from typing import Dict, Any, Tuple


class TestL2AdapterInitialization:
    """L2适配器初始化试验"""

    def test_l2_adapter_initialization(self):
        """L2-ADP-01: L2适配器初始化"""
        from enlighten.hybrid_architecture import L2WorkingMemoryAdapter

        adapter = L2WorkingMemoryAdapter(
            memory_size=256,
            embedding_dim=768
        )

        assert adapter.memory_size == 256
        assert adapter.embedding_dim == 768
        assert adapter.skeleton_l2 is not None

    def test_l2_adapter_default_disabled(self):
        """L2-ADP-02: L2适配器默认禁用"""
        from enlighten.hybrid_architecture import HybridEnlightenLM

        model = HybridEnlightenLM(use_skeleton_l2=False)

        assert model.use_skeleton_l2 == False
        assert model.l2_adapter is None

    def test_hybrid_with_l2_adapter_enabled(self):
        """L2-ADP-03: 带L2适配器的混合模型"""
        from enlighten.hybrid_architecture import HybridEnlightenLM

        model = HybridEnlightenLM(
            use_skeleton_l2=True,
            l2_config={"memory_size": 256, "embedding_dim": 768}
        )

        assert model.use_skeleton_l2 == True
        assert model.l2_adapter is not None

    def test_l2_adapter_with_custom_config(self):
        """L2-ADP-04: L2适配器自定义配置"""
        from enlighten.hybrid_architecture import HybridEnlightenLM

        l2_config = {
            "memory_size": 256,
            "embedding_dim": 768,
            "update_strategy": "topk",
            "entropy_window": 50,
            "sparse_mode": "topk"
        }

        model = HybridEnlightenLM(use_skeleton_l2=True, l2_config=l2_config)

        assert model.l2_adapter.memory_size == 256
        assert model.l2_adapter.embedding_dim == 768

    def test_l2_adapter_skeleton_components(self):
        """L2-ADP-05: L2适配器骨架组件"""
        from enlighten.hybrid_architecture import L2WorkingMemoryAdapter

        adapter = L2WorkingMemoryAdapter()

        assert adapter.skeleton_l2 is not None
        assert adapter.skeleton_l2.working_memory is not None
        assert adapter.skeleton_l2.entropy_tracker is not None
        assert adapter.sparse_attention is not None


class TestSparseAttentionSelection:
    """稀疏注意力选择试验"""

    def test_sparse_attention_output_shape(self):
        """L2-SPARSE-01: 稀疏注意力输出形状"""
        from enlighten.hybrid_architecture import SimplifiedL2Adapter

        adapter = SimplifiedL2Adapter(memory_size=64, embedding_dim=128)

        batch_size = 1
        seq_len = 16
        hidden_states = torch.randn(batch_size, seq_len, 128)

        attention_weights = torch.rand(batch_size, seq_len, seq_len)
        attention_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True)

        result = adapter.forward(hidden_states, attention_weights)

        assert result.sparse_kv is not None
        assert len(result.active_indices) >= 0
        assert isinstance(result.entropy_stats, dict)

    def test_select_sparse_indices(self):
        """L2-SPARSE-02: 选择稀疏索引"""
        from enlighten.hybrid_architecture import SimplifiedL2Adapter

        adapter = SimplifiedL2Adapter(memory_size=64, embedding_dim=128)

        hidden_states = torch.randn(1, 32, 128)
        attention_weights = torch.rand(1, 32, 32)
        attention_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True)

        adapter.forward(hidden_states, attention_weights)

        selected_k, selected_v = adapter.select_sparse_indices(attention_weights, topk=16)

        assert selected_k is not None
        assert selected_v is not None
        assert selected_k.shape[0] <= 16

    def test_get_sparse_attention_output(self):
        """L2-SPARSE-03: 获取稀疏注意力输出"""
        from enlighten.hybrid_architecture import SimplifiedL2Adapter

        adapter = SimplifiedL2Adapter(memory_size=64, embedding_dim=128)

        hidden_states = torch.randn(1, 32, 128)
        attention_weights = torch.rand(1, 32, 32)
        attention_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True)

        adapter.forward(hidden_states, attention_weights)

        query = torch.randn(1, 16, 128)
        output, attn_weights = adapter.get_sparse_attention_output(query)

        assert output.shape == (1, 16, 128)
        assert attn_weights is not None

    def test_sparse_attention_topk_mode(self):
        """L2-SPARSE-04: 稀疏注意力TopK模式"""
        from enlighten.hybrid_architecture import SimplifiedL2Adapter

        adapter = SimplifiedL2Adapter(memory_size=32, embedding_dim=128)

        hidden_states = torch.randn(1, 64, 128)
        attention_weights = torch.rand(1, 64, 64)
        attention_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True)

        result = adapter.forward(hidden_states, attention_weights)

        assert result.use_skeleton == True
        assert result.sparse_kv is not None

    def test_sparse_attention_empty_case(self):
        """L2-SPARSE-05: 空情况处理"""
        from enlighten.hybrid_architecture import SimplifiedL2Adapter

        adapter = SimplifiedL2Adapter(memory_size=8, embedding_dim=64)

        hidden_states = torch.randn(1, 4, 64)
        attention_weights = torch.rand(1, 4, 4)
        attention_weights = attention_weights / (attention_weights.sum(dim=-1, keepdim=True) + 1e-10)

        result = adapter.forward(hidden_states, attention_weights)

        assert result is not None
        assert isinstance(result.entropy_stats, dict)


class TestL1ToL2DataFlow:
    """L1→L2数据流试验"""

    def test_l1_l2_data_flow_via_simplified(self):
        """L2-FLOW-01: 通过简化适配器的L1到L2数据流"""
        from enlighten.hybrid_architecture import SimplifiedL2Adapter

        adapter = SimplifiedL2Adapter(memory_size=128, embedding_dim=256)

        seq_len = 8
        hidden_states = torch.randn(1, seq_len, 256)

        attention_weights = torch.rand(1, seq_len, seq_len)
        attention_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True)

        l2_result = adapter.forward(
            hidden_states=hidden_states,
            attention_weights=attention_weights,
            update_memory=True
        )

        assert l2_result is not None
        assert isinstance(l2_result.entropy_stats, dict)

    def test_hybrid_l1_l2_integration_status(self):
        """L2-FLOW-02: 混合模型L1/L2集成状态"""
        from enlighten.hybrid_architecture import HybridEnlightenLM

        model = HybridEnlightenLM(
            use_l1_adapter=True,
            use_skeleton_l2=True
        )

        status = model.get_status()

        assert status["use_l1_adapter"] == True
        assert status["use_skeleton_l2"] == True
        assert "l1_adapter" in status
        assert "l2_adapter" in status

    def test_simplified_l2_adapter_forward(self):
        """L2-FLOW-03: 简化L2适配器前向传播"""
        from enlighten.hybrid_architecture import SimplifiedL2Adapter

        adapter = SimplifiedL2Adapter(memory_size=128, embedding_dim=256)

        hidden_states = torch.randn(1, 16, 256)
        attention_weights = torch.rand(1, 16, 16)
        attention_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True)

        result = adapter.forward(hidden_states, attention_weights)

        assert result is not None
        assert result.sparse_kv is not None

    def test_sparse_attention_select_interface(self):
        """L2-FLOW-04: 稀疏注意力选择接口"""
        from enlighten.hybrid_architecture import SimplifiedL2Adapter

        adapter = SimplifiedL2Adapter(memory_size=64, embedding_dim=128)

        hidden_states = torch.randn(1, 32, 128)
        attention_weights = torch.rand(1, 32, 32)
        attention_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True)

        adapter.forward(hidden_states, attention_weights)

        selected_k, selected_v = adapter.select_sparse_indices(attention_weights, topk=16)

        assert selected_k is not None
        assert selected_v is not None

    def test_l1_l2_cutoff_integration(self):
        """L2-FLOW-05: L1/L2截断集成"""
        from enlighten.hybrid_architecture import HybridEnlightenLM

        model = HybridEnlightenLM(
            use_l1_adapter=True,
            use_skeleton_l2=True,
            l2_config={"memory_size": 64, "embedding_dim": 128}
        )

        should_cutoff = model.should_l2_cutoff()

        assert isinstance(should_cutoff, bool)


class TestEntropyAndCutoff:
    """熵统计和截断判断试验"""

    def test_simplified_l2_entropy_stats(self):
        """L2-ENTROPY-01: 简化L2熵统计"""
        from enlighten.hybrid_architecture import SimplifiedL2Adapter

        adapter = SimplifiedL2Adapter(memory_size=64, embedding_dim=128)

        hidden_states = torch.randn(1, 16, 128)
        attention_weights = torch.rand(1, 16, 16)
        attention_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True)

        adapter.forward(hidden_states, attention_weights)

        entropy_stats = adapter.get_entropy_stats()

        assert "mean" in entropy_stats
        assert "variance" in entropy_stats
        assert "trend" in entropy_stats
        assert "current" in entropy_stats

    def test_entropy_stats_update(self):
        """L2-ENTROPY-02: 熵统计更新"""
        from enlighten.hybrid_architecture import SimplifiedL2Adapter

        adapter = SimplifiedL2Adapter(memory_size=64, embedding_dim=128)

        for i in range(5):
            hidden_states = torch.randn(1, 16, 128)
            attention_weights = torch.rand(1, 16, 16)
            attention_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True)

            adapter.forward(hidden_states, attention_weights)

        entropy_stats = adapter.get_entropy_stats()

        assert entropy_stats["current"] >= 0

    def test_should_cutoff_logic(self):
        """L2-ENTROPY-03: 截断判断逻辑"""
        from enlighten.hybrid_architecture import SimplifiedL2Adapter

        adapter = SimplifiedL2Adapter()

        should_cutoff = adapter.should_cutoff()

        assert isinstance(should_cutoff, bool)

    def test_hybrid_get_l2_entropy_stats(self):
        """L2-ENTROPY-04: 混合模型获取L2熵统计"""
        from enlighten.hybrid_architecture import HybridEnlightenLM

        model = HybridEnlightenLM(use_skeleton_l2=True)

        entropy_stats = model.get_l2_entropy_stats()

        assert isinstance(entropy_stats, dict)
        assert "mean" in entropy_stats

    def test_hybrid_without_l2_returns_working_memory_stats(self):
        """L2-ENTROPY-05: 无L2适配器时返回WorkingMemory统计"""
        from enlighten.hybrid_architecture import HybridEnlightenLM

        model = HybridEnlightenLM(use_skeleton_l2=False)

        entropy_stats = model.get_l2_entropy_stats()

        assert isinstance(entropy_stats, dict)


class TestResetAndStatus:
    """重置和状态试验"""

    def test_simplified_l2_reset(self):
        """L2-RESET-01: 简化L2重置"""
        from enlighten.hybrid_architecture import SimplifiedL2Adapter

        adapter = SimplifiedL2Adapter(memory_size=64, embedding_dim=256)

        hidden_states = torch.randn(1, 16, 256)
        attention_weights = torch.rand(1, 16, 16)
        attention_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True)

        adapter.forward(hidden_states, attention_weights)

        adapter.reset()

        new_entropy_stats = adapter.get_entropy_stats()
        assert new_entropy_stats["mean"] == 0.0

    def test_hybrid_reset_with_l2_adapter(self):
        """L2-RESET-02: 带L2适配器的混合模型重置"""
        from enlighten.hybrid_architecture import HybridEnlightenLM

        model = HybridEnlightenLM(
            use_l1_adapter=True,
            use_skeleton_l2=True
        )

        model.reset()

        assert model.working_memory.token_count == 0

    def test_l2_adapter_status_in_hybrid(self):
        """L2-RESET-03: 混合模型中L2适配器状态"""
        from enlighten.hybrid_architecture import HybridEnlightenLM

        model = HybridEnlightenLM(
            use_skeleton_l2=True,
            l2_config={"memory_size": 256, "embedding_dim": 768}
        )

        status = model.get_status()

        assert "l2_adapter" in status
        assert status["l2_adapter"]["memory_size"] == 256
        assert status["l2_adapter"]["embedding_dim"] == 768

    def test_get_sparse_attention_output_interface(self):
        """L2-RESET-04: 获取稀疏注意力输出接口"""
        from enlighten.hybrid_architecture import HybridEnlightenLM

        model = HybridEnlightenLM(
            use_skeleton_l2=True,
            l2_config={"memory_size": 64, "embedding_dim": 256}
        )

        query = torch.randn(1, 8, 256)
        output, attn = model.get_sparse_attention_output(query)

        assert output.shape[0] == 1
        assert output.shape[2] == 256


class TestSimplifiedL2Adapter:
    """简化版L2适配器试验"""

    def test_simplified_l2_adapter(self):
        """L2-SIMPL-01: 简化版L2适配器"""
        from enlighten.hybrid_architecture import SimplifiedL2Adapter

        adapter = SimplifiedL2Adapter(memory_size=128, embedding_dim=256)

        hidden_states = torch.randn(1, 16, 256)
        attention_weights = torch.rand(1, 16, 16)
        attention_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True)

        result = adapter.forward(hidden_states, attention_weights)

        assert result.sparse_kv is not None
        assert isinstance(result.entropy_stats, dict)

    def test_simplified_l2_should_cutoff(self):
        """L2-SIMPL-02: 简化版L2截断判断"""
        from enlighten.hybrid_architecture import SimplifiedL2Adapter

        adapter = SimplifiedL2Adapter()

        should_cutoff = adapter.should_cutoff()

        assert isinstance(should_cutoff, bool)

    def test_simplified_l2_sparse_selection(self):
        """L2-SIMPL-03: 简化版L2稀疏选择"""
        from enlighten.hybrid_architecture import SimplifiedL2Adapter

        adapter = SimplifiedL2Adapter(memory_size=64, embedding_dim=128)

        hidden_states = torch.randn(1, 32, 128)
        attention_weights = torch.rand(1, 32, 32)
        attention_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True)

        adapter.forward(hidden_states, attention_weights)

        selected_k, selected_v = adapter.select_sparse_indices(attention_weights, topk=16)

        assert selected_k is not None
        assert selected_v is not None


class TestHybridEnlightenL2Integration:
    """HybridEnlightenLM L2集成测试"""

    def test_hybrid_l2_interface_methods(self):
        """L2-INT-01: HybridEnlightenLM L2接口方法"""
        from enlighten.hybrid_architecture import HybridEnlightenLM

        model = HybridEnlightenLM(use_skeleton_l2=True)

        assert hasattr(model, 'process_with_l2_adapter')
        assert hasattr(model, 'get_l2_entropy_stats')
        assert hasattr(model, 'sparse_attention_select')
        assert hasattr(model, 'get_sparse_attention_output')
        assert hasattr(model, 'should_l2_cutoff')

    def test_l2_result_dataclass(self):
        """L2-INT-02: L2Result数据类"""
        from enlighten.hybrid_architecture import L2Result

        result = L2Result(
            sparse_kv=(torch.randn(10, 128), torch.randn(10, 128)),
            active_indices=[1, 2, 3],
            entropy_stats={"mean": 0.5, "variance": 0.1, "trend": 0.0, "current": 0.5},
            memory_snapshot={},
            use_skeleton=True
        )

        assert result.sparse_kv is not None
        assert result.use_skeleton == True
        assert len(result.active_indices) == 3

    def test_sparse_attention_selection_returns_tuple(self):
        """L2-INT-03: 稀疏注意力选择返回元组"""
        from enlighten.hybrid_architecture import HybridEnlightenLM

        model = HybridEnlightenLM(use_skeleton_l2=True)

        attention_weights = torch.rand(1, 32, 32)
        attention_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True)

        result = model.sparse_attention_select(attention_weights, topk=8)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_l2_adapter_not_None_when_enabled(self):
        """L2-INT-04: L2适配器启用时不为None"""
        from enlighten.hybrid_architecture import HybridEnlightenLM

        model = HybridEnlightenLM(use_skeleton_l2=True)

        assert model.l2_adapter is not None

    def test_l2_adapter_None_when_disabled(self):
        """L2-INT-05: L2适配器禁用时为None"""
        from enlighten.hybrid_architecture import HybridEnlightenLM

        model = HybridEnlightenLM(use_skeleton_l2=False)

        assert model.l2_adapter is None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])