"""
L2 Working Memory Layer - L2工作记忆层
上下文压缩 + 熵统计 + 稀疏注意力
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass

from .memory.working_memory import WorkingMemory, HierarchicalMemory
from .memory.entropy_tracker import EntropyTracker, EntropyStatistics
from .memory.active_indices import ActiveIndices
from .attention.sparse import SparseAttention


@dataclass
class L2Output:
    """L2层输出"""
    sparse_kv: Tuple[torch.Tensor, torch.Tensor]
    active_indices: list
    entropy_stats: Dict[str, float]
    memory_snapshot: Dict[str, Any]


class L2WorkingMemory(nn.Module):
    """
    L2 工作记忆层

    功能:
    - 上下文压缩: n个token → m个活跃token
    - 熵统计计算: 追踪注意力熵的滑动统计
    - 活跃索引管理: 维护活跃token索引集A
    - 稀疏键值提供: (K̃, Ṽ) 给L1

    数据结构:
    - 记忆矩阵 M ∈ ℝ^(m×d), m=512, d=1024
    - 活跃索引集 A, |A| = m
    """

    def __init__(
        self,
        memory_size: int = 512,
        embedding_dim: int = 1024,
        config: Optional[Dict] = None
    ):
        super().__init__()
        self.memory_size = memory_size
        self.embedding_dim = embedding_dim

        config = config or {}

        self.working_memory = WorkingMemory(
            memory_size=memory_size,
            embedding_dim=embedding_dim,
            update_strategy=config.get("update_strategy", "topk")
        )

        self.hierarchical_memory = HierarchicalMemory(
            memory_sizes=config.get("hierarchical_sizes", [64, 256, 512]),
            embedding_dim=embedding_dim
        )

        self.entropy_tracker = EntropyTracker(
            window_size=config.get("entropy_window", 100),
            compute_interval=config.get("entropy_compute_interval", 1),
            ema_decay=config.get("ema_decay", 0.99)
        )

        self.active_indices_manager = ActiveIndices(
            max_size=memory_size,
            eviction_policy=config.get("eviction_policy", "lru")
        )

        self.sparse_attention = SparseAttention(
            embed_dim=embedding_dim,
            memory_size=memory_size,
            mode=config.get("sparse_mode", "topk")
        )

        self.use_hierarchical = config.get("use_hierarchical", False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None,
        update_memory: bool = True
    ) -> L2Output:
        """
        L2层前向传播

        Args:
            hidden_states: [batch, seq_len, embed_dim] 来自L1的hidden states
            attention_weights: [batch, seq_len, seq_len] 注意力权重
            update_memory: 是否更新记忆

        Returns:
            L2Output: 包含稀疏KV、熵统计和记忆快照
        """
        batch_size, seq_len, _ = hidden_states.shape

        key = hidden_states
        value = hidden_states

        if update_memory:
            self.working_memory.update(key, value, attention_weights)
            if self.use_hierarchical:
                self.hierarchical_memory.update_all(key, value)

        sparse_k, sparse_v, active_indices = self.working_memory.get_sparse_kv()

        if attention_weights is not None:
            self.entropy_tracker.update(attention_weights)

        entropy_stats = self.entropy_tracker.get_statistics()

        memory_snapshot = self.working_memory.get_memory_snapshot()

        return L2Output(
            sparse_kv=(sparse_k, sparse_v),
            active_indices=active_indices,
            entropy_stats=entropy_stats,
            memory_snapshot=memory_snapshot
        )

    def should_cutoff(self) -> bool:
        """
        判断是否应该截断

        使用熵统计和活跃索引判断
        """
        entropy_stats = self.entropy_tracker.get_statistics()

        entropy_threshold = 0.5
        variance_threshold = 0.05

        if entropy_stats["mean"] < entropy_threshold:
            if entropy_stats["variance"] < variance_threshold:
                if entropy_stats["trend"] < 0:
                    return True

        return False

    def reset(self) -> None:
        """
        重置工作记忆
        """
        self.entropy_tracker.reset()
        self.active_indices_manager.clear()

    def get_entropy_stats(self) -> EntropyStatistics:
        """
        获取当前熵统计
        """
        stats = self.entropy_tracker.get_statistics()
        return EntropyStatistics.from_dict(stats)

    def load_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """
        加载记忆快照
        """
        self.working_memory.load_snapshot(snapshot)


class SimplifiedL2(nn.Module):
    """
    简化版L2工作记忆层 - 用于快速原型验证
    """

    def __init__(self, memory_size: int = 512, embedding_dim: int = 512):
        super().__init__()
        self.memory_size = memory_size
        self.embedding_dim = embedding_dim

        self.memory = nn.Parameter(torch.zeros(memory_size, embedding_dim))

        self.entropy_tracker = EntropyTracker(window_size=50)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        简化版前向传播

        Returns:
            sparse_kv: [memory_size, embed_dim]
            entropy_stats: 熵统计字典
        """
        if attention_weights is not None:
            self.entropy_tracker.update(attention_weights)

        # 计算每个 token 的重要性，对 batch 维度取平均 (seq_len,)
        importance = torch.norm(hidden_states, dim=-1).mean(dim=0)

        # 确保 k 不超过重要性张量的长度
        k = min(self.memory_size, importance.size(0))
        if k > 0:
            topk_indices = torch.topk(importance, k=k).indices

            for i, idx in enumerate(topk_indices):
                # 对 batch 维度取平均
                self.memory.data[i] = hidden_states[:, idx, :].mean(dim=0).detach()

        entropy_stats = self.entropy_tracker.get_statistics()

        return self.memory, entropy_stats
