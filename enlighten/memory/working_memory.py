"""
Working Memory - 工作记忆矩阵
用于L2层，压缩上下文为固定大小的记忆矩阵 M (m × d)

T3新增功能:
- 滑动窗口刷新: 超出容量丢弃最旧的非敏感token
- 定期Top-K刷新: 基于注意力得分的定期刷新
- VAN敏感token保护: VAN标记的敏感token永不淘汰
"""

import torch
import torch.nn as nn
from typing import Tuple, List, Optional, Set
from dataclasses import dataclass


@dataclass
class RefreshResult:
    """刷新结果"""
    evicted_indices: List[int]
    refreshed_indices: List[int]
    protected_indices: Set[int]
    refresh_triggered: bool


class SlidingWindowRefresh:
    """
    滑动窗口刷新策略
    始终启用，超出容量丢弃最旧的非敏感token
    """

    def __init__(self, capacity: int):
        self.capacity = capacity

    def refresh(
        self,
        active_indices: List[int],
        protected_indices: Set[int]
    ) -> Tuple[List[int], List[int]]:
        """
        执行滑动窗口刷新

        Args:
            active_indices: 当前活跃索引
            protected_indices: 受保护的索引集合

        Returns:
            (new_active_indices, evicted_indices)
        """
        if len(active_indices) <= self.capacity:
            return active_indices, []

        evicted = []
        new_indices = []

        for idx in active_indices:
            if idx in protected_indices:
                new_indices.append(idx)
            elif len(new_indices) < self.capacity:
                new_indices.append(idx)
            else:
                evicted.append(idx)

        return new_indices, evicted


class TopkRefresh:
    """
    基于注意力得分的定期刷新

    Args:
        interval: 刷新间隔步数
        capacity: 记忆容量
    """

    def __init__(self, interval: int = 32, capacity: int = 512):
        self.interval = interval
        self.capacity = capacity
        self.step_counter = 0
        self.last_refresh_indices: Optional[List[int]] = None

    def should_refresh(self) -> bool:
        """检查是否应该刷新"""
        self.step_counter += 1
        return self.step_counter >= self.interval

    def reset_counter(self) -> None:
        """重置计数器"""
        self.step_counter = 0

    def refresh(
        self,
        memory: torch.Tensor,
        attention_scores: torch.Tensor,
        protected_indices: Set[int],
        current_indices: List[int]
    ) -> Tuple[List[int], List[int]]:
        """
        基于attention_scores执行Top-K刷新

        Args:
            memory: [memory_size, embed_dim] 记忆矩阵
            attention_scores: [seq_len] 注意力分数
            protected_indices: 受保护的索引
            current_indices: 当前活跃索引

        Returns:
            (new_indices, evicted_indices)
        """
        self.reset_counter()

        if attention_scores is None or len(attention_scores) == 0:
            return current_indices, []

        num_to_keep = min(self.capacity, len(attention_scores))
        _, topk_local_indices = torch.topk(attention_scores, k=num_to_keep)

        topk_indices = [current_indices[i] for i in topk_local_indices.tolist()]

        new_indices = []
        evicted = []

        for idx in topk_indices:
            if idx in protected_indices:
                new_indices.append(idx)
            else:
                if len(new_indices) < self.capacity:
                    new_indices.append(idx)
                else:
                    evicted.append(idx)

        remaining_protected = [idx for idx in protected_indices if idx not in new_indices]
        new_indices.extend(remaining_protected)

        self.last_refresh_indices = new_indices

        return new_indices[:self.capacity], evicted


class WorkingMemory(nn.Module):
    """
    工作记忆矩阵 M (m × d)

    功能:
    - 将n个token压缩为m个活跃token (m << n)
    - 维护活跃token索引集 A
    - 提供稀疏键值对 (K̃, Ṽ) 给L1
    - T3: 可配置刷新策略 (滑动窗口 + 定期TopK刷新)
    - T3: VAN敏感token保护

    参数:
        memory_size: m 活跃token数量
        embedding_dim: d 嵌入维度
        use_topk_refresh: 是否启用定期TopK刷新
        refresh_interval: TopK刷新间隔步数
    """

    def __init__(
        self,
        memory_size: int = 512,
        embedding_dim: int = 1024,
        update_strategy: str = "topk",
        use_topk_refresh: bool = True,
        refresh_interval: int = 32
    ):
        super().__init__()
        self.memory_size = memory_size
        self.embedding_dim = embedding_dim
        self.update_strategy = update_strategy
        self.use_topk_refresh = use_topk_refresh
        self.refresh_interval = refresh_interval

        self.M = nn.Parameter(torch.zeros(memory_size, embedding_dim))

        self.active_indices: List[List[int]] = []
        self.importance_scores_history = []
        self.sensitive_indices: List[Set[int]] = []

        self.sliding_window = SlidingWindowRefresh(memory_size)
        self.topk_refresh = TopkRefresh(refresh_interval, memory_size)

        self.register_buffer('initialized', torch.tensor(False))
        self.refresh_triggered = False

    def initialize(self, batch_size: int, device: torch.device):
        """
        初始化工作记忆
        """
        if not self.initialized:
            nn.init.xavier_uniform_(self.M)
            self.initialized = torch.tensor(True)

        self.active_indices = [[] for _ in range(batch_size)]
        self.sensitive_indices = [set() for _ in range(batch_size)]

    def mark_sensitive(self, batch_idx: int, indices: List[int]) -> None:
        """
        标记敏感token (VAN触发的token永不淘汰)

        Args:
            batch_idx: batch索引
            indices: 敏感token的索引列表
        """
        if batch_idx >= len(self.sensitive_indices):
            self.sensitive_indices.append(set())
        self.sensitive_indices[batch_idx].update(indices)

    def unmark_sensitive(self, batch_idx: int, indices: List[int]) -> None:
        """
        取消标记敏感token

        Args:
            batch_idx: batch索引
            indices: 要取消标记的索引列表
        """
        if batch_idx < len(self.sensitive_indices):
            self.sensitive_indices[batch_idx].difference_update(indices)

    def get_protected_indices(self, batch_idx: int) -> Set[int]:
        """
        获取指定batch的受保护索引

        Args:
            batch_idx: batch索引

        Returns:
            受保护的索引集合
        """
        if batch_idx < len(self.sensitive_indices):
            return self.sensitive_indices[batch_idx]
        return set()

    def update(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None
    ) -> RefreshResult:
        """
        更新工作记忆

        Args:
            key: [batch, seq_len, embed_dim] 键向量
            value: [batch, seq_len, embed_dim] 值向量
            attention_weights: [batch, seq_len] 注意力权重，用于计算重要性

        Returns:
            RefreshResult: 刷新结果
        """
        batch_size, seq_len, _ = key.shape

        if len(self.active_indices) != batch_size:
            self.initialize(batch_size, key.device)

        if attention_weights is None:
            importance = torch.norm(key, dim=-1)
        else:
            importance = attention_weights

        all_evicted = []
        all_refreshed = []
        self.refresh_triggered = False

        for b in range(batch_size):
            imp = importance[b]

            if self.update_strategy == "topk":
                topk_indices = torch.topk(imp, k=min(self.memory_size, seq_len)).indices
            elif self.update_strategy == "threshold":
                threshold = imp.mean()
                topk_indices = (imp > threshold).nonzero(as_tuple=True)[0]
                if len(topk_indices) > self.memory_size:
                    topk_indices = topk_indices[:self.memory_size]
            else:
                topk_indices = torch.arange(min(self.memory_size, seq_len), device=key.device)

            self._update_single(b, key[b], value[b], topk_indices)

            protected = self.get_protected_indices(b)

            if self.use_topk_refresh and self.topk_refresh.should_refresh():
                self.refresh_triggered = True
                new_indices, evicted = self.topk_refresh.refresh(
                    self.M.data,
                    imp,
                    protected,
                    self.active_indices[b]
                )
                self.active_indices[b] = new_indices
                all_evicted.extend(evicted)
                all_refreshed.extend(new_indices)
            else:
                new_indices, evicted = self.sliding_window.refresh(
                    self.active_indices[b],
                    protected
                )
                if new_indices != self.active_indices[b]:
                    self.refresh_triggered = True
                    self.active_indices[b] = new_indices
                    all_evicted.extend(evicted)

        return RefreshResult(
            evicted_indices=all_evicted,
            refreshed_indices=all_refreshed,
            protected_indices=set().union(*[s for s in self.sensitive_indices if s]),
            refresh_triggered=self.refresh_triggered
        )

    def _update_single(
        self,
        batch_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
        indices: torch.Tensor
    ) -> None:
        """
        更新单个样本的记忆
        """
        for i, idx in enumerate(indices[:self.memory_size]):
            self.M.data[i] = value[idx].detach()
            if len(self.active_indices[batch_idx]) < self.memory_size:
                self.active_indices[batch_idx].append(idx.item())
            else:
                self.active_indices[batch_idx][i] = idx.item()

    def evict_oldest(self, batch_idx: int = 0) -> Optional[int]:
        """
        淘汰最旧的非保护token

        Args:
            batch_idx: batch索引

        Returns:
            被淘汰的索引，如果无可淘汰的token则返回None
        """
        protected = self.get_protected_indices(batch_idx)
        for idx in self.active_indices[batch_idx]:
            if idx not in protected:
                self.active_indices[batch_idx].remove(idx)
                return idx
        return None

    def reset(self) -> None:
        """
        重置工作记忆状态
        """
        self.active_indices = []
        self.sensitive_indices = []
        self.importance_scores_history = []
        self.refresh_triggered = False
        self.initialized = torch.tensor(False)

    def get_sparse_kv(self) -> Tuple[torch.Tensor, torch.Tensor, List[List[int]]]:
        """
        获取稀疏键值对

        Returns:
            K_tilde: [memory_size, embed_dim] 稀疏键矩阵
            V_tilde: [memory_size, embed_dim] 稀疏值矩阵
            active_indices: 每batch的活跃索引列表
        """
        return self.M, self.M, self.active_indices

    def get_memory_snapshot(self) -> dict:
        """
        获取记忆快照，用于审计
        """
        return {
            'memory': self.M.data.clone(),
            'active_indices': [list(idx) for idx in self.active_indices],
            'sensitive_indices': [list(idx) for idx in self.sensitive_indices],
            'memory_size': self.memory_size,
            'embedding_dim': self.embedding_dim
        }

    def load_snapshot(self, snapshot: dict) -> None:
        """
        加载记忆快照
        """
        self.M.data = snapshot['memory']
        self.active_indices = [list(idx) for idx in snapshot['active_indices']]
        if 'sensitive_indices' in snapshot:
            self.sensitive_indices = [set(idx) for idx in snapshot['sensitive_indices']]


class MemoryBuffer(nn.Module):
    """
    记忆缓冲区
    用于临时存储历史记忆状态
    """

    def __init__(self, max_history: int = 10):
        super().__init__()
        self.max_history = max_history
        self.history = []

    def push(self, memory_state: dict) -> None:
        """
        推入记忆状态
        """
        self.history.append(memory_state)
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def get_last(self) -> Optional[dict]:
        """
        获取最近的记忆状态
        """
        return self.history[-1] if self.history else None

    def clear(self) -> None:
        """
        清空历史
        """
        self.history = []


class HierarchicalMemory(nn.Module):
    """
    分层记忆
    多层记忆矩阵，用于不同粒度的记忆
    """

    def __init__(
        self,
        memory_sizes: List[int] = [64, 256, 512],
        embedding_dim: int = 1024
    ):
        super().__init__()
        self.memory_sizes = memory_sizes
        self.embedding_dim = embedding_dim

        self.layers = nn.ModuleList([
            WorkingMemory(size, embedding_dim)
            for size in memory_sizes
        ])

    def update_all(self, key: torch.Tensor, value: torch.Tensor) -> None:
        """
        更新所有层的记忆
        """
        for layer in self.layers:
            layer.update(key, value)

    def query(self, level: int = 0) -> Tuple[torch.Tensor, torch.Tensor, List]:
        """
        查询指定层的记忆
        """
        return self.layers[level].get_sparse_kv()
