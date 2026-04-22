"""
Active Indices - 活跃token索引管理
管理活跃token索引集A，实现LRU/FIFO等淘汰策略
"""

import torch
from collections import OrderedDict
from typing import List, Set, Optional


class ActiveIndices:
    """
    活跃token索引管理器

    功能:
    - 维护活跃token索引集 A
    - 支持多种淘汰策略 (LRU, FIFO, Random)
    - 提供索引查询和更新
    """

    def __init__(
        self,
        max_size: int = 512,
        eviction_policy: str = "lru"
    ):
        self.max_size = max_size
        self.eviction_policy = eviction_policy

        self.indices_per_batch = []

        if eviction_policy == "lru":
            self.lru_cache = OrderedDict()
        elif eviction_policy == "fifo":
            self.fifo_queue = []
        elif eviction_policy == "random":
            pass

    def initialize(self, batch_size: int) -> None:
        """
        初始化活跃索引

        Args:
            batch_size: batch大小
        """
        self.indices_per_batch = [[] for _ in range(batch_size)]

        if self.eviction_policy == "lru":
            self.lru_cache = OrderedDict()
        elif self.eviction_policy == "fifo":
            self.fifo_queue = []

    def add(self, batch_idx: int, indices: List[int]) -> None:
        """
        添加活跃索引

        Args:
            batch_idx: batch索引
            indices: 要添加的索引列表
        """
        if batch_idx >= len(self.indices_per_batch):
            self.indices_per_batch.append([])

        current = self.indices_per_batch[batch_idx]

        for idx in indices:
            if idx not in current:
                if len(current) >= self.max_size:
                    self._evict(batch_idx)

                current.append(idx)
                self._update_cache(idx)

    def _evict(self, batch_idx: int) -> Optional[int]:
        """
        淘汰一个索引

        Returns:
            evicted_idx: 被淘汰的索引
        """
        if self.eviction_policy == "lru":
            if self.lru_cache:
                evicted_idx, _ = self.lru_cache.popitem(last=False)
                if evicted_idx in self.indices_per_batch[batch_idx]:
                    self.indices_per_batch[batch_idx].remove(evicted_idx)
                return evicted_idx

        elif self.eviction_policy == "fifo":
            if self.fifo_queue:
                evicted_idx = self.fifo_queue.pop(0)
                if evicted_idx in self.indices_per_batch[batch_idx]:
                    self.indices_per_batch[batch_idx].remove(evicted_idx)
                return evicted_idx

        elif self.eviction_policy == "random":
            if self.indices_per_batch[batch_idx]:
                evicted_idx = self.indices_per_batch[batch_idx].pop(0)
                return evicted_idx

        return None

    def _update_cache(self, idx: int) -> None:
        """
        更新缓存状态
        """
        if self.eviction_policy == "lru":
            if idx in self.lru_cache:
                self.lru_cache.move_to_end(idx)
            else:
                self.lru_cache[idx] = True

        elif self.eviction_policy == "fifo":
            if idx not in self.fifo_queue:
                self.fifo_queue.append(idx)

    def get(self, batch_idx: int) -> List[int]:
        """
        获取指定batch的活跃索引

        Args:
            batch_idx: batch索引

        Returns:
            indices: 活跃索引列表
        """
        if batch_idx < len(self.indices_per_batch):
            return self.indices_per_batch[batch_idx]
        return []

    def get_all(self) -> List[List[int]]:
        """
        获取所有batch的活跃索引
        """
        return self.indices_per_batch

    def remove(self, batch_idx: int, indices: List[int]) -> None:
        """
        移除指定索引

        Args:
            batch_idx: batch索引
            indices: 要移除的索引列表
        """
        if batch_idx < len(self.indices_per_batch):
            for idx in indices:
                if idx in self.indices_per_batch[batch_idx]:
                    self.indices_per_batch[batch_idx].remove(idx)

                if self.eviction_policy == "lru" and idx in self.lru_cache:
                    del self.lru_cache[idx]
                elif self.eviction_policy == "fifo" and idx in self.fifo_queue:
                    self.fifo_queue.remove(idx)

    def clear(self, batch_idx: Optional[int] = None) -> None:
        """
        清空活跃索引

        Args:
            batch_idx: 如果指定，只清空该batch；否则清空所有
        """
        if batch_idx is not None:
            if batch_idx < len(self.indices_per_batch):
                self.indices_per_batch[batch_idx] = []
        else:
            self.indices_per_batch = []

        if self.eviction_policy == "lru":
            self.lru_cache.clear()
        elif self.eviction_policy == "fifo":
            self.fifo_queue = []


class SparseIndexManager:
    """
    稀疏索引管理器
    用于高效管理大规模稀疏索引
    """

    def __init__(self, max_size: int = 512):
        self.max_size = max_size
        self.indices = set()

    def update(self, new_indices: Set[int]) -> None:
        """
        更新索引集合

        Args:
            new_indices: 新的索引集合
        """
        self.indices.update(new_indices)

        if len(self.indices) > self.max_size:
            excess = len(self.indices) - self.max_size
            self.indices = set(list(self.indices)[excess:])

    def get_indices(self) -> List[int]:
        """
        获取所有索引
        """
        return list(self.indices)

    def contains(self, idx: int) -> bool:
        """
        检查是否包含索引
        """
        return idx in self.indices
