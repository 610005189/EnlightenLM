"""
Engram 记忆优化器
基于神经科学的记忆巩固和优化机制
实现 +5% 性能目标
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
import time


class MemoryCell:
    """记忆细胞"""

    def __init__(self, index: int, embedding_dim: int):
        self.index = index
        self.embedding = torch.zeros(embedding_dim)
        self.activation = 0.0
        self.last_access = 0.0
        self.consolidated = False

    def update(
        self,
        embedding: torch.Tensor,
        activation: float,
        timestamp: float,
        consolidate: bool = False
    ):
        """更新记忆细胞"""
        self.embedding = embedding
        self.activation = max(0.0, min(1.0, activation))
        self.last_access = timestamp

        if consolidate and self.activation > 0.7:
            self.consolidated = True

    def should_evict(self, current_time: float, threshold: float = 0.1) -> bool:
        """判断是否应该被驱逐"""
        if self.consolidated:
            return False

        time_since_access = current_time - self.last_access
        activation_decay = self.activation * (0.99 ** time_since_access)

        return activation_decay < threshold


@dataclass
class EngramConfig:
    """Engram 配置"""
    memory_size: int = 256
    embedding_dim: int = 256
    consolidation_threshold: float = 0.6
    activation_threshold: float = 0.3
    compression_ratio: float = 0.5
    enable_compression: bool = True
    use_fast_snapshots: bool = True


class EngramMemory:
    """
    Engram 记忆实现

    基于神经科学的记忆巩固和优化机制
    """

    def __init__(self, config: EngramConfig):
        self.config = config
        self.cells = [MemoryCell(i, config.embedding_dim) for i in range(config.memory_size)]
        self.memory = torch.zeros(config.memory_size, config.embedding_dim)
        
        self.consolidated_indices = []
        self.active_indices = []
        self.fast_snapshots = []

    def update(
        self,
        hidden_states: torch.Tensor,
        importance_scores: Optional[torch.Tensor] = None,
        timestamp: float = None
    ):
        """
        更新 Engram 记忆

        Args:
            hidden_states: [batch, seq_len, embed_dim]
            importance_scores: [batch, seq_len]
            timestamp: 时间戳
        """
        if timestamp is None:
            timestamp = time.time()

        batch_size, seq_len, embed_dim = hidden_states.shape

        if importance_scores is None:
            importance_scores = torch.ones(batch_size, seq_len)

        for b in range(batch_size):
            for s in range(seq_len):
                cell_idx = (b * seq_len + s) % self.config.memory_size
                score = importance_scores[b, s]
                embedding = hidden_states[b, s]

                self.cells[cell_idx].update(
                    embedding=embedding,
                    activation=score.item(),
                    timestamp=timestamp,
                    consolidate=self.config.use_fast_snapshots
                )

                self.memory[cell_idx] = embedding

        self.active_indices = [c.index for c in self.cells if c.activation > 0]

    def consolidate(self) -> List[int]:
        """
        执行记忆巩固

        将重要的短期记忆巩固为长期记忆
        """
        consolidated = []

        for cell in self.cells:
            if cell.activation > self.config.consolidation_threshold and not cell.consolidated:
                cell.consolidated = True
                consolidated.append(cell.index)

                if cell.index not in self.consolidated_indices:
                    self.consolidated_indices.append(cell.index)

        return consolidated

    def snapshot(self) -> Dict[str, Any]:
        """保存快速快照"""
        snapshot = {
            "memory": self.memory.clone(),
            "consolidated_indices": self.consolidated_indices.copy(),
            "active_indices": self.active_indices.copy(),
            "cell_activations": [c.activation for c in self.cells]
        }

        if self.config.use_fast_snapshots:
            self.fast_snapshots.append(snapshot)

        return snapshot

    def restore(self, snapshot: Dict[str, Any]) -> None:
        """恢复快照"""
        self.memory = snapshot["memory"].clone()
        self.consolidated_indices = snapshot["consolidated_indices"].copy()
        self.active_indices = snapshot["active_indices"].copy()

        for i, activation in enumerate(snapshot["cell_activations"]):
            if i < len(self.cells):
                self.cells[i].activation = activation

    def prune(self, current_time: float) -> int:
        """
        修剪弱记忆

        移除长时间未访问且激活低的记忆
        """
        pruned = 0

        for cell in self.cells:
            if cell.should_evict(current_time):
                cell.activation = 0.0
                pruned += 1

        return pruned

    def compress(self) -> Tuple[torch.Tensor, List[int]]:
        """
        压缩记忆矩阵

        返回压缩后的记忆和保留的索引
        """
        if not self.config.enable_compression:
            return self.memory, list(range(self.config.memory_size))

        activations = torch.tensor([c.activation for c in self.cells])

        k = int(self.config.memory_size * self.config.compression_ratio)
        keep_indices = torch.topk(activations, k=k).indices.tolist()

        compressed_memory = self.memory[keep_indices]

        return compressed_memory, keep_indices

    def get_sparse_representation(self) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """
        获取稀疏表示

        Returns:
            (keys, values, indices)
        """
        activations = torch.tensor([c.activation for c in self.cells])

        threshold = self.config.activation_threshold
        sparse_mask = activations > threshold

        sparse_indices = sparse_mask.nonzero().squeeze(-1).tolist()

        if not sparse_indices:
            return self.memory, self.memory, list(range(self.config.memory_size))

        sparse_memory = self.memory[sparse_indices]

        return sparse_memory, sparse_memory, sparse_indices


class EngramOptimizer:
    """
    Engram 优化器

    整合 Engram 记忆和优化策略，实现 +5% 性能目标
    """

    def __init__(self, config: EngramConfig):
        self.config = config
        self.engram_memory = EngramMemory(config)

        self.step_count = 0
        self.last_consolidation = 0
        self.consolidation_interval = 10

        self.stats = {
            "total_updates": 0,
            "consolidations": 0,
            "prunes": 0,
            "compressions": 0,
            "avg_activation": 0.0
        }

    def update(
        self,
        hidden_states: torch.Tensor,
        importance_scores: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        更新 Engram 记忆

        Args:
            hidden_states: 隐藏状态
            importance_scores: 重要性分数

        Returns:
            更新统计信息
        """
        self.step_count += 1
        self.stats["total_updates"] += 1

        self.engram_memory.update(
            hidden_states=hidden_states,
            importance_scores=importance_scores,
            timestamp=float(self.step_count)
        )

        if self.step_count - self.last_consolidation >= self.consolidation_interval:
            consolidated = self.engram_memory.consolidate()
            if consolidated:
                self.stats["consolidations"] += 1
                self.last_consolidation = self.step_count

        if self.step_count % 50 == 0:
            pruned = self.engram_memory.prune(float(self.step_count))
            self.stats["prunes"] += pruned

            compressed, _ = self.engram_memory.compress()
            self.stats["compressions"] += 1

        activations = [c.activation for c in self.engram_memory.cells]
        self.stats["avg_activation"] = sum(activations) / len(activations) if activations else 0

        return self.get_stats()

    def get_stats(self) -> Dict[str, Any]:
        """获取优化统计"""
        return {
            **self.stats,
            "step_count": self.step_count,
            "consolidated_count": len(self.engram_memory.consolidated_indices),
            "active_count": len(self.engram_memory.active_indices),
            "memory_utilization": len(self.engram_memory.active_indices) / self.config.memory_size
        }

    def get_optimized_kv(self) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """
        获取优化后的 KV 缓存

        Returns:
            (keys, values, indices)
        """
        return self.engram_memory.get_sparse_representation()

    def estimate_overhead(self) -> float:
        """
        估算当前开销

        Returns:
            开销百分比
        """
        base_overhead = 2.0

        memory_util = len(self.engram_memory.active_indices) / self.config.memory_size

        consolidation_bonus = 0.5 if self.stats["consolidations"] > 0 else 0

        compression_bonus = 0.3 if self.config.enable_compression else 0

        estimated = base_overhead + (1 - memory_util) * 3 + consolidation_bonus + compression_bonus

        return min(estimated, 8.0)
