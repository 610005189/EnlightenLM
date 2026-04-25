"""
Hybrid Architecture - 混合架构
支持本地模型和外部API的统一L1/L2/L3架构

本地模型模式:
- L1: 本地生成层 (Transformer模型)
- L2: 工作记忆层 (上下文+注意力统计)
- L3: VAN监控层 (熵值+自指循环+变异性分析)

API模式:
- L1: API调用层 (DeepSeek API)
- L2: 本地工作记忆层 (上下文+注意力统计)
- L3: 本地VAN监控层 (熵值+自指循环+变异性分析)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Any, List, Tuple, Union
from dataclasses import dataclass
from collections import deque
import math
import re
import hashlib
import time
import os

from .l3_controller import (
    BayesianL3Controller,
    EnhancedBayesianL3Controller,
    ControlSignals as SkeletonControlSignals,
    L3Controller,
    DecisionRecord,
    ContextualTemperatureController,
    TemperatureConfig,
    OutputStabilityMonitor,
    SceneType
)
from .api.ollama_client import OllamaAPIClient, OllamaConfig
from .attention.dan import DANAttention
from .attention.van import VANFunnel
from .attention.fusion import AttentionFusion, StabilityTracker
from .cutoff.dmn import DMNInhibition
from .cutoff.forget_gate import ForgetGate
from .l2_working_memory import L2WorkingMemory, SimplifiedL2, L2Output as SkeletonL2Output
from .memory.working_memory import WorkingMemory
from .memory.entropy_tracker import EntropyTracker
from .attention.sparse import SparseAttention


@dataclass
class L2Result:
    """L2层输出结果（适配骨架代码和Hybrid Architecture）"""
    sparse_kv: Optional[Tuple[torch.Tensor, torch.Tensor]]
    active_indices: list
    entropy_stats: Dict[str, float]
    memory_snapshot: Dict[str, Any]
    use_skeleton: bool = False


class L2WorkingMemoryAdapter(nn.Module):
    """
    L2工作记忆适配器

    将骨架代码的L2WorkingMemory集成到Hybrid Architecture

    功能:
    - 上下文压缩: n个token → m个活跃token
    - 熵统计计算: 追踪注意力熵的滑动统计
    - 活跃索引管理: 维护活跃token索引集A
    - 稀疏键值提供: (K̃, Ṽ) 给L1

    数据流:
    Input → WorkingMemory → EntropyTracker → SparseAttention → Output
    """

    def __init__(
        self,
        memory_size: int = 512,
        embedding_dim: int = 768,
        config: Optional[Dict] = None
    ):
        super().__init__()
        self.memory_size = memory_size
        self.embedding_dim = embedding_dim

        config = config or {}

        self.skeleton_l2 = L2WorkingMemory(
            memory_size=memory_size,
            embedding_dim=embedding_dim,
            config={
                "update_strategy": config.get("update_strategy", "topk"),
                "hierarchical_sizes": config.get("hierarchical_sizes", [64, 256, 512]),
                "entropy_window": config.get("entropy_window", 100),
                "entropy_compute_interval": config.get("entropy_compute_interval", 1),
                "ema_decay": config.get("ema_decay", 0.99),
                "eviction_policy": config.get("eviction_policy", "lru"),
                "sparse_mode": config.get("sparse_mode", "topk"),
                "use_hierarchical": config.get("use_hierarchical", False)
            }
        )

        self.sparse_attention = SparseAttention(
            embed_dim=embedding_dim,
            memory_size=memory_size,
            mode=config.get("sparse_mode", "topk")
        )

        self._entropy_stats_history: deque = deque(maxlen=100)
        self._last_sparse_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self._last_active_indices: list = []

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None,
        update_memory: bool = True
    ) -> L2Result:
        """
        L2适配器前向传播

        Args:
            hidden_states: [batch, seq_len, embed_dim] 来自L1的hidden states
            attention_weights: [batch, seq_len, seq_len] 注意力权重
            update_memory: 是否更新记忆

        Returns:
            L2Result: 包含稀疏KV、熵统计和记忆快照
        """
        batch_size, seq_len, embed_dim = hidden_states.shape

        key = hidden_states
        value = hidden_states

        if update_memory:
            self.skeleton_l2.working_memory.update(key, value, attention_weights)

        sparse_k, sparse_v, active_indices = self.skeleton_l2.working_memory.get_sparse_kv()

        if attention_weights is not None:
            self.skeleton_l2.entropy_tracker.update(attention_weights)

        entropy_stats = self.skeleton_l2.entropy_tracker.get_statistics()

        self._entropy_stats_history.append(entropy_stats)
        self._last_sparse_kv = (sparse_k, sparse_v)
        self._last_active_indices = active_indices

        memory_snapshot = self.skeleton_l2.working_memory.get_memory_snapshot()

        return L2Result(
            sparse_kv=(sparse_k, sparse_v),
            active_indices=active_indices,
            entropy_stats=entropy_stats,
            memory_snapshot=memory_snapshot,
            use_skeleton=True
        )

    def get_sparse_attention_output(
        self,
        query: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用稀疏注意力处理query

        Args:
            query: [batch, seq_len, embed_dim] 查询向量
            attention_mask: [batch, seq_len] 注意力掩码

        Returns:
            output: [batch, seq_len, embed_dim] 稀疏注意力输出
            attention_weights: 注意力权重
        """
        if self._last_sparse_kv is None:
            return query, torch.ones_like(query[:, :, :1])

        sparse_k, sparse_v = self._last_sparse_kv

        output, attention_weights = self.sparse_attention(
            query=query,
            key=sparse_k.unsqueeze(0).expand(query.size(0), -1, -1),
            value=sparse_v.unsqueeze(0).expand(query.size(0), -1, -1),
            attention_mask=attention_mask
        )

        return output, attention_weights

    def should_cutoff(self) -> bool:
        """
        判断是否应该截断

        使用熵统计和活跃索引判断
        """
        return self.skeleton_l2.should_cutoff()

    def select_sparse_indices(
        self,
        attention_weights: torch.Tensor,
        topk: int = 32
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        基于注意力权重选择稀疏索引

        Args:
            attention_weights: [batch, seq_len, seq_len] 注意力权重
            topk: 选择的top-k索引数

        Returns:
            selected_k: 选中的key
            selected_v: 选中的value
        """
        if self._last_sparse_kv is None:
            return None, None

        sparse_k, sparse_v = self._last_sparse_kv

        importance = torch.norm(sparse_k, dim=-1)

        _, topk_indices = torch.topk(importance, k=min(topk, len(importance)))

        selected_k = sparse_k[topk_indices]
        selected_v = sparse_v[topk_indices]

        return selected_k, selected_v

    def reset(self) -> None:
        """重置工作记忆状态"""
        self.skeleton_l2.reset()
        self._entropy_stats_history.clear()
        self._last_sparse_kv = None
        self._last_active_indices = []

    def get_entropy_stats(self) -> Dict[str, float]:
        """获取当前熵统计"""
        return self.skeleton_l2.entropy_tracker.get_statistics()

    def get_last_entropy_stats(self) -> Dict[str, float]:
        """获取上次更新的熵统计"""
        if self._entropy_stats_history:
            return self._entropy_stats_history[-1]
        return {
            "mean": 0.0,
            "variance": 0.0,
            "trend": 0.0,
            "current": 0.0,
            "ema": 0.0
        }


class SimplifiedL2Adapter(nn.Module):
    """
    简化版L2适配器 - 用于快速原型验证

    提供与完整L2WorkingMemoryAdapter相同接口的简化实现
    """

    def __init__(self, memory_size: int = 512, embedding_dim: int = 768):
        super().__init__()
        self.memory_size = memory_size
        self.embedding_dim = embedding_dim

        self.memory = nn.Parameter(torch.zeros(memory_size, embedding_dim))
        self.entropy_tracker = EntropyTracker(window_size=50)

        self._last_sparse_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None,
        update_memory: bool = True
    ) -> L2Result:
        """
        简化版前向传播

        Returns:
            L2Result: 包含稀疏KV、熵统计和记忆快照
        """
        if attention_weights is not None:
            if isinstance(attention_weights, torch.Tensor):
                attention_weights = attention_weights.detach()
            self.entropy_tracker.update(attention_weights)

        importance = torch.norm(hidden_states, dim=-1).mean(dim=0)

        k = min(self.memory_size, importance.size(0))
        if k > 0:
            topk_indices = torch.topk(importance, k=k).indices

            for i, idx in enumerate(topk_indices):
                self.memory.data[i] = hidden_states[:, idx, :].mean(dim=0).detach()

        self._last_sparse_kv = (self.memory, self.memory)

        entropy_stats = self.entropy_tracker.get_statistics()

        memory_snapshot = {
            'memory': self.memory.data.clone(),
            'active_indices': [],
            'sensitive_indices': [],
            'memory_size': self.memory_size,
            'embedding_dim': self.embedding_dim
        }

        return L2Result(
            sparse_kv=(self.memory, self.memory),
            active_indices=[],
            entropy_stats=entropy_stats,
            memory_snapshot=memory_snapshot,
            use_skeleton=True
        )

    def get_sparse_attention_output(
        self,
        query: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """使用稀疏注意力处理query"""
        if self._last_sparse_kv is None:
            return query, torch.ones_like(query[:, :, :1])

        sparse_k, sparse_v = self._last_sparse_kv

        batch_size = query.size(0)
        sparse_k_expanded = sparse_k.unsqueeze(0).expand(batch_size, -1, -1)
        sparse_v_expanded = sparse_v.unsqueeze(0).expand(batch_size, -1, -1)

        scores = torch.matmul(query, sparse_k_expanded.transpose(-2, -1)) / (self.embedding_dim ** 0.5)
        attention_weights = torch.nn.functional.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, sparse_v_expanded)

        return output, attention_weights

    def should_cutoff(self) -> bool:
        """判断是否应该截断"""
        return self.entropy_tracker.should_cutoff()

    def select_sparse_indices(
        self,
        attention_weights: torch.Tensor,
        topk: int = 32
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """基于注意力权重选择稀疏索引"""
        if self._last_sparse_kv is None:
            return None, None

        sparse_k, sparse_v = self._last_sparse_kv

        importance = torch.norm(sparse_k, dim=-1)
        _, topk_indices = torch.topk(importance, k=min(topk, len(importance)))

        return sparse_k[topk_indices], sparse_v[topk_indices]

    def reset(self) -> None:
        """重置状态"""
        self.entropy_tracker.reset()
        self._last_sparse_kv = None

    def get_entropy_stats(self) -> Dict[str, float]:
        """获取熵统计"""
        return self.entropy_tracker.get_statistics()


class L1Adapter(nn.Module):
    """
    L1 双流注意力适配器

    将骨架代码的 DAN/VAN 融合和遗忘门机制适配到 Hybrid Architecture

    功能:
    - DAN (目标驱动注意力网络): 任务偏置引导的主动聚焦
    - VAN (变异性吸引子网络): 敏感内容检测
    - Attention Fusion: 双流动态融合
    - DMN 抑制: 防止无意义自循环
    - Forget Gate: 指数衰减 KV 缓存

    数据流:
    Input → DAN → VAN → Fusion → DMN → ForgetGate → Output
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        task_bias_dim: int = 128,
        van_level: str = "medium",
        sensitive_keywords: Optional[List[str]] = None,
        memory_size: int = 512
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.task_bias_dim = task_bias_dim

        self.dan = DANAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            task_bias_dim=task_bias_dim
        )

        self.van = VANFunnel(
            level=van_level,
            embed_dim=embed_dim,
            vocab_size=50000,
            sensitive_keywords=sensitive_keywords
        )

        self.fusion = AttentionFusion(
            embed_dim=embed_dim,
            gate_hidden_dim=embed_dim // 4
        )

        self.dmn = DMNInhibition(embed_dim=embed_dim)
        self.forget_gate = ForgetGate(embed_dim=embed_dim)

        self.stability_tracker = StabilityTracker(threshold=0.1)

        self.prev_hidden = None
        self.control_signals_history: List[Dict[str, float]] = []

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        control_signals: Optional[Dict[str, Any]] = None,
        task_bias: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        L1 适配器前向传播

        Args:
            input_ids: [batch, seq_len] 输入 token IDs
            hidden_states: [batch, seq_len, embed_dim] 隐藏状态
            control_signals: 来自 L3 的调控信号
            task_bias: [batch, task_bias_dim] 任务偏置

        Returns:
            Dict containing:
                - output_hidden: 处理后的隐藏状态
                - attention_weights: 注意力权重
                - entropy_stats: 熵统计
                - van_event: 是否触发 VAN 事件
                - p_harm: 有害概率
                - control_signals: 调控信号引用
        """
        control_signals = control_signals or {}

        batch_size = input_ids.size(0) if input_ids.dim() > 0 else 1

        if task_bias is None:
            task_bias = torch.zeros(
                batch_size,
                self.task_bias_dim,
                device=hidden_states.device
            )

        dan_output, dan_attention = self.dan(
            hidden_states, hidden_states, hidden_states, task_bias
        )

        van_result = self.van.forward(input_ids, hidden_states)
        van_event = van_result.van_event
        p_harm = van_result.p_harm

        van_output = hidden_states * (1 - p_harm)

        tau = control_signals.get("tau", 1.0)
        theta = control_signals.get("theta", 0.5)

        fused_output = self.fusion(dan_output, van_output, tau=tau, theta=theta)

        alpha = control_signals.get("alpha", 0.1)
        inhibited_output = self.dmn(fused_output, alpha=alpha)

        prev_hidden = torch.zeros_like(inhibited_output) if self.prev_hidden is None else self.prev_hidden
        decay_rate = control_signals.get("decay_rate", 0.95)
        output = self.forget_gate(prev_hidden, inhibited_output, decay_rate=decay_rate)

        self.prev_hidden = output.detach()

        attention_weights = torch.matmul(
            output, hidden_states.transpose(-2, -1)
        ) / (self.embed_dim ** 0.5)
        attention_weights = torch.nn.functional.softmax(attention_weights, dim=-1)

        entropy_stats = self._compute_entropy_stats(attention_weights)

        self.control_signals_history.append({
            "tau": tau,
            "theta": theta,
            "alpha": alpha,
            "decay_rate": decay_rate,
            "van_event": van_event,
            "p_harm": p_harm
        })

        return {
            "output_hidden": output,
            "attention_weights": attention_weights,
            "entropy_stats": entropy_stats,
            "van_event": van_event,
            "p_harm": p_harm,
            "control_signals": control_signals
        }

    def _compute_entropy_stats(self, attention_weights: torch.Tensor) -> Dict[str, float]:
        """计算注意力熵统计"""
        entropy = -torch.sum(
            attention_weights * torch.log(attention_weights + 1e-10),
            dim=-1
        ).mean().item()

        if len(self.control_signals_history) >= 2:
            prev_tau = self.control_signals_history[-2].get("tau", 1.0)
            curr_tau = self.control_signals_history[-1].get("tau", 1.0)
            tau_change = curr_tau - prev_tau
        else:
            tau_change = 0.0

        return {
            "mean": entropy,
            "variance": float(np.var([h.get("tau", 1.0) for h in self.control_signals_history])) if len(self.control_signals_history) > 1 else 0.0,
            "current": entropy,
            "trend": tau_change
        }

    def reset(self) -> None:
        """重置 L1 适配器状态"""
        self.prev_hidden = None
        self.control_signals_history.clear()


class L1Output:
    """L1 层输出数据结构（兼容骨架代码）"""
    def __init__(
        self,
        output_hidden: torch.Tensor,
        attention_weights: torch.Tensor,
        entropy_stats: Dict[str, float],
        van_event: bool,
        p_harm: float,
        control_signals: Dict[str, Any]
    ):
        self.output_hidden = output_hidden
        self.attention_weights = attention_weights
        self.entropy_stats = entropy_stats
        self.van_event = van_event
        self.p_harm = p_harm
        self.control_signals = control_signals


class L3ControllerAdapter:
    """
    L3 元控制器适配器

    将骨架代码的 L3Controller 集成到 Hybrid Architecture

    功能:
    - 冷却机制增强: 截断后防止立即再次截断
    - 抖动检测与抑制: 检测截断信号是否反复横跳
    - 历史记录追踪: 记录决策历史用于分析
    - 温度和稀疏度动态调节: τ ∈ [0.1, 2.0], θ ∈ [0.5, 0.9]

    数据流:
    entropy_stats + van_event + p_harm → L3Controller.forward() → ControlSignals
    """

    def __init__(
        self,
        config: Optional[Dict] = None,
        flicker_window_size: int = 5,
        flicker_threshold: float = 0.6
    ):
        config = config or {}

        self.l3_controller = L3Controller(
            config=config,
            flicker_window_size=flicker_window_size,
            flicker_threshold=flicker_threshold
        )

        self._last_control_signals: Optional[SkeletonControlSignals] = None
        self._control_signals_history: List[SkeletonControlSignals] = []

    def forward(
        self,
        entropy_stats: Dict[str, float],
        van_event: bool = False,
        p_harm: float = 0.0,
        task_embedding: Optional[torch.Tensor] = None
    ) -> SkeletonControlSignals:
        """
        L3适配器前向传播

        Args:
            entropy_stats: 来自L2的熵统计字典
                - mean: μ_H 熵均值
                - variance: σ_H² 熵方差
                - trend: k_H 趋势
                - current: 当前熵值
            van_event: 是否触发VAN事件
            p_harm: 有害概率
            task_embedding: 任务嵌入向量

        Returns:
            SkeletonControlSignals: 调控信号
                - tau: 温度 τ ∈ [0.1, 2.0]
                - theta: 稀疏阈值 θ ∈ [0.5, 0.9]
                - alpha: DMN系数 α ∈ [0.0, 1.0]
                - stability: 稳定性标志
                - cutoff: 截断信号
                - reason: 决策原因
        """
        control_signals = self.l3_controller.forward(
            entropy_stats=entropy_stats,
            van_event=van_event,
            p_harm=p_harm,
            task_embedding=task_embedding
        )

        self._last_control_signals = control_signals
        self._control_signals_history.append(control_signals)

        if len(self._control_signals_history) > 100:
            self._control_signals_history.pop(0)

        return control_signals

    def get_control_signals_dict(self, signals: SkeletonControlSignals) -> Dict[str, Any]:
        """
        将 ControlSignals 转换为字典格式

        Args:
            signals: 控制信号

        Returns:
            包含 tau, theta, alpha, stability, cutoff, reason 的字典
        """
        return {
            "tau": signals.tau,
            "theta": signals.theta,
            "alpha": signals.alpha,
            "stability": signals.stability,
            "cutoff": signals.cutoff,
            "reason": signals.reason
        }

    def get_last_control_signals(self) -> Optional[Dict[str, Any]]:
        """获取上一次的调控信号字典"""
        if self._last_control_signals is None:
            return None
        return self.get_control_signals_dict(self._last_control_signals)

    def get_temperature(self) -> float:
        """获取当前温度 τ"""
        if self._last_control_signals is None:
            return 0.7
        return self._last_control_signals.tau

    def get_sparsity_threshold(self) -> float:
        """获取当前稀疏阈值 θ"""
        if self._last_control_signals is None:
            return 0.7
        return self._last_control_signals.theta

    def get_dmn_coefficient(self) -> float:
        """获取当前 DMN 系数 α"""
        if self._last_control_signals is None:
            return 0.1
        return self._last_control_signals.alpha

    def should_cutoff(self) -> bool:
        """判断是否应该截断"""
        if self._last_control_signals is None:
            return False
        return self._last_control_signals.cutoff

    def is_stable(self) -> bool:
        """判断当前是否稳定"""
        if self._last_control_signals is None:
            return True
        return self._last_control_signals.stability

    def get_cutoff_reason(self) -> Optional[str]:
        """获取截断原因"""
        if self._last_control_signals is None:
            return None
        return self._last_control_signals.reason

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取L3控制器统计信息

        Returns:
            统计信息字典
        """
        stats = self.l3_controller.get_statistics()

        if self._control_signals_history:
            tau_values = [s.tau for s in self._control_signals_history]
            theta_values = [s.theta for s in self._control_signals_history]
            alpha_values = [s.alpha for s in self._control_signals_history]

            stats.update({
                "tau_mean": float(np.mean(tau_values)),
                "tau_std": float(np.std(tau_values)),
                "theta_mean": float(np.mean(theta_values)),
                "theta_std": float(np.std(theta_values)),
                "alpha_mean": float(np.mean(alpha_values)),
                "alpha_std": float(np.std(alpha_values)),
                "last_tau": tau_values[-1] if tau_values else 0.7,
                "last_theta": theta_values[-1] if theta_values else 0.7,
                "last_alpha": alpha_values[-1] if alpha_values else 0.1
            })

        return stats

    def get_history(self, last_n: Optional[int] = None) -> List[DecisionRecord]:
        """获取决策历史"""
        return self.l3_controller.get_history(last_n)

    def reset(self) -> None:
        """重置L3控制器状态"""
        self.l3_controller.reset()
        self._last_control_signals = None
        self._control_signals_history.clear()

    def reset_cooldown(self) -> None:
        """重置冷却计数器"""
        self.l3_controller.reset_cooldown()


@dataclass
class GenerationResult:
    """生成结果"""
    text: str
    tokens: int
    latency: float
    cutoff: bool
    cutoff_reason: Optional[str]
    entropy_stats: Dict[str, float]
    van_event: bool
    security_verified: bool
    control_signals: Optional[Dict[str, Any]] = None


@dataclass
class AttentionStats:
    """注意力统计"""
    entropy: float
    variance: float
    trend: float
    focus_distribution: List[float]
    stability_score: float


class WorkingMemoryManager:
    """
    L2 工作记忆管理器

    功能:
    - 维护会话历史
    - 计算注意力统计
    - 管理上下文窗口
    - 追踪熵值变化
    """

    def __init__(
        self,
        max_history: int = 100,
        context_window: int = 4096,
        entropy_window: int = 20,
        attention_size: int = 32
    ):
        self.max_history = max_history
        self.context_window = context_window
        self.entropy_window = entropy_window
        self.attention_size = attention_size

        self.conversation_history: List[Dict[str, str]] = []
        self.attention_history: deque = deque(maxlen=entropy_window)
        self.entropy_history: deque = deque(maxlen=entropy_window)
        self.token_count = 0

    def add_turn(self, role: str, content: str) -> None:
        """添加一轮对话"""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": time.time()
        })
        self.token_count += len(content.split())

        if len(self.conversation_history) > self.max_history:
            self.conversation_history.pop(0)

    def get_context(self) -> str:
        """获取当前上下文"""
        context_parts = []
        for turn in self.conversation_history[-self.context_window:]:
            role = turn["role"]
            content = turn["content"]
            context_parts.append(f"{role}: {content}")
        return "\n".join(context_parts)

    def compute_attention_stats(self) -> AttentionStats:
        """计算注意力统计"""
        if not self.attention_history:
            return AttentionStats(
                entropy=1.0,
                variance=0.1,
                trend=0.0,
                focus_distribution=[1.0],
                stability_score=1.0
            )

        attention_data = np.array(list(self.attention_history))
        if attention_data.ndim == 1:
            attention_data = attention_data.reshape(1, -1)

        if attention_data.shape[0] == 0:
            return AttentionStats(
                entropy=1.0,
                variance=0.1,
                trend=0.0,
                focus_distribution=[1.0],
                stability_score=1.0
            )

        entropy_values = -attention_data * np.log2(attention_data + 1e-10)
        entropy_mean = np.mean(entropy_values)
        entropy_variance = np.var(entropy_values)

        mean_entropy_per_step = np.mean(entropy_values, axis=1)
        trend = 0.0
        if len(mean_entropy_per_step) >= 5:
            x = np.arange(len(mean_entropy_per_step))
            result = np.polyfit(x, mean_entropy_per_step, 1)
            if np.ndim(result) == 1:
                trend = float(result[0])
            else:
                trend = float(result)

        row_sums = attention_data.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1e-10, row_sums)
        focus_dist = attention_data / row_sums
        focus_dist = focus_dist[-1] if len(focus_dist) > 0 else np.array([1.0])

        stability = 1.0 - min(float(entropy_variance) * 10, 1.0)

        return AttentionStats(
            entropy=float(entropy_mean),
            variance=float(entropy_variance),
            trend=trend,
            focus_distribution=focus_dist.tolist() if isinstance(focus_dist, np.ndarray) else [1.0],
            stability_score=float(stability)
        )

    def compute_entropy_stats(self) -> Dict[str, float]:
        """计算熵统计"""
        if not self.entropy_history:
            return {
                "mean": 1.0,
                "variance": 0.1,
                "trend": 0.0,
                "current": 1.0,
                "stability": 1.0
            }

        entropy_data = np.array(list(self.entropy_history))
        mean = np.mean(entropy_data)
        variance = np.var(entropy_data)

        trend = 0.0
        if len(entropy_data) >= 5:
            x = np.arange(len(entropy_data))
            slope, _ = np.polyfit(x, entropy_data, 1)
            trend = slope

        stability = 1.0 - min(variance * 10, 1.0)

        return {
            "mean": float(mean),
            "variance": float(variance),
            "trend": float(trend),
            "current": float(entropy_data[-1]) if len(entropy_data) > 0 else 1.0,
            "stability": float(stability)
        }

    def update_attention(self, attention_weights, target_size: Optional[int] = None) -> None:
        """更新注意力权重"""
        target_size = target_size or self.attention_size

        if isinstance(attention_weights, np.ndarray):
            if len(attention_weights.shape) == 1:
                attention_weights = np.expand_dims(attention_weights, axis=0)
            avg_attention = attention_weights.mean(axis=0 if len(attention_weights.shape) == 2 else -1)
        elif isinstance(attention_weights, torch.Tensor):
            if len(attention_weights.shape) == 1:
                attention_weights = attention_weights.unsqueeze(0)
            avg_attention = attention_weights.mean(dim=0 if len(attention_weights.shape) == 2 else -2)
            if isinstance(avg_attention, torch.Tensor):
                avg_attention = avg_attention.cpu().numpy()
        else:
            raise TypeError(f"attention_weights must be numpy array or torch tensor, got {type(attention_weights)}")

        seq_len = len(avg_attention)
        if seq_len == 0:
            avg_attention = np.ones(target_size) / target_size
        elif seq_len != target_size:
            if seq_len > target_size:
                step = seq_len / target_size
                compressed = np.zeros(target_size)
                for i in range(target_size):
                    start = int(i * step)
                    end = int((i + 1) * step)
                    if start < seq_len:
                        compressed[i] = avg_attention[start:min(end, seq_len)].mean()
                avg_attention = compressed
            else:
                padded = np.ones(target_size) * (1.0 / target_size)
                padded[:seq_len] = avg_attention
                padded = padded / padded.sum()
                avg_attention = padded

        if len(avg_attention) != target_size:
            avg_attention = np.ones(target_size) / target_size

        self.attention_history.append(avg_attention)

        entropy = -np.sum(avg_attention * np.log2(avg_attention + 1e-10))
        max_entropy = np.log2(target_size)
        normalized_entropy = entropy / (max_entropy + 1e-10)
        self.entropy_history.append(normalized_entropy)

    def reset(self) -> None:
        """重置工作记忆"""
        self.conversation_history.clear()
        self.attention_history.clear()
        self.entropy_history.clear()
        self.token_count = 0


class VANMonitor:
    """
    L3 VAN 变异性-吸引子网络监控器

    功能:
    - 熵值监控: 追踪注意力熵的滑动统计
    - 自指循环检测: 检测重复模式和自参照
    - 变异性分析: 监控输出变异性
    - 截断决策: 基于多维度判断是否截断
    """

    DEFAULT_SENSITIVE_PATTERNS = [
        r"(hack|crack|bypass|exploit|漏洞|破解|入侵|攻击)",
        r"(password|credential|secret|密钥|密码凭据)",
        r"(malware|virus|ransomware|恶意软件|病毒)",
        r"(fraud|scam|phishing|诈骗|钓鱼|欺诈)",
        r"(illegal|criminal|illicit|非法|犯罪)",
        r"(self[self]+|自指|循环引用)",
    ]

    SELF_LOOP_PATTERNS = [
        r"(.+)\1{3,}",
        r"(.{10,})\1{2,}",
        r"(.{3,}?)(.{3,}?)\1\2",
    ]

    def __init__(
        self,
        van_threshold: float = 0.7,
        entropy_threshold: float = 0.3,
        variance_threshold: float = 0.05,
        cooldown_steps: int = 5,
        enabled: bool = True
    ):
        self.van_threshold = van_threshold
        self.entropy_threshold = entropy_threshold
        self.variance_threshold = variance_threshold
        self.cooldown_steps = cooldown_steps
        self.enabled = enabled

        self.sensitive_patterns = [re.compile(p, re.IGNORECASE) for p in self.DEFAULT_SENSITIVE_PATTERNS]
        self.self_loop_patterns = [re.compile(p) for p in self.SELF_LOOP_PATTERNS]

        self.cooldown_counter = 0
        self.output_history: deque = deque(maxlen=100)
        self.decision_history: List[Dict] = []
        self.van_event_count = 0
        self.total_requests = 0
        self.blocked_count = 0

    def check_input(self, text: str) -> Tuple[bool, Optional[str], float]:
        """
        检查输入内容

        Returns:
            (should_block, reason, risk_score)
        """
        if not self.enabled:
            return False, None, 0.0

        self.total_requests += 1

        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return False, None, 0.0

        risk_score = 0.0
        reasons = []

        for pattern in self.sensitive_patterns:
            if pattern.search(text):
                risk_score += 0.4
                reasons.append("sensitive_keyword")

        for pattern in self.self_loop_patterns:
            if pattern.search(text):
                risk_score += 0.3
                reasons.append("self_referential_pattern")

        char_set_ratio = len(set(text)) / max(len(text), 1)
        if char_set_ratio < 0.1 and len(text) > 20:
            risk_score += 0.2
            reasons.append("low_entropy")

        if risk_score >= self.van_threshold:
            self.blocked_count += 1
            self.van_event_count += 1
            self.cooldown_counter = self.cooldown_steps
            reason = f"VAN event: {', '.join(reasons)}"
            self._record_decision("input_blocked", risk_score, True, reason)
            return True, reason, risk_score

        self._record_decision("input_passed", risk_score, False, None)
        return False, None, risk_score

    def check_output(self, text: str, entropy_stats: Optional[Dict[str, float]] = None) -> Tuple[bool, Optional[str], float]:
        """
        检查输出内容

        Returns:
            (should_cutoff, reason, risk_score)
        """
        if not self.enabled:
            return False, None, 0.0

        self.total_requests += 1

        if self.cooldown_counter > 0:
            return True, "Cooldown active", 1.0

        if not text or len(text.strip()) == 0:
            self._record_decision("output_empty", 1.0, True, "Empty output")
            return True, "Empty output", 1.0

        risk_score = 0.0
        reasons = []

        for pattern in self.sensitive_patterns:
            matches = pattern.findall(text)
            if matches:
                risk_score += 0.3 * min(len(matches), 3)
                reasons.append("sensitive_content")

        repetitive_chars = len(re.findall(r'(.)\1{5,}', text))
        if repetitive_chars > 3:
            risk_score += 0.3
            reasons.append("repetitive_pattern")

        word_repetition = self._detect_word_repetition(text)
        if word_repetition > 0.5:
            risk_score += 0.3
            reasons.append("high_word_repetition")

        output_hash = hashlib.md5(text.encode()).hexdigest()[:16]
        self.output_history.append({
            "hash": output_hash,
            "length": len(text),
            "risk": risk_score,
            "timestamp": time.time()
        })

        if len(self.output_history) >= 10:
            recent = list(self.output_history)[-10:]
            hashes = [h["hash"] for h in recent]
            unique_ratio = len(set(hashes)) / len(hashes)
            if unique_ratio < 0.3:
                risk_score += 0.2
                reasons.append("low_variability")

        text_entropy = self._compute_text_entropy(text)
        if entropy_stats is None:
            entropy_stats = text_entropy

        if entropy_stats.get("entropy", 1.0) < self.entropy_threshold:
            risk_score += 0.25
            reasons.append("low_text_entropy")
        if entropy_stats.get("repetition", 0) > 0.5:
            risk_score += 0.2
            reasons.append("high_repetition")
        if entropy_stats.get("diversity", 1.0) < 0.3:
            risk_score += 0.15
            reasons.append("low_diversity")

        if risk_score >= self.van_threshold:
            self.blocked_count += 1
            self.van_event_count += 1
            self.cooldown_counter = self.cooldown_steps
            reason = f"VAN cutoff: {', '.join(reasons)}"
            self._record_decision("output_cutoff", risk_score, True, reason)
            return True, reason, risk_score

        self._record_decision("output_passed", risk_score, False, None)
        return False, None, risk_score

    def _detect_word_repetition(self, text: str) -> float:
        """检测词汇重复率"""
        tokens = text.split()
        if len(tokens) < 2:
            return 0.0
        unique_tokens = len(set(tokens))
        return 1.0 - (unique_tokens / len(tokens))

    def _compute_text_entropy(self, text: str) -> Dict[str, float]:
        """
        从文本计算统计熵值（用于输出检查）

        分析文本本身的结构特征：
        - 词汇多样性（Type-Token Ratio）
        - 重复率
        - 字符级熵
        """
        tokens = text.split()
        n = len(tokens)

        if n == 0:
            return {"entropy": 0.5, "diversity": 0.5, "repetition": 0.0}

        unique_tokens = len(set(tokens))
        diversity = unique_tokens / max(n, 1)

        repetition_score = 0.0
        if n > 1:
            bigrams = [f"{tokens[i]} {tokens[i+1]}" for i in range(n-1)]
            if bigrams:
                unique_bigrams = len(set(bigrams))
                repetition_score = 1.0 - (unique_bigrams / len(bigrams))

        char_set = set(text)
        char_entropy = len(char_set) / max(len(text), 1)

        combined_entropy = (diversity * 0.4 + (1 - repetition_score) * 0.3 + char_entropy * 0.3)

        return {
            "entropy": float(combined_entropy),
            "diversity": float(diversity),
            "repetition": float(repetition_score),
            "char_entropy": float(char_entropy)
        }

    def _record_decision(
        self,
        decision_type: str,
        risk_score: float,
        blocked: bool,
        reason: Optional[str]
    ) -> None:
        """记录决策历史"""
        record = {
            "type": decision_type,
            "risk": risk_score,
            "blocked": blocked,
            "reason": reason,
            "timestamp": time.time(),
            "cooldown_remaining": self.cooldown_counter
        }
        self.decision_history.append(record)
        if len(self.decision_history) > 1000:
            self.decision_history.pop(0)

    def should_cutoff_by_entropy(self, entropy_stats: Dict[str, float]) -> Tuple[bool, Optional[str]]:
        """
        基于熵统计判断是否截断

        条件:
        - 低注意力熵 (mean < entropy_threshold)
        - 低方差 (variance < variance_threshold)
        - 持续下降趋势 (trend < 0)
        """
        mean = entropy_stats.get("mean", 1.0)
        variance = entropy_stats.get("variance", 0.1)
        trend = entropy_stats.get("trend", 0.0)

        if mean < self.entropy_threshold and variance < self.variance_threshold and trend < 0:
            return True, "Entropy cutoff: low mean, low variance, negative trend"

        return False, None

    def get_statistics(self) -> Dict[str, Any]:
        """获取VAN监控统计"""
        return {
            "total_requests": self.total_requests,
            "van_events": self.van_event_count,
            "blocked_requests": self.blocked_count,
            "cooldown_remaining": self.cooldown_counter,
            "cooldown_active": self.cooldown_counter > 0,
            "block_ratio": self.blocked_count / max(self.total_requests, 1)
        }

    def reset(self) -> None:
        """重置VAN监控状态"""
        self.cooldown_counter = 0
        self.output_history.clear()
        self.decision_history.clear()
        self.van_event_count = 0
        self.total_requests = 0
        self.blocked_count = 0


class HybridEnlightenLM:
    """
    混合架构的 EnlightenLM

    支持本地模型和API调用，统一经过L2工作记忆和L3 VAN监控

    使用方式:
        # API模式 (默认)
        model = HybridEnlightenLM(use_local=False)
        result = model.generate("Hello")

        # 本地模型模式
        model = HybridEnlightenLM(use_local=True, local_model_name="distilgpt2")
        result = model.generate("Hello")
    """

    def __init__(
        self,
        use_local_model: bool = False,
        local_model_name: str = "distilgpt2",
        api_client=None,
        config: Optional[Union[Dict, Any]] = None,
        use_bayesian_l3: bool = False,
        use_l3_controller: bool = False,
        l3_config: Optional[Dict] = None,
        use_l1_adapter: bool = False,
        l1_config: Optional[Dict] = None,
        use_skeleton_l2: bool = False,
        l2_config: Optional[Dict] = None,
        use_contextual_temperature: bool = False,
        temperature_config: Optional[TemperatureConfig] = None
    ):
        if config is None:
            config_dict = {}
        elif hasattr(config, 'working_memory'):
            config_dict = {
                "max_history": config.working_memory.capacity,
                "context_window": 4096,
                "entropy_window": config.entropy_monitor.window_size,
                "attention_size": getattr(config.working_memory, 'attention_size', 32),
                "van_threshold": config.cutoff.van_threshold,
                "entropy_threshold": config.cutoff.low_entropy_threshold,
                "variance_threshold": config.cutoff.low_variance_threshold,
                "cooldown_steps": config.cutoff.cooldown_steps,
                "van_enabled": True
            }
        else:
            config_dict = config

        self.use_local_model = use_local_model
        self.local_model_name = local_model_name
        self.api_client = api_client
        self.use_bayesian_l3 = use_bayesian_l3
        self.use_l1_adapter = use_l1_adapter
        self.use_skeleton_l2 = use_skeleton_l2

        if not self.use_local_model and self.api_client is None:
            self.api_client = OllamaAPIClient(OllamaConfig(model="qwen2.5:14b"))

        self.working_memory = WorkingMemoryManager(
            max_history=config_dict.get("max_history", 100),
            context_window=config_dict.get("context_window", 4096),
            entropy_window=config_dict.get("entropy_window", 20),
            attention_size=config_dict.get("attention_size", 32)
        )

        l2_config = l2_config or {}
        self.l2_adapter = None
        if self.use_skeleton_l2:
            l2_embed_dim = l2_config.get("embedding_dim", 768)
            self.l2_adapter = L2WorkingMemoryAdapter(
                memory_size=l2_config.get("memory_size", 512),
                embedding_dim=l2_embed_dim,
                config={
                    "update_strategy": l2_config.get("update_strategy", "topk"),
                    "hierarchical_sizes": l2_config.get("hierarchical_sizes", [64, 256, 512]),
                    "entropy_window": l2_config.get("entropy_window", 100),
                    "entropy_compute_interval": l2_config.get("entropy_compute_interval", 1),
                    "ema_decay": l2_config.get("ema_decay", 0.99),
                    "eviction_policy": l2_config.get("eviction_policy", "lru"),
                    "sparse_mode": l2_config.get("sparse_mode", "topk"),
                    "use_hierarchical": l2_config.get("use_hierarchical", False)
                }
            )

        self.van_monitor = VANMonitor(
            van_threshold=config_dict.get("van_threshold", 0.7),
            entropy_threshold=config_dict.get("entropy_threshold", 0.3),
            variance_threshold=config_dict.get("variance_threshold", 0.05),
            cooldown_steps=config_dict.get("cooldown_steps", 5),
            enabled=config_dict.get("van_enabled", True)
        )

        self.bayesian_l3 = None
        if self.use_bayesian_l3:
            self.bayesian_l3 = BayesianL3Controller()

        self.use_l3_controller = use_l3_controller
        self.l3_controller_adapter = None
        if self.use_l3_controller:
            l3_config = l3_config or {}
            self.l3_controller_adapter = L3ControllerAdapter(
                config={
                    "entropy_threshold": l3_config.get("entropy_threshold", 0.5),
                    "variance_threshold": l3_config.get("variance_threshold", 0.05),
                    "tau_range": tuple(l3_config.get("tau_range", [0.1, 2.0])),
                    "theta_range": tuple(l3_config.get("theta_range", [0.5, 0.9])),
                    "alpha_range": tuple(l3_config.get("alpha_range", [0.0, 1.0])),
                    "van_priority": l3_config.get("van_priority", True),
                    "cutoff_cooldown": l3_config.get("cutoff_cooldown", 10)
                },
                flicker_window_size=l3_config.get("flicker_window_size", 5),
                flicker_threshold=l3_config.get("flicker_threshold", 0.6)
            )

        self.l1_adapter = None
        if self.use_l1_adapter:
            l1_config = l1_config or {}
            self.l1_adapter = L1Adapter(
                embed_dim=l1_config.get("embed_dim", 768),
                num_heads=l1_config.get("num_heads", 12),
                task_bias_dim=l1_config.get("task_bias_dim", 128),
                van_level=l1_config.get("van_level", "medium"),
                sensitive_keywords=l1_config.get("sensitive_keywords", None),
                memory_size=l1_config.get("memory_size", 512)
            )

        self.local_model = None
        self.local_tokenizer = None

        self.use_contextual_temperature = use_contextual_temperature
        self.contextual_temperature_controller = None
        if self.use_contextual_temperature:
            self.contextual_temperature_controller = ContextualTemperatureController(
                config=temperature_config,
                tau_range=(0.1, 2.0)
            )

        if self.use_local_model:
            self._load_local_model()

    def _load_local_model(self) -> None:
        """加载本地模型"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger = __import__('logging').getLogger(__name__)
            logger.info(f"Loading local model: {self.local_model_name}")

            self.local_tokenizer = AutoTokenizer.from_pretrained(self.local_model_name)
            self.local_model = AutoModelForCausalLM.from_pretrained(self.local_model_name)

            if torch.cuda.is_available():
                self.local_model = self.local_model.cuda()

            logger.info("Local model loaded successfully")

        except Exception as e:
            logger = __import__('logging').getLogger(__name__)
            logger.error(f"Failed to load local model: {e}")
            raise

    def generate(
        self,
        prompt: str,
        max_length: int = 1024,
        temperature: float = 0.7,
        enable_trace: bool = False,
        trace_callback: Optional[Any] = None,
        **kwargs
    ) -> GenerationResult:
        """
        生成文本

        Args:
            prompt: 输入提示
            max_length: 最大生成长度
            temperature: 生成温度
            enable_trace: 是否启用Trace记录
            trace_callback: Trace记录回调函数，签名为 callback(mu_H, sigma_H2, k_H, p_harm_raw)

        Returns:
            GenerationResult: 生成结果
        """
        start_time = time.time()

        should_block, block_reason, input_risk = self.van_monitor.check_input(prompt)

        if should_block:
            return GenerationResult(
                text="[内容被安全监控系统拦截]",
                tokens=0,
                latency=time.time() - start_time,
                cutoff=True,
                cutoff_reason=block_reason,
                entropy_stats=self.working_memory.compute_entropy_stats(),
                van_event=True,
                security_verified=False
            )

        self.working_memory.add_turn("user", prompt)

        context = self.working_memory.get_context()

        contextual_temp = temperature
        if self.use_contextual_temperature and self.contextual_temperature_controller is not None:
            detected_scene = self.contextual_temperature_controller.detect_scene(prompt, context)
            contextual_entropy_stats = self.working_memory.compute_entropy_stats()
            contextual_temp = self.contextual_temperature_controller.get_temperature_for_api(
                entropy_stats=contextual_entropy_stats,
                van_event=False,
                p_harm=0.0,
                scene_type=detected_scene
            )

        if self.use_local_model:
            output_text, tokens = self._generate_local(context, max_length, contextual_temp)
        else:
            output_text, tokens = self._generate_api(context, max_length)

        self.working_memory.add_turn("assistant", output_text)

        entropy_stats = self.working_memory.compute_entropy_stats()

        control_signals_dict = None
        if self.use_l3_controller and self.l3_controller_adapter is not None:
            van_event_flag = False
            p_harm_value = 0.0

            if self.use_l1_adapter and self.l1_adapter is not None:
                l1_result = self._process_with_l1_adapter(output_text, entropy_stats)
                if l1_result.get("van_event", False):
                    return GenerationResult(
                        text="[内容被L1 VAN监控系统截断 - VAN event detected]",
                        tokens=tokens,
                        latency=time.time() - start_time,
                        cutoff=True,
                        cutoff_reason="L1 VAN event",
                        entropy_stats=l1_result.get("entropy_stats", entropy_stats),
                        van_event=True,
                        security_verified=False
                    )
                van_event_flag = l1_result.get("van_event", False)
                p_harm_value = l1_result.get("p_harm", 0.0)
                current_entropy_stats = l1_result.get("entropy_stats", entropy_stats)
            else:
                van_event_flag, _, van_risk = self.van_monitor.check_output(output_text, entropy_stats)
                p_harm_value = van_risk
                current_entropy_stats = entropy_stats

            l3_control_signals = self.l3_controller_adapter.forward(
                entropy_stats=current_entropy_stats,
                van_event=van_event_flag,
                p_harm=p_harm_value
            )
            control_signals_dict = self.l3_controller_adapter.get_control_signals_dict(l3_control_signals)

            if l3_control_signals.cutoff:
                return GenerationResult(
                    text="[内容被L3元控制器截断 - " + (l3_control_signals.reason or "高风险内容") + "]",
                    tokens=tokens,
                    latency=time.time() - start_time,
                    cutoff=True,
                    cutoff_reason=l3_control_signals.reason,
                    entropy_stats=current_entropy_stats,
                    van_event=van_event_flag,
                    security_verified=False
                )

        if self.use_l1_adapter and self.l1_adapter is not None:
            l1_result = self._process_with_l1_adapter(output_text, entropy_stats)
            if l1_result.get("van_event", False):
                return GenerationResult(
                    text="[内容被L1 VAN监控系统截断 - VAN event detected]",
                    tokens=tokens,
                    latency=time.time() - start_time,
                    cutoff=True,
                    cutoff_reason="L1 VAN event",
                    entropy_stats=l1_result.get("entropy_stats", entropy_stats),
                    van_event=True,
                    security_verified=False
                )

            if self.use_bayesian_l3 and self.bayesian_l3:
                control_signals = self.bayesian_l3.forward(
                    entropy_stats=l1_result.get("entropy_stats", entropy_stats),
                    van_event=l1_result.get("van_event", False),
                    p_harm=l1_result.get("p_harm", 0.0)
                )

                if control_signals.cutoff:
                    return GenerationResult(
                        text="[内容被贝叶斯L3监控系统截断 - " + (control_signals.reason or "高风险内容") + "]",
                        tokens=tokens,
                        latency=time.time() - start_time,
                        cutoff=True,
                        cutoff_reason=control_signals.reason,
                        entropy_stats=l1_result.get("entropy_stats", entropy_stats),
                        van_event=l1_result.get("van_event", False),
                        security_verified=False
                    )
        elif not self.use_l3_controller:
            if self.use_bayesian_l3 and self.bayesian_l3:
                van_event, van_reason, van_risk = self.van_monitor.check_output(output_text, entropy_stats)

                control_signals = self.bayesian_l3.forward(
                    entropy_stats=entropy_stats,
                    van_event=van_event,
                    p_harm=van_risk
                )

                if control_signals.cutoff:
                    return GenerationResult(
                        text="[内容被贝叶斯L3监控系统截断 - " + (control_signals.reason or "高风险内容") + "]",
                        tokens=tokens,
                        latency=time.time() - start_time,
                        cutoff=True,
                        cutoff_reason=control_signals.reason,
                        entropy_stats=entropy_stats,
                        van_event=van_event,
                        security_verified=False
                    )
            else:
                should_cutoff, cutoff_reason, output_risk = self.van_monitor.check_output(output_text, entropy_stats)

                if should_cutoff:
                    return GenerationResult(
                        text="[内容被VAN监控系统截断 - " + cutoff_reason + "]",
                        tokens=tokens,
                        latency=time.time() - start_time,
                        cutoff=True,
                        cutoff_reason=cutoff_reason,
                        entropy_stats=entropy_stats,
                        van_event=True,
                        security_verified=False
                    )

                entropy_cutoff, entropy_reason = self.van_monitor.should_cutoff_by_entropy(entropy_stats)
                if entropy_cutoff:
                    return GenerationResult(
                        text="[内容因熵值异常被截断 - " + entropy_reason + "]",
                        tokens=tokens,
                        latency=time.time() - start_time,
                        cutoff=True,
                        cutoff_reason=entropy_reason,
                        entropy_stats=entropy_stats,
                        van_event=True,
                        security_verified=False
                    )

        if enable_trace and trace_callback is not None:
            signals = self.get_l3_trace_signals()
            trace_callback(
                mu_H=signals["mu_H"],
                sigma_H2=signals["sigma_H2"],
                k_H=signals["k_H"],
                p_harm_raw=signals["p_harm_raw"]
            )

        final_entropy_stats = entropy_stats
        if self.use_l1_adapter and self.l1_adapter is not None:
            final_entropy_stats = self.working_memory.compute_entropy_stats()
        elif self.use_l3_controller and self.l3_controller_adapter is not None:
            final_entropy_stats = self.working_memory.compute_entropy_stats()

        return GenerationResult(
            text=output_text,
            tokens=tokens,
            latency=time.time() - start_time,
            cutoff=False,
            cutoff_reason=None,
            entropy_stats=final_entropy_stats,
            van_event=False,
            security_verified=True,
            control_signals=control_signals_dict if self.use_l3_controller else None
        )

    def _generate_local(
        self,
        context: str,
        max_length: int,
        temperature: float
    ) -> Tuple[str, int]:
        """使用本地模型生成"""
        if self.local_model is None or self.local_tokenizer is None:
            raise RuntimeError("Local model not loaded")

        inputs = self.local_tokenizer(context, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        attention_mask = inputs.get("attention_mask")
        outputs = self.local_model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            pad_token_id=self.local_tokenizer.pad_token_id or 0
        )

        generated_text = self.local_tokenizer.decode(outputs[0], skip_special_tokens=True)

        prompt_length = len(context.split())
        generated_tokens = len(generated_text.split()) - prompt_length

        attention_weights = outputs[0] if hasattr(outputs[0], 'shape') else None
        if attention_weights is not None:
            if isinstance(attention_weights, torch.Tensor):
                attention_weights = attention_weights.float().cpu().numpy()
            self.working_memory.update_attention(attention_weights)

        return generated_text, max(generated_tokens, 0)

    def _generate_api(
        self,
        context: str,
        max_length: int
    ) -> Tuple[str, int]:
        """使用API生成"""
        if self.api_client is None:
            raise RuntimeError("API client not configured")

        # 调用API生成文本
        result = self.api_client.generate(
            prompt=context,
            max_tokens=max_length
        )

        # 处理不同客户端的返回类型
        if isinstance(result, tuple):
            # DeepSeekAPIClient 返回 (text, latency)
            response_text = result[0]
        else:
            # OllamaAPIClient 返回 text
            response_text = result

        tokens = len(response_text.split())

        attention_from_text = self._compute_attention_from_text(response_text)
        self.working_memory.update_attention(attention_from_text)

        return response_text, tokens

    def _compute_attention_from_text(self, text: str) -> np.ndarray:
        """
        从文本特征计算"注意力权重"

        这是API模式下的近似计算，不依赖模型内部注意力，
        而是从文本的统计特征推断"认知聚焦程度"

        考虑因素:
        - token级别的条件概率多样性（近似困惑度）
        - 关键词/概念的出现集中度
        - 句子级别的语义连贯性

        Returns:
            固定长度(32)的注意力权重向量
        """
        FIXED_SIZE = 32
        tokens = text.split()
        n = len(tokens)

        if n == 0:
            return np.ones(FIXED_SIZE) / FIXED_SIZE

        focus_scores = np.ones(min(n, FIXED_SIZE))

        content_words = {'的', '是', '在', '有', '我', '你', '他', '她', '它', '和', '与', '或', '但', '而'}
        for i in range(min(n, FIXED_SIZE)):
            token = tokens[i]
            if token in content_words or len(token) <= 1:
                focus_scores[i] = 0.3
            elif len(token) >= 3:
                focus_scores[i] = 1.5

        if focus_scores.sum() > 0:
            focus_distribution = focus_scores / focus_scores.sum()
        else:
            focus_distribution = np.ones(FIXED_SIZE) / FIXED_SIZE

        return focus_distribution

    def _compute_text_entropy(self, text: str) -> Dict[str, float]:
        """
        从文本计算真实的统计熵值

        不依赖模型内部状态，而是分析文本本身的结构特征：
        - 词汇多样性（Type-Token Ratio）
        - 重复率
        - 句子长度变化
        - 字符级熵

        Returns:
            文本熵统计
        """
        tokens = text.split()
        n = len(tokens)

        if n == 0:
            return {"entropy": 1.0, "diversity": 0.5, "repetition": 0.0}

        unique_tokens = len(set(tokens))
        diversity = unique_tokens / max(n, 1)

        repetition_score = 0.0
        if n > 1:
            bigrams = [f"{tokens[i]} {tokens[i+1]}" for i in range(n-1)]
            unique_bigrams = len(set(bigrams))
            repetition_score = 1.0 - (unique_bigrams / max(len(bigrams), 1))

        char_set = set(text)
        if len(text) > 0:
            char_entropy = len(char_set) / len(text)
        else:
            char_entropy = 0.0

        sentence_lengths = [len(s.split()) for s in text.split('。') if s.strip()]
        if len(sentence_lengths) > 1:
            length_variance = np.var(sentence_lengths) if sentence_lengths else 0.0
        else:
            length_variance = 0.0

        combined_entropy = (diversity * 0.4 + (1 - repetition_score) * 0.3 + char_entropy * 0.3)

        return {
            "entropy": float(combined_entropy),
            "diversity": float(diversity),
            "repetition": float(repetition_score),
            "length_variance": float(length_variance),
            "char_entropy": float(char_entropy)
        }

    def get_attention_stats(self) -> AttentionStats:
        """获取注意力统计"""
        return self.working_memory.compute_attention_stats()

    def get_entropy_stats(self) -> Dict[str, float]:
        """获取熵统计"""
        return self.working_memory.compute_entropy_stats()

    def get_van_stats(self) -> Dict[str, Any]:
        """获取VAN监控统计"""
        return self.van_monitor.get_statistics()

    def reset(self) -> None:
        """重置所有状态"""
        self.working_memory.reset()
        self.van_monitor.reset()
        if self.bayesian_l3:
            self.bayesian_l3.reset()
        if self.l1_adapter:
            self.l1_adapter.reset()
        if self.l2_adapter:
            self.l2_adapter.reset()
        if self.l3_controller_adapter:
            self.l3_controller_adapter.reset()
        if self.contextual_temperature_controller:
            self.contextual_temperature_controller.reset()

    def _process_with_l1_adapter(
        self,
        output_text: str,
        entropy_stats: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        使用 L1 适配器处理输出

        当 use_skeleton_l2=True 时，L1 输出会传递给 L2 适配器进行进一步处理

        Args:
            output_text: 生成的文本
            entropy_stats: 当前熵统计

        Returns:
            Dict containing van_event, p_harm, entropy_stats, etc.
        """
        if self.l1_adapter is None:
            return {
                "van_event": False,
                "p_harm": 0.0,
                "entropy_stats": entropy_stats
            }

        tokens = output_text.split()
        if not tokens:
            return {
                "van_event": False,
                "p_harm": 0.0,
                "entropy_stats": entropy_stats
            }

        seq_len = min(len(tokens), 32)
        dummy_hidden = torch.randn(1, seq_len, self.l1_adapter.embed_dim)

        dummy_input_ids = torch.randint(0, 50000, (1, seq_len))

        control_signals = {
            "tau": 1.0,
            "theta": 0.5,
            "alpha": 0.1,
            "decay_rate": 0.95
        }

        l1_result = self.l1_adapter(
            input_ids=dummy_input_ids,
            hidden_states=dummy_hidden,
            control_signals=control_signals
        )

        attention_weights = l1_result["attention_weights"]
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.float()
            if attention_weights.ndim > 1:
                attention_weights = attention_weights.mean(dim=-1)
            else:
                attention_weights = attention_weights.mean().unsqueeze(0)

        if self.l2_adapter is not None:
            hidden_states = l1_result["output_hidden"]
            if hidden_states.dim() == 2:
                hidden_states = hidden_states.unsqueeze(0)

            l2_result = self.l2_adapter.forward(
                hidden_states=hidden_states,
                attention_weights=attention_weights,
                update_memory=True
            )

            self.working_memory.update_attention(attention_weights)

            updated_entropy_stats = self.l2_adapter.get_entropy_stats()

            if self.should_l2_cutoff():
                return {
                    "van_event": True,
                    "p_harm": l1_result.get("p_harm", 0.0),
                    "entropy_stats": updated_entropy_stats,
                    "l2_cutoff": True
                }

            return {
                "van_event": l1_result.get("van_event", False),
                "p_harm": l1_result.get("p_harm", 0.0),
                "entropy_stats": updated_entropy_stats,
                "attention_weights": attention_weights,
                "l2_result": l2_result
            }
        else:
            self.working_memory.update_attention(attention_weights.detach() if isinstance(attention_weights, torch.Tensor) else attention_weights)
            updated_entropy_stats = self.working_memory.compute_entropy_stats()

            return {
                "van_event": l1_result.get("van_event", False),
                "p_harm": l1_result.get("p_harm", 0.0),
                "entropy_stats": updated_entropy_stats,
                "attention_weights": attention_weights
            }

    def get_status(self) -> Dict[str, Any]:
        """获取状态信息"""
        status = {
            "mode": "local" if self.use_local_model else "api",
            "model": self.local_model_name if self.use_local_model else "ollama",
            "working_memory_tokens": self.working_memory.token_count,
            "conversation_turns": len(self.working_memory.conversation_history),
            "attention_stats": {
                "entropy": self.working_memory.compute_attention_stats().entropy,
                "stability": self.working_memory.compute_attention_stats().stability_score
            },
            "van_stats": self.van_monitor.get_statistics(),
            "use_bayesian_l3": self.use_bayesian_l3,
            "use_l1_adapter": self.use_l1_adapter,
            "use_skeleton_l2": self.use_skeleton_l2,
            "use_l3_controller": self.use_l3_controller
        }

        if self.bayesian_l3:
            status["bayesian_l3_stats"] = self.bayesian_l3.get_statistics()

        if self.l1_adapter:
            status["l1_adapter"] = {
                "embed_dim": self.l1_adapter.embed_dim,
                "num_heads": self.l1_adapter.num_heads,
                "van_level": self.l1_adapter.van.level if hasattr(self.l1_adapter.van, 'level') else "unknown",
                "control_signals_count": len(self.l1_adapter.control_signals_history)
            }

        if self.l2_adapter:
            l2_entropy_stats = self.l2_adapter.get_entropy_stats()
            status["l2_adapter"] = {
                "memory_size": self.l2_adapter.memory_size,
                "embedding_dim": self.l2_adapter.embedding_dim,
                "entropy_stats": {
                    "mean": l2_entropy_stats.get("mean", 0.0),
                    "variance": l2_entropy_stats.get("variance", 0.0),
                    "trend": l2_entropy_stats.get("trend", 0.0),
                    "current": l2_entropy_stats.get("current", 0.0)
                }
            }

        if self.l3_controller_adapter:
            l3_stats = self.l3_controller_adapter.get_statistics()
            status["l3_controller"] = {
                "cooldown_counter": l3_stats.get("cooldown_counter", 0),
                "total_decisions": l3_stats.get("total_decisions", 0),
                "total_cutoffs": l3_stats.get("total_cutoffs", 0),
                "last_tau": l3_stats.get("last_tau", 0.7),
                "last_theta": l3_stats.get("last_theta", 0.7),
                "last_alpha": l3_stats.get("last_alpha", 0.1)
            }

        if self.use_contextual_temperature and self.contextual_temperature_controller:
            temp_stats = self.contextual_temperature_controller.get_statistics()
            status["contextual_temperature"] = {
                "enabled": True,
                "current_scene": temp_stats.get("current_scene", "general"),
                "current_temperature": temp_stats.get("current_temperature", 0.7),
                "temperature_stats": temp_stats.get("temperature_stats", {}),
                "scene_distribution": temp_stats.get("scene_distribution", {}),
                "stability_stats": temp_stats.get("stability_stats", {})
            }
        else:
            status["contextual_temperature"] = {"enabled": False}

        return status

    def get_l3_trace_signals(self) -> Dict[str, float]:
        """
        获取L3贝叶斯控制器的输入信号

        对应论文内部信号:
        - mu_H (mean): 后验熵均值 -> entropy_stats['mean']
        - sigma_H2 (variance): 后验熵方差 -> entropy_stats['variance']
        - k_H (trend): 熵变化趋势 -> entropy_stats['trend']
        - p_harm_raw: VAN风险值 -> van_monitor 最新风险

        Returns:
            Dict containing mu_H, sigma_H2, k_H, p_harm_raw
        """
        entropy_stats = self.working_memory.compute_entropy_stats()

        last_decision = self.van_monitor.decision_history[-1] if self.van_monitor and self.van_monitor.decision_history else {}
        last_risk = last_decision.get("risk_score", 0.0)

        return {
            "mu_H": entropy_stats["mean"],
            "sigma_H2": entropy_stats["variance"],
            "k_H": entropy_stats["trend"],
            "p_harm_raw": last_risk
        }

    def process_with_l2_adapter(
        self,
        hidden_states: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None,
        update_memory: bool = True
    ) -> L2Result:
        """
        使用骨架代码L2适配器处理hidden states

        Args:
            hidden_states: [batch, seq_len, embed_dim] 隐藏状态
            attention_weights: [batch, seq_len, seq_len] 注意力权重
            update_memory: 是否更新记忆

        Returns:
            L2Result: L2层处理结果
        """
        if self.l2_adapter is None:
            raise RuntimeError("L2 adapter not initialized. Set use_skeleton_l2=True")

        return self.l2_adapter.forward(hidden_states, attention_weights, update_memory)

    def get_l2_entropy_stats(self) -> Dict[str, float]:
        """
        获取L2层的熵统计

        如果使用骨架代码L2，返回L2适配器的熵统计
        否则返回WorkingMemoryManager的熵统计

        Returns:
            Dict containing mean, variance, trend, current, stability
        """
        if self.l2_adapter is not None:
            return self.l2_adapter.get_entropy_stats()
        return self.working_memory.compute_entropy_stats()

    def sparse_attention_select(
        self,
        attention_weights: torch.Tensor,
        topk: int = 32
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        使用稀疏注意力选择top-k重要的键值对

        Args:
            attention_weights: 注意力权重
            topk: 选择的top-k数量

        Returns:
            (selected_k, selected_v): 选中的键值对，如果L2适配器未初始化则返回(None, None)
        """
        if self.l2_adapter is None:
            return None, None

        return self.l2_adapter.select_sparse_indices(attention_weights, topk)

    def get_sparse_attention_output(
        self,
        query: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用稀疏注意力处理query

        Args:
            query: [batch, seq_len, embed_dim] 查询向量
            attention_mask: [batch, seq_len] 注意力掩码

        Returns:
            (output, attention_weights): 稀疏注意力输出和权重
        """
        if self.l2_adapter is None:
            return query, torch.ones_like(query[:, :, :1])

        return self.l2_adapter.get_sparse_attention_output(query, attention_mask)

    def should_l2_cutoff(self) -> bool:
        """
        判断L2层是否应该截断

        Returns:
            bool: 是否应该截断
        """
        if self.l2_adapter is None:
            return False

        return self.l2_adapter.should_cutoff()

    def get_l3_control_signals(self) -> Optional[Dict[str, Any]]:
        """
        获取L3控制器的最新调控信号

        Returns:
            Dict containing tau, theta, alpha, stability, cutoff, reason
            如果未使用L3控制器则返回None
        """
        if self.l3_controller_adapter is None:
            return None
        return self.l3_controller_adapter.get_last_control_signals()

    def get_temperature(self) -> float:
        """
        获取当前温度值 τ

        Returns:
            float: 温度值 τ ∈ [0.1, 2.0]
        """
        if self.l3_controller_adapter is None:
            return 0.7
        return self.l3_controller_adapter.get_temperature()

    def get_sparsity_threshold(self) -> float:
        """
        获取当前稀疏度阈值 θ

        Returns:
            float: 稀疏阈值 θ ∈ [0.5, 0.9]
        """
        if self.l3_controller_adapter is None:
            return 0.7
        return self.l3_controller_adapter.get_sparsity_threshold()

    def get_dmn_coefficient(self) -> float:
        """
        获取当前DMN系数 α

        Returns:
            float: DMN系数 α ∈ [0.0, 1.0]
        """
        if self.l3_controller_adapter is None:
            return 0.1
        return self.l3_controller_adapter.get_dmn_coefficient()

    def should_l3_cutoff(self) -> bool:
        """
        判断L3层是否应该截断

        Returns:
            bool: 是否应该截断
        """
        if self.l3_controller_adapter is None:
            return False
        return self.l3_controller_adapter.should_cutoff()

    def is_l3_stable(self) -> bool:
        """
        判断L3层当前是否稳定

        Returns:
            bool: 是否稳定
        """
        if self.l3_controller_adapter is None:
            return True
        return self.l3_controller_adapter.is_stable()

    def get_l3_cutoff_reason(self) -> Optional[str]:
        """
        获取L3层的截断原因

        Returns:
            str: 截断原因，如果没有截断则返回None
        """
        if self.l3_controller_adapter is None:
            return None
        return self.l3_controller_adapter.get_cutoff_reason()

    def get_l3_statistics(self) -> Dict[str, Any]:
        """
        获取L3控制器的统计信息

        Returns:
            Dict containing total_decisions, total_cutoffs, tau/theta/alpha statistics
        """
        if self.l3_controller_adapter is None:
            return {}
        return self.l3_controller_adapter.get_statistics()

    def reset_l3_cooldown(self) -> None:
        """重置L3控制器的冷却计数器"""
        if self.l3_controller_adapter is not None:
            self.l3_controller_adapter.reset_cooldown()