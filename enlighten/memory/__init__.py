"""
工作记忆子模块
包含工作记忆矩阵、熵统计追踪器和活跃索引管理

T3新增:
- SlidingWindowRefresh: 滑动窗口刷新策略
- TopkRefresh: 定期TopK刷新策略
- RefreshResult: 刷新结果数据类
- SignalAdaptivePreprocessor: 信号自适应预处理模块 (v3.1新增)
"""

from .working_memory import (
    WorkingMemory,
    RefreshResult,
    SlidingWindowRefresh,
    TopkRefresh,
)
from .entropy_tracker import EntropyTracker
from .active_indices import ActiveIndices
from .signal_preprocessor import (
    SignalAdaptivePreprocessor,
    SignalPreprocessorConfig,
    StateClassifier,
    FFTPreprocessor,
    LaplacePreprocessor,
    ZTransformPreprocessor,
    SignalWindow,
    StructuredFeatures,
    SignalState,
)

__all__ = [
    "WorkingMemory",
    "RefreshResult",
    "SlidingWindowRefresh",
    "TopkRefresh",
    "EntropyTracker",
    "ActiveIndices",
    "SignalAdaptivePreprocessor",
    "SignalPreprocessorConfig",
    "StateClassifier",
    "FFTPreprocessor",
    "LaplacePreprocessor",
    "ZTransformPreprocessor",
    "SignalWindow",
    "StructuredFeatures",
    "SignalState",
]
