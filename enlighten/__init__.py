"""
EnlightenLM - 觉悟三层架构
基于认知神经科学的大模型安全推理与元认知框架

主要组件:
- HybridEnlightenLM: 核心架构实现（API + 本地模型）
- WorkingMemoryManager: L2工作记忆层
- VANMonitor: L3安全监控
- API Server: FastAPI 服务
- MetaCognition: 元认知自检模块
"""

__version__ = "2.3.0"

from .hybrid_architecture import (
    HybridEnlightenLM,
    GenerationResult,
    WorkingMemoryManager,
    AttentionStats,
    VANMonitor,
)
from .config.modes import (
    EnlightenMode,
    ModeConfig,
    WorkingMemoryConfig,
    EntropyMonitorConfig,
    CutoffConfig,
    ModelProviderConfig,
)
from .config.loader import load_config, ConfigManager
from .metacognition import MetaCognition, MetaCognitionConfig, SelfCheckResult

__all__ = [
    # 核心类
    "HybridEnlightenLM",
    "WorkingMemoryManager",
    "VANMonitor",
    "MetaCognition",
    
    # 数据类
    "GenerationResult",
    "AttentionStats",
    "SelfCheckResult",
    
    # 配置
    "EnlightenMode",
    "ModeConfig",
    "WorkingMemoryConfig",
    "EntropyMonitorConfig",
    "CutoffConfig",
    "ModelProviderConfig",
    "MetaCognitionConfig",
    "load_config",
    "ConfigManager",
]
