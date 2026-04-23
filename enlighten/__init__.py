"""
EnlightenLM - 觉悟三层架构
基于认知神经科学的大模型安全推理与元认知框架

主要组件:
- L1Generation: L1生成层 (双流注意力 + DMN抑制)
- L2WorkingMemory: L2工作记忆层 (稀疏注意力 + 熵追踪)
- L3Controller: L3元控制器 (熵监控 + 截断决策)
"""

__version__ = "2.1.0"

from .l1_generation import L1Generation
from .l2_working_memory import L2WorkingMemory
from .l3_controller import L3Controller
from .main import EnlightenLM
from .config import EnlightenMode, ModeConfig, load_config, ConfigManager

__all__ = [
    "L1Generation",
    "L2WorkingMemory",
    "L3Controller",
    "EnlightenLM",
    "EnlightenMode",
    "ModeConfig",
    "load_config",
    "ConfigManager",
]
