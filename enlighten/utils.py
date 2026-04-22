"""
Utility Functions - 工具函数
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import json
import time
from pathlib import Path


def set_seed(seed: int = 42) -> None:
    """
    设置随机种子

    Args:
        seed: 种子值
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def compute_attention_entropy(attention_weights: torch.Tensor) -> float:
    """
    计算注意力熵

    Args:
        attention_weights: [batch, num_heads, seq_len, seq_len] or [batch, seq_len, seq_len]

    Returns:
        entropy: 熵值
    """
    if attention_weights.dim() == 4:
        attn = attention_weights.mean(dim=(1, 2, 3))
    elif attention_weights.dim() == 3:
        attn = attention_weights.mean(dim=(1, 2))
    else:
        attn = attention_weights.mean(dim=-1)

    entropy = -torch.sum(
        attention_weights * torch.log(attention_weights + 1e-10),
        dim=-1
    ).mean().item()

    return entropy


def clip_tensor(
    tensor: torch.Tensor,
    min_val: float,
    max_val: float
) -> torch.Tensor:
    """
    裁剪张量值

    Args:
        tensor: 输入张量
        min_val: 最小值
        max_val: 最大值

    Returns:
        clipped: 裁剪后的张量
    """
    return torch.clamp(tensor, min_val, max_val)


def moving_average(
    values: List[float],
    window: int = 10
) -> List[float]:
    """
    计算移动平均

    Args:
        values: 值列表
        window: 窗口大小

    Returns:
        ma: 移动平均值列表
    """
    if len(values) < window:
        return values

    ma = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        ma.append(sum(values[start:i+1]) / (i - start + 1))

    return ma


def format_timestamp(timestamp: Optional[float] = None) -> str:
    """
    格式化时间戳

    Args:
        timestamp: 时间戳（秒）

    Returns:
        formatted: 格式化的时间字符串
    """
    if timestamp is None:
        timestamp = time.time()

    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """
    安全除法

    Args:
        a: 被除数
        b: 除数
        default: 默认值

    Returns:
        result: 结果
    """
    if b == 0:
        return default
    return a / b


def normalize_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    归一化字典值

    Args:
        data: 输入字典

    Returns:
        normalized: 归一化后的字典
    """
    if not data:
        return data

    values = [v for v in data.values() if isinstance(v, (int, float))]

    if not values:
        return data

    min_val = min(values)
    max_val = max(values)

    if max_val == min_val:
        return data

    result = {}
    for k, v in data.items():
        if isinstance(v, (int, float)):
            result[k] = (v - min_val) / (max_val - min_val)
        else:
            result[k] = v

    return result


class Timer:
    """简单的计时器"""

    def __init__(self):
        self.start_time = None
        self.elapsed = 0

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time

    def get_elapsed(self) -> float:
        """获取经过的时间（秒）"""
        if self.start_time is None:
            return 0
        return time.time() - self.start_time


class PerformanceProfiler:
    """性能分析器"""

    def __init__(self):
        self.records = {}

    def record(self, name: str, duration: float) -> None:
        """记录操作耗时"""
        if name not in self.records:
            self.records[name] = []
        self.records[name].append(duration)

    def get_average(self, name: str) -> float:
        """获取平均耗时"""
        if name not in self.records or not self.records[name]:
            return 0
        return sum(self.records[name]) / len(self.records[name])

    def get_report(self) -> Dict[str, Dict[str, float]]:
        """获取分析报告"""
        report = {}
        for name, durations in self.records.items():
            if durations:
                report[name] = {
                    'count': len(durations),
                    'total': sum(durations),
                    'average': sum(durations) / len(durations),
                    'min': min(durations),
                    'max': max(durations)
                }
        return report


def save_json(data: Dict, path: str) -> None:
    """
    保存JSON文件

    Args:
        data: 数据
        path: 路径
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Dict:
    """
    加载JSON文件

    Args:
        path: 路径

    Returns:
        data: 数据
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """
    统计模型参数量

    Args:
        model: 模型

    Returns:
        total: 总参数量
        trainable: 可训练参数量
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_device(prefer_gpu: bool = True) -> torch.device:
    """
    获取计算设备

    Args:
        prefer_gpu: 是否优先使用GPU

    Returns:
        device: torch设备
    """
    if prefer_gpu and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')
