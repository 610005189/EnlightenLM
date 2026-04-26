"""
配置加载器
支持从文件加载、环境变量覆盖等功能
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union

from .modes import (
    EnlightenMode,
    ModeConfig,
    MODE_PRESETS,
    get_mode_preset,
    get_mode_from_env,
    VANStreamConfig,
    WorkingMemoryConfig,
    EntropyMonitorConfig,
    CutoffConfig,
    AsyncReviewConfig
)


def load_config(
    mode: Optional[str] = None,
    config_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None
) -> ModeConfig:
    """
    加载配置

    优先级（从低到高）:
    1. 预设模式默认值
    2. YAML 配置文件
    3. 环境变量
    4. overrides 参数字典

    Args:
        mode: 预设模式 ("full" | "balanced" | "lightweight")
        config_path: YAML 配置文件路径
        overrides: 配置覆盖字典

    Returns:
        ModeConfig: 合并后的配置
    """
    if mode is None:
        mode = get_mode_from_env() or "balanced"

    if mode not in MODE_PRESETS:
        raise ValueError(f"Unknown mode: {mode}. Available: {list(MODE_PRESETS.keys())}")

    config = get_mode_preset(mode)

    if config_path and os.path.exists(config_path):
        file_config = _load_from_yaml(config_path)
        config = _merge_config(config, file_config)

    env_overrides = _load_from_env()
    if env_overrides:
        config = _merge_config(config, env_overrides)

    if overrides:
        config = _merge_config(config, overrides)

    return config


def _load_from_yaml(path: str) -> Dict[str, Any]:
    """从 YAML 文件加载配置"""
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    if 'enlighten' in data:
        data = data['enlighten']

    config = {}

    # 处理components
    components = data.get('components', {})

    if 'van_stream' in components:
        config['van_level'] = components['van_stream'].get('level', 'medium')

    if 'gate_fusion' in components:
        config['gate_fusion'] = components['gate_fusion']

    if 'dmn_noise' in components:
        config['dmn_noise'] = components['dmn_noise']

    if 'working_memory' in components:
        config['working_memory'] = components['working_memory']

    if 'entropy_monitor' in components:
        config['entropy_monitor'] = components['entropy_monitor']

    if 'cutoff' in components:
        config['cutoff'] = components['cutoff']

    if 'async_review' in components:
        config['async_review'] = components['async_review']

    if 'model_provider' in components:
        config['model_provider'] = components['model_provider']

    return config


def _load_from_env() -> Dict[str, Any]:
    """从环境变量加载配置覆盖"""
    overrides = {}

    if os.environ.get("ENLIGHTEN_VAN_LEVEL"):
        overrides['van_level'] = os.environ["ENLIGHTEN_VAN_LEVEL"]

    if os.environ.get("ENLIGHTEN_GATE_FUSION"):
        overrides['gate_fusion'] = os.environ["ENLIGHTEN_GATE_FUSION"].lower() == "true"

    if os.environ.get("ENLIGHTEN_DMN_NOISE"):
        overrides['dmn_noise'] = os.environ["ENLIGHTEN_DMN_NOISE"].lower() == "true"

    if os.environ.get("ENLIGHTEN_WORKING_MEMORY_CAPACITY"):
        overrides.setdefault('working_memory', {})['capacity'] = int(os.environ["ENLIGHTEN_WORKING_MEMORY_CAPACITY"])

    if os.environ.get("ENLIGHTEN_WORKING_MEMORY_REFRESH_INTERVAL"):
        overrides.setdefault('working_memory', {})['refresh_interval'] = int(os.environ["ENLIGHTEN_WORKING_MEMORY_REFRESH_INTERVAL"])

    if os.environ.get("ENLIGHTEN_WORKING_MEMORY_USE_TOPK_REFRESH"):
        overrides.setdefault('working_memory', {})['use_topk_refresh'] = os.environ["ENLIGHTEN_WORKING_MEMORY_USE_TOPK_REFRESH"].lower() == "true"

    if os.environ.get("ENLIGHTEN_ENTROPY_WINDOW_SIZE"):
        overrides.setdefault('entropy_monitor', {})['window_size'] = int(os.environ["ENLIGHTEN_ENTROPY_WINDOW_SIZE"])

    if os.environ.get("ENLIGHTEN_CUTOFF_LOW_ENTROPY_THRESHOLD"):
        overrides.setdefault('cutoff', {})['low_entropy_threshold'] = float(os.environ["ENLIGHTEN_CUTOFF_LOW_ENTROPY_THRESHOLD"])

    if os.environ.get("ENLIGHTEN_ASYNC_REVIEW_ENABLED"):
        overrides.setdefault('async_review', {})['enabled'] = os.environ["ENLIGHTEN_ASYNC_REVIEW_ENABLED"].lower() == "true"

    # 模型提供者配置
    if os.environ.get("ENLIGHTEN_MODEL_USE_LOCAL"):
        overrides.setdefault('model_provider', {})['use_local_model'] = os.environ["ENLIGHTEN_MODEL_USE_LOCAL"].lower() == "true"

    if os.environ.get("ENLIGHTEN_MODEL_LOCAL_NAME"):
        overrides.setdefault('model_provider', {})['local_model_name'] = os.environ["ENLIGHTEN_MODEL_LOCAL_NAME"]

    if os.environ.get("ENLIGHTEN_MODEL_API_PROVIDER"):
        overrides.setdefault('model_provider', {})['api_provider'] = os.environ["ENLIGHTEN_MODEL_API_PROVIDER"]

    if os.environ.get("ENLIGHTEN_MODEL_API_MODEL"):
        overrides.setdefault('model_provider', {})['api_model'] = os.environ["ENLIGHTEN_MODEL_API_MODEL"]

    return overrides


def _merge_config(base: ModeConfig, overrides: Dict[str, Any]) -> ModeConfig:
    """合并配置"""
    base_dict = base.to_dict() if isinstance(base, ModeConfig) else base

    merged = base_dict.copy()

    for key, value in overrides.items():
        if key == 'mode':
            continue

        if key in ['working_memory', 'entropy_monitor', 'cutoff', 'async_review', 'model_provider']:
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                # 深度合并字典
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = value
        else:
            merged[key] = value

    return ModeConfig.from_dict(merged)


def save_config(config: ModeConfig, path: str) -> None:
    """
    保存配置到 YAML 文件

    Args:
        config: 模式配置
        path: 保存路径
    """
    data = {
        'enlighten': {
            'mode': config.mode.value,
            'components': {
                'van_stream': {
                    'level': config.van_level
                },
                'gate_fusion': config.gate_fusion,
                'dmn_noise': config.dmn_noise,
                'working_memory': {
                    'capacity': config.working_memory.capacity,
                    'refresh_interval': config.working_memory.refresh_interval,
                    'use_topk_refresh': config.working_memory.use_topk_refresh,
                    'eviction_policy': config.working_memory.eviction_policy
                },
                'entropy_monitor': {
                    'window_size': config.entropy_monitor.window_size
                },
                'cutoff': {
                    'low_entropy_threshold': config.cutoff.low_entropy_threshold,
                    'low_variance_threshold': config.cutoff.low_variance_threshold,
                    'min_duration': config.cutoff.min_duration,
                    'van_threshold': config.cutoff.van_threshold,
                    'cooldown_steps': config.cutoff.cooldown_steps
                },
                'async_review': {
                    'enabled': config.async_review.enabled,
                    'model': config.async_review.model,
                    'interval': config.async_review.interval
                }
            }
        }
    }

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False)


class ConfigManager:
    """
    配置管理器

    支持运行时切换模式和配置热更新
    """

    def __init__(self, initial_mode: str = "balanced"):
        self._mode = initial_mode
        self._config = get_mode_preset(initial_mode)
        self._listeners = []

    @property
    def mode(self) -> str:
        """获取当前模式"""
        return self._mode

    @property
    def config(self) -> ModeConfig:
        """获取当前配置"""
        return self._config

    def set_mode(self, mode: str, reload: bool = True) -> None:
        """
        设置运行模式

        Args:
            mode: 模式名称
            reload: 是否重新加载配置
        """
        if mode not in MODE_PRESETS:
            raise ValueError(f"Unknown mode: {mode}. Available: {list(MODE_PRESETS.keys())}")

        old_mode = self._mode
        self._mode = mode
        self._config = get_mode_preset(mode)

        for listener in self._listeners:
            listener(old_mode, mode, self._config)

    def add_listener(self, listener) -> None:
        """
        添加配置变更监听器

        Args:
            listener: 回调函数，签名: (old_mode, new_mode, config) -> None
        """
        self._listeners.append(listener)

    def remove_listener(self, listener) -> None:
        """移除配置变更监听器"""
        if listener in self._listeners:
            self._listeners.remove(listener)

    def reload(self) -> None:
        """重新加载配置"""
        self._config = get_mode_preset(self._mode)
        for listener in self._listeners:
            listener(self._mode, self._mode, self._config)
