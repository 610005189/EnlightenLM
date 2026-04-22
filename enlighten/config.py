"""
Configuration Management - 配置管理
"""

import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path


@dataclass
class L1Config:
    """L1生成层配置"""
    model_name: str = "deepseek-ai/DeepSeek-V3"
    device: str = "cuda"
    embed_dim: int = 1024
    num_heads: int = 12
    task_bias_dim: int = 128
    memory_size: int = 512


@dataclass
class L2Config:
    """L2工作记忆层配置"""
    memory_size: int = 512
    embedding_dim: int = 1024
    entropy_window: int = 100
    update_strategy: str = "topk"
    eviction_policy: str = "lru"


@dataclass
class L3Config:
    """L3元控制器配置"""
    entropy_threshold: float = 0.5
    variance_threshold: float = 0.05
    tau_range: List[float] = field(default_factory=lambda: [0.1, 2.0])
    theta_range: List[float] = field(default_factory=lambda: [0.5, 0.9])
    alpha_range: List[float] = field(default_factory=lambda: [0.0, 1.0])
    van_priority: bool = True
    cutoff_cooldown: int = 10


@dataclass
class AuditConfig:
    """审计系统配置"""
    storage_path: str = "logs/audit"
    hash_algorithm: str = "sha256"
    hmac_enabled: bool = True
    key_rotation_interval: int = 86400


@dataclass
class EnlightenConfig:
    """EnlightenLM完整配置"""
    l1: L1Config = field(default_factory=L1Config)
    l2: L2Config = field(default_factory=L2Config)
    l3: L3Config = field(default_factory=L3Config)
    audit: AuditConfig = field(default_factory=AuditConfig)

    @classmethod
    def from_yaml(cls, path: str) -> 'EnlightenConfig':
        """从YAML文件加载配置"""
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict) -> 'EnlightenConfig':
        """从字典加载配置"""
        l1_data = data.get('l1_generation', {})
        l2_data = data.get('l2_working_memory', {})
        l3_data = data.get('l3_controller', {})
        audit_data = data.get('audit', {})

        l1 = L1Config(**l1_data.get('model', {}), **l1_data.get('inference', {}))

        l2 = L2Config(**{
            'memory_size': l2_data.get('memory', {}).get('size', 512),
            'embedding_dim': l2_data.get('memory', {}).get('embedding_dim', 1024),
            'entropy_window': l2_data.get('entropy_tracker', {}).get('window_size', 100),
            'update_strategy': l2_data.get('update_strategy', {}).get('method', 'topk'),
            'eviction_policy': l2_data.get('active_indices', {}).get('eviction_policy', 'lru')
        })

        l3 = L3Config(
            entropy_threshold=l3_data.get('cutoff_criteria', {}).get('entropy_threshold', 0.5),
            variance_threshold=l3_data.get('cutoff_criteria', {}).get('variance_threshold', 0.05),
            tau_range=l3_data.get('control_signal_ranges', {}).get('tau', {}).get('range', [0.1, 2.0]),
            theta_range=l3_data.get('control_signal_ranges', {}).get('theta', {}).get('range', [0.5, 0.9]),
            alpha_range=l3_data.get('control_signal_ranges', {}).get('alpha', {}).get('range', [0.0, 1.0]),
            van_priority=l3_data.get('van_handling', {}).get('priority_enabled', True),
            cutoff_cooldown=l3_data.get('van_handling', {}).get('cool_down_steps', 10)
        )

        audit = AuditConfig(
            storage_path=audit_data.get('storage', {}).get('path', 'logs/audit'),
            hash_algorithm=audit_data.get('hash_chain', {}).get('algorithm', 'sha256'),
            hmac_enabled=audit_data.get('hmac', {}).get('enabled', True),
            key_rotation_interval=audit_data.get('hmac', {}).get('key_rotation_interval', 86400)
        )

        return cls(l1=l1, l2=l2, l3=l3, audit=audit)

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'l1_generation': {
                'model': {
                    'name': self.l1.model_name,
                    'device': self.l1.device,
                    'embed_dim': self.l1.embed_dim
                },
                'inference': {
                    'embed_dim': self.l1.embed_dim,
                    'num_heads': self.l1.num_heads,
                    'task_bias_dim': self.l1.task_bias_dim
                }
            },
            'l2_working_memory': {
                'memory': {
                    'size': self.l2.memory_size,
                    'embedding_dim': self.l2.embedding_dim
                },
                'entropy_tracker': {
                    'window_size': self.l2.entropy_window
                },
                'update_strategy': {
                    'method': self.l2.update_strategy
                },
                'active_indices': {
                    'eviction_policy': self.l2.eviction_policy
                }
            },
            'l3_controller': {
                'cutoff_criteria': {
                    'entropy_threshold': self.l3.entropy_threshold,
                    'variance_threshold': self.l3.variance_threshold
                },
                'control_signal_ranges': {
                    'tau': {'range': self.l3.tau_range},
                    'theta': {'range': self.l3.theta_range},
                    'alpha': {'range': self.l3.alpha_range}
                },
                'van_handling': {
                    'priority_enabled': self.l3.van_priority,
                    'cool_down_steps': self.l3.cutoff_cooldown
                }
            },
            'audit': {
                'storage': {
                    'path': self.audit.storage_path
                },
                'hash_chain': {
                    'algorithm': self.audit.hash_algorithm
                },
                'hmac': {
                    'enabled': self.audit.hmac_enabled,
                    'key_rotation_interval': self.audit.key_rotation_interval
                }
            }
        }

    def save(self, path: str) -> None:
        """保存配置到YAML文件"""
        data = self.to_dict()
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False)


def load_config(config_path: str = "configs/hyperparameters.yaml") -> EnlightenConfig:
    """
    加载配置

    Args:
        config_path: 配置文件路径

    Returns:
        EnlightenConfig: 配置对象
    """
    path = Path(config_path)
    if not path.exists():
        return EnlightenConfig()

    return EnlightenConfig.from_yaml(str(path))
