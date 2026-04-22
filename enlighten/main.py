"""
EnlightenLM Main Entry - 主入口
协调L1、L2、L3组件完成推理
"""

import torch
import logging
import importlib.util
from typing import Dict, Optional, Any
from dataclasses import dataclass

from .l1_generation import L1Generation, L1Output
from .l2_working_memory import L2WorkingMemory, L2Output
from .l3_controller import L3Controller, ControlSignals
from .audit.chain import AuditHashChain
from .audit.hmac_sign import HMACSigner
from .audit.offline_review import OfflineReviewService
from .config import EnlightenMode, load_config, ModeConfig, get_mode_preset
from .utils import Timer

_spec = importlib.util.spec_from_file_location("config_module", "enlighten/config.py")
_config_file = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_config_file)
EnlightenConfig = _config_file.EnlightenConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """推理结果"""
    session_id: str
    output: str
    meta_description: str
    user_params: Dict[str, Any]
    cutoff: bool
    cutoff_reason: Optional[str]
    attention_stats: Dict[str, Any]
    tokens: list


class EnlightenLM:
    """
    EnlightenLM 主类

    协调L1、L2、L3组件完成推理流程:

    1. L3根据熵统计生成调控信号
    2. L2压缩上下文，计算熵统计
    3. L1根据调控信号生成输出
    4. 审计日志记录全过程
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: Optional[EnlightenConfig] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or EnlightenConfig()

        self.l1 = L1Generation(model, tokenizer, self._get_l1_config())

        self.l2 = L2WorkingMemory(
            memory_size=self.config.l2.memory_size,
            embedding_dim=self.config.l2.embedding_dim,
            config=self._get_l2_config()
        )

        self.l3 = L3Controller(self._get_l3_config())

        self.audit_chain = AuditHashChain(
            algorithm=self.config.audit.hash_algorithm
        )

        self.hmac_signer = HMACSigner() if self.config.audit.hmac_enabled else None

        self.offline_review = OfflineReviewService(
            audit_chain=self.audit_chain,
            l2_model=self.l2
        )

        self.step_count = 0

        logger.info("EnlightenLM initialized successfully")

    def _get_l1_config(self) -> Dict:
        return {
            "embed_dim": self.config.l1.embed_dim,
            "num_heads": self.config.l1.num_heads,
            "task_bias_dim": self.config.l1.task_bias_dim,
            "memory_size": self.config.l2.memory_size
        }

    def _get_l2_config(self) -> Dict:
        return {
            "memory_size": self.config.l2.memory_size,
            "update_strategy": self.config.l2.update_strategy,
            "entropy_window": self.config.l2.entropy_window,
            "eviction_policy": self.config.l2.eviction_policy
        }

    def _get_l3_config(self) -> Dict:
        return {
            "entropy_threshold": self.config.l3.entropy_threshold,
            "variance_threshold": self.config.l3.variance_threshold,
            "tau_range": self.config.l3.tau_range,
            "theta_range": self.config.l3.theta_range,
            "alpha_range": self.config.l3.alpha_range,
            "van_priority": self.config.l3.van_priority,
            "cutoff_cooldown": self.config.l3.cutoff_cooldown
        }

    def generate(
        self,
        text: str,
        user_params: Optional[Dict] = None,
        session_id: Optional[str] = None,
        max_length: int = 2048
    ) -> InferenceResult:
        """
        执行推理

        Args:
            text: 输入文本
            user_params: 用户参数
            session_id: 会话ID
            max_length: 最大生成长度

        Returns:
            InferenceResult: 推理结果
        """
        import uuid
        session_id = session_id or str(uuid.uuid4())

        with Timer() as timer:
            l3_signals = self._l3_step()

            l2_output = self._l2_step(text)

            l1_output = self._l1_step(text, l3_signals)

            self._log_audit(session_id, text, l1_output, l2_output, l3_signals)

        self.step_count += 1

        result = InferenceResult(
            session_id=session_id,
            output=self._decode(l1_output.output_ids),
            meta_description=self._generate_meta_description(l2_output),
            user_params=user_params or {},
            cutoff=l3_signals.cutoff,
            cutoff_reason=l3_signals.reason,
            attention_stats=l2_output.entropy_stats,
            tokens=l1_output.output_ids.tolist()
        )

        logger.info(f"Inference completed in {timer.get_elapsed():.2f}s, cutoff={result.cutoff}")

        return result

    def _l3_step(self) -> ControlSignals:
        """
        L3元控制步骤
        """
        entropy_stats = self.l2.get_entropy_stats().to_dict()

        control_signals = self.l3(entropy_stats)

        return control_signals

    def _l2_step(self, text: str) -> L2Output:
        """
        L2工作记忆步骤
        """
        inputs = self.tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"]

        hidden_states = self.model(input_ids, output_hidden_states=True).last_hidden_state

        attention_weights = torch.matmul(
            hidden_states, hidden_states.transpose(-2, -1)
        ) / (self.config.l1.embed_dim ** 0.5)
        attention_weights = torch.softmax(attention_weights, dim=-1)

        l2_output = self.l2(hidden_states, attention_weights)

        return l2_output

    def _l1_step(
        self,
        text: str,
        control_signals: ControlSignals
    ) -> L1Output:
        """
        L1生成步骤
        """
        inputs = self.tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"]

        l1_output = self.l1(
            input_ids,
            control_signals={
                "tau": control_signals.tau,
                "theta": control_signals.theta,
                "alpha": control_signals.alpha,
                "stability": control_signals.stability,
                "cutoff": control_signals.cutoff
            }
        )

        return l1_output

    def _log_audit(
        self,
        session_id: str,
        input_text: str,
        l1_output: L1Output,
        l2_output: L2Output,
        control_signals: ControlSignals
    ) -> None:
        """
        记录审计日志
        """
        data = {
            "session_id": session_id,
            "input_text": input_text,
            "output_text": self._decode(l1_output.output_ids),
            "entropy_stats": l2_output.entropy_stats,
            "cutoff": control_signals.cutoff,
            "cutoff_reason": control_signals.reason,
            "tau": control_signals.tau,
            "theta": control_signals.theta,
            "alpha": control_signals.alpha
        }

        signature = self.hmac_signer.sign(data) if self.hmac_signer else None

        self.audit_chain.append(data, signature)

    def _decode(self, output_ids: torch.Tensor) -> str:
        """
        解码输出
        """
        if output_ids.dim() > 1:
            output_ids = output_ids[0]

        return self.tokenizer.decode(output_ids, skip_special_tokens=True)

    def _generate_meta_description(self, l2_output: L2Output) -> str:
        """
        生成元描述
        """
        entropy_stats = l2_output.entropy_stats

        return (
            f"本次推理的平均注意力熵为{entropy_stats.get('mean', 0):.4f}，"
            f"当前熵为{entropy_stats.get('current', 0):.4f}。"
        )

    def verify_audit_chain(self) -> bool:
        """
        验证审计链完整性
        """
        return self.audit_chain.verify()

    def generate_review_report(self, session_id: str) -> str:
        """
        生成复盘报告
        """
        report = self.offline_review.generate_report(session_id)
        return report.summary

    def set_mode(self, mode: str) -> None:
        """
        热切换运行模式
        会重置L2和L3的状态

        Args:
            mode: 目标模式 ("full" | "balanced" | "lightweight" | "custom")

        Raises:
            ValueError: 未知模式
        """
        valid_modes = ["full", "balanced", "lightweight", "custom"]
        if mode not in valid_modes:
            raise ValueError(f"Unknown mode: {mode}. Valid modes: {valid_modes}")

        old_mode = self.config.mode if hasattr(self.config, 'mode') else "unknown"
        logger.info(f"Switching mode from {old_mode} to {mode}")

        self.config = load_config(mode=mode)

        self._update_components_from_config()

        self.l2.reset()
        self.l3.reset()

        logger.info(f"Successfully switched to {mode} mode")

    def _update_components_from_config(self) -> None:
        """
        根据配置更新组件参数
        """
        if hasattr(self.config, 'van_level'):
            van_level = self.config.van_level
            logger.info(f"Updating VAN level to {van_level}")

        if hasattr(self.config, 'use_topk_refresh'):
            use_refresh = self.config.use_topk_refresh
            refresh_interval = getattr(self.config, 'refresh_interval', 32)
            logger.info(f"Updating refresh: use_topk_refresh={use_refresh}, interval={refresh_interval}")

        if hasattr(self.config, 'cutoff_cooldown'):
            cooldown_steps = self.config.cutoff_cooldown
            self.l3.cutoff_cooldown = cooldown_steps
            logger.info(f"Updating L3 cooldown to {cooldown_steps}")

    def get_mode(self) -> str:
        """
        获取当前运行模式

        Returns:
            当前模式名称
        """
        return self.config.mode if hasattr(self.config, 'mode') else "unknown"

    def get_status(self) -> Dict[str, Any]:
        """
        获取系统状态

        Returns:
            状态信息字典
        """
        return {
            "mode": self.get_mode(),
            "step_count": self.step_count,
            "l2_memory_size": self.config.l2.memory_size if hasattr(self.config, 'l2') else None,
            "l3_cooldown": self.l3.cooldown_counter,
            "l3_stats": self.l3.get_statistics() if hasattr(self.l3, 'get_statistics') else {}
        }
