"""
Cutoff Decision - 截断决策模块
判断是否应该截断，生成截断响应
"""

import torch
from typing import Dict, Optional, List
from dataclasses import dataclass
from enum import Enum


class CutoffReason(Enum):
    """截断原因枚举"""
    VAN_EVENT = "van_event"
    SELF_REFERENTIAL_LOOP = "self_referential_loop"
    LOW_ENTROPY = "low_entropy"
    HIGH_VARIANCE = "high_variance"
    STABILITY_LOSS = "stability_loss"
    MANUAL = "manual"


@dataclass
class CutoffDecision:
    """
    截断决策

    Attributes:
        should_cutoff: 是否应该截断
        reason: 截断原因
        confidence: 决策置信度
        alternative_action: 替代动作
    """
    should_cutoff: bool
    reason: Optional[CutoffReason]
    confidence: float
    alternative_action: Optional[str] = None


class CutoffDecisionMaker:
    """
    截断决策器

    基于熵统计和VAN事件做出截断决策
    """

    def __init__(
        self,
        entropy_threshold: float = 0.5,
        variance_threshold: float = 0.05,
        trend_threshold: float = 0.0,
        confidence_boost: float = 0.1
    ):
        self.entropy_threshold = entropy_threshold
        self.variance_threshold = variance_threshold
        self.trend_threshold = trend_threshold
        self.confidence_boost = confidence_boost

    def decide(
        self,
        entropy_stats: Dict[str, float],
        van_event: bool = False,
        history: Optional[List[CutoffDecision]] = None
    ) -> CutoffDecision:
        """
        做出截断决策

        Args:
            entropy_stats: 熵统计字典
            van_event: 是否触发VAN事件
            history: 历史决策列表

        Returns:
            CutoffDecision
        """
        if van_event:
            return CutoffDecision(
                should_cutoff=True,
                reason=CutoffReason.VAN_EVENT,
                confidence=1.0,
                alternative_action="immediate_stop"
            )

        mu_h = entropy_stats.get("mean", 0.0)
        sigma_h = entropy_stats.get("variance", 0.0) ** 0.5
        k_h = entropy_stats.get("trend", 0.0)

        reasons = []
        confidence = 0.0

        if mu_h < self.entropy_threshold:
            reasons.append(CutoffReason.LOW_ENTROPY)
            confidence += 0.3

        if sigma_h < self.variance_threshold:
            reasons.append(CutoffReason.HIGH_VARIANCE)
            confidence += 0.3

        if k_h < self.trend_threshold:
            reasons.append(CutoffReason.SELF_REFERENTIAL_LOOP)
            confidence += 0.4

        if history:
            recent_cutoffs = sum(1 for d in history[-5:] if d.should_cutoff)
            if recent_cutoffs >= 3:
                confidence += self.confidence_boost

        should_cutoff = len(reasons) >= 2 and confidence >= 0.6

        return CutoffDecision(
            should_cutoff=should_cutoff,
            reason=reasons[0] if reasons else None,
            confidence=confidence,
            alternative_action="reduce_temperature" if confidence < 0.6 else None
        )


class AdaptiveCutoffDecisionMaker(CutoffDecisionMaker):
    """
    自适应截断决策器
    根据历史反馈调整阈值
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.success_history = []
        self.failure_history = []

    def report_result(self, decision: CutoffDecision, success: bool) -> None:
        """
        报告决策结果，用于自适应调整

        Args:
            decision: 做出的决策
            success: 决策是否正确
        """
        if decision.should_cutoff:
            if success:
                self.success_history.append(decision)
            else:
                self.failure_history.append(decision)

        if len(self.success_history) + len(self.failure_history) > 10:
            self._adjust_thresholds()

    def _adjust_thresholds(self):
        """
        根据历史反馈调整阈值
        """
        total = len(self.success_history) + len(self.failure_history)

        if total == 0:
            return

        failure_rate = len(self.failure_history) / total

        if failure_rate > 0.5:
            self.entropy_threshold *= 1.1
            self.variance_threshold *= 1.1
        elif failure_rate < 0.2:
            self.entropy_threshold *= 0.95
            self.variance_threshold *= 0.95


class EnsembleCutoffDecisionMaker:
    """
    集成截断决策器
    多个决策器投票
    """

    def __init__(self, num_deciders: int = 3):
        self.deciders = [
            CutoffDecisionMaker(
                entropy_threshold=0.4 + i * 0.1,
                variance_threshold=0.04 + i * 0.01
            )
            for i in range(num_deciders)
        ]

    def decide(
        self,
        entropy_stats: Dict[str, float],
        van_event: bool = False
    ) -> CutoffDecision:
        """
        集成决策

        多数投票
        """
        decisions = [d.decide(entropy_stats, van_event) for d in self.deciders]

        cutoff_votes = sum(1 for d in decisions if d.should_cutoff)

        should_cutoff = cutoff_votes > len(decisions) // 2

        avg_confidence = sum(d.confidence for d in decisions) / len(decisions)

        return CutoffDecision(
            should_cutoff=should_cutoff,
            reason=decisions[0].reason if should_cutoff else None,
            confidence=avg_confidence
        )


class ProbabilisticCutoffDecisionMaker:
    """
    概率截断决策器
    输出截断的概率而非硬决策
    """

    def __init__(self, embed_dim: int = 64):
        self.embed_dim = embed_dim

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(4, embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(embed_dim, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, entropy_stats: Dict[str, float]) -> float:
        """
        输出截断概率

        Returns:
            cutoff_probability: 截断概率 ∈ [0, 1]
        """
        features = torch.tensor([
            entropy_stats.get("mean", 0.0),
            entropy_stats.get("variance", 0.0) ** 0.5,
            entropy_stats.get("trend", 0.0),
            entropy_stats.get("current", 0.0)
        ], dtype=torch.float32).unsqueeze(0)

        prob = self.classifier(features).item()

        return prob

    def decide(self, entropy_stats: Dict[str, float]) -> CutoffDecision:
        """
        基于概率做出决策
        """
        prob = self.forward(entropy_stats)

        return CutoffDecision(
            should_cutoff=prob > 0.5,
            reason=CutoffReason.SELF_REFERENTIAL_LOOP if prob > 0.5 else None,
            confidence=prob if prob > 0.5 else 1 - prob
        )
