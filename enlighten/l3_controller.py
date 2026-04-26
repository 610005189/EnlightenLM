"""
L3 Meta Controller - L3元控制器
熵监控 + VAN事件处理 + 截断决策 + 调控信号生成

T4新增功能:
- 冷却机制增强: 截断后防止立即再次截断
- 抖动检测与抑制: 检测截断信号是否反复横跳
- 历史记录追踪: 记录决策历史用于分析
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Any, List, Tuple
from dataclasses import dataclass, field


@dataclass
class ControlSignals:
    """调控信号"""
    tau: float
    theta: float
    alpha: float
    stability: bool
    cutoff: bool
    reason: Optional[str]


@dataclass
class DecisionRecord:
    """决策记录"""
    step: int
    entropy_mean: float
    entropy_variance: float
    van_event: bool
    p_harm: float
    cutoff: bool
    cooldown_remaining: int
    reason: Optional[str]


class L3Controller(nn.Module):
    """
    L3 元注意力控制器 (前额叶模拟)

    T4增强:
    - 冷却机制: 截断后等待指定步数才允许再次截断
    - 抖动检测: 检测截断信号的反复横跳
    - 历史追踪: 记录每步决策用于分析

    输入:
    - L2的熵统计 (μ_H, σ_H², k_H)
    - VAN事件标志
    - 任务嵌入

    输出:
    - 温度 τ ∈ [0.1, 2.0]
    - 稀疏阈值 θ ∈ [0.5, 0.9]
    - DMN系数 α ∈ [0.0, 1.0]
    - 稳定性标志 s ∈ {0, 1}
    - 截断信号 cutoff ∈ {0, 1}
    """

    def __init__(
        self,
        config: Optional[Dict] = None,
        flicker_window_size: int = 5,
        flicker_threshold: float = 0.6
    ):
        super().__init__()
        config = config or {}

        self.entropy_threshold = config.get("entropy_threshold", 0.5)
        self.variance_threshold = config.get("variance_threshold", 0.05)

        self.tau_range = tuple(config.get("tau_range", [0.1, 2.0]))
        self.theta_range = tuple(config.get("theta_range", [0.5, 0.9]))
        self.alpha_range = tuple(config.get("alpha_range", [0.0, 1.0]))

        self.van_priority = config.get("van_priority", True)
        self.cutoff_cooldown = config.get("cutoff_cooldown", 10)

        self.cooldown_counter = 0
        self.last_cutoff_reason = None

        self.flicker_window_size = flicker_window_size
        self.flicker_threshold = flicker_threshold
        self.cutoff_history: List[bool] = []

        self.decision_history: List[DecisionRecord] = []
        self.step_counter = 0

    def forward(
        self,
        entropy_stats: Dict[str, float],
        van_event: bool = False,
        p_harm: float = 0.0,
        task_embedding: Optional[torch.Tensor] = None
    ) -> ControlSignals:
        """
        L3层前向传播 - 元控制决策

        Args:
            entropy_stats: 来自L2的熵统计字典
                - mean: μ_H 熵均值
                - variance: σ_H² 熵方差
                - trend: k_H 趋势
                - current: 当前熵值
            van_event: 是否触发VAN事件
            p_harm: VAN检测的有害概率
            task_embedding: 任务嵌入向量

        Returns:
            ControlSignals: 调控信号
        """
        self.step_counter += 1

        mu_h = entropy_stats.get("mean", 0.0)
        sigma_h = entropy_stats.get("variance", 0.0) ** 0.5
        k_h = entropy_stats.get("trend", 0.0)

        self.cutoff_history.append(False)
        if len(self.cutoff_history) > self.flicker_window_size:
            self.cutoff_history.pop(0)

        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self._record_decision(mu_h, sigma_h, van_event, p_harm, False, "Cooldown")
            return ControlSignals(
                tau=0.7,
                theta=0.7,
                alpha=0.1,
                stability=True,
                cutoff=False,
                reason="Cooldown"
            )

        if self.van_priority and van_event:
            cutoff_signals = self._van_cutoff_response()
            self.cutoff_history[-1] = True
            self._record_decision(mu_h, sigma_h, van_event, p_harm, True, cutoff_signals.reason)
            return cutoff_signals

        if self._should_cutoff(mu_h, sigma_h, k_h):
            if self._detect_flickering():
                self._record_decision(mu_h, sigma_h, van_event, p_harm, False, "Flicker suppressed")
                return self._flicker_suppression_response()

            cutoff_signals = self._cutoff_response(mu_h)
            self.cutoff_history[-1] = True
            self._record_decision(mu_h, sigma_h, van_event, p_harm, True, cutoff_signals.reason)
            return cutoff_signals

        self._record_decision(mu_h, sigma_h, van_event, p_harm, False, None)
        return self._normal_control(task_embedding)

    def _detect_flickering(self) -> bool:
        """
        检测截断信号是否在抖动

        如果最近N次决策反复横跳（cutoff状态不稳定），返回True

        Returns:
            是否检测到抖动
        """
        if len(self.cutoff_history) < self.flicker_window_size:
            return False

        recent = self.cutoff_history[-self.flicker_window_size:]
        num_changes = sum(1 for i in range(1, len(recent)) if recent[i] != recent[i-1])

        change_ratio = num_changes / (len(recent) - 1)
        return change_ratio >= self.flicker_threshold

    def _flicker_suppression_response(self) -> ControlSignals:
        """
        抖动抑制响应
        强制保持当前状态，防止反复横跳
        """
        return ControlSignals(
            tau=0.7,
            theta=0.7,
            alpha=0.1,
            stability=True,
            cutoff=False,
            reason="Flicker suppression: decision stabilized"
        )

    def _record_decision(
        self,
        entropy_mean: float,
        entropy_variance: float,
        van_event: bool,
        p_harm: float,
        cutoff: bool,
        reason: Optional[str]
    ) -> None:
        """
        记录决策历史

        Args:
            entropy_mean: 熵均值
            entropy_variance: 熵方差
            van_event: VAN事件标志
            cutoff: 是否截断
            reason: 决策原因
        """
        record = DecisionRecord(
            step=self.step_counter,
            entropy_mean=entropy_mean,
            entropy_variance=entropy_variance,
            van_event=van_event,
            p_harm=p_harm,
            cutoff=cutoff,
            cooldown_remaining=self.cooldown_counter,
            reason=reason
        )
        self.decision_history.append(record)

        if len(self.decision_history) > 1000:
            self.decision_history.pop(0)

    def _should_cutoff(self, mu_h: float, sigma_h: float, k_h: float) -> bool:
        """
        截断判据检查

        条件:
        - 低注意力熵 (μ_H < entropy_threshold)
        - 低方差 (σ_H < variance_threshold)
        - 持续下降趋势 (k_H < 0)
        """
        return (mu_h < self.entropy_threshold and
                sigma_h < self.variance_threshold and
                k_h < 0)

    def _van_cutoff_response(self) -> ControlSignals:
        """
        VAN事件触发的立即截断
        """
        self.cooldown_counter = self.cutoff_cooldown
        self.last_cutoff_reason = "VAN event"

        return ControlSignals(
            tau=0.1,
            theta=0.9,
            alpha=0.5,
            stability=False,
            cutoff=True,
            reason="VAN event: sensitive content detected"
        )

    def _cutoff_response(self, mu_h: float) -> ControlSignals:
        """
        自指循环检测触发的截断
        """
        self.cooldown_counter = self.cutoff_cooldown
        self.last_cutoff_reason = "Self-referential loop"

        tau = max(self.tau_range[0], min(self.tau_range[1], mu_h * 2))

        return ControlSignals(
            tau=tau,
            theta=0.8,
            alpha=0.3,
            stability=False,
            cutoff=True,
            reason="Self-referential loop detected"
        )

    def _normal_control(
        self,
        task_embedding: Optional[torch.Tensor]
    ) -> ControlSignals:
        """
        正常调控
        """
        tau = 0.7
        theta = 0.7
        alpha = 0.1

        if task_embedding is not None:
            if task_embedding.dim() == 1:
                task_embedding = task_embedding.unsqueeze(0)

            tau = torch.sigmoid(task_embedding[:, 0].mean()).item() * 1.5 + 0.2
            tau = max(self.tau_range[0], min(self.tau_range[1], tau))

        return ControlSignals(
            tau=tau,
            theta=theta,
            alpha=alpha,
            stability=True,
            cutoff=False,
            reason=None
        )

    def get_control_signals_dict(self, signals: ControlSignals) -> Dict[str, Any]:
        """
        将ControlSignals转换为字典
        """
        return {
            "tau": signals.tau,
            "theta": signals.theta,
            "alpha": signals.alpha,
            "stability": signals.stability,
            "cutoff": signals.cutoff,
            "reason": signals.reason
        }

    def reset_cooldown(self) -> None:
        """
        重置冷却计数器
        """
        self.cooldown_counter = 0

    def reset(self) -> None:
        """
        重置L3控制器状态
        """
        self.cooldown_counter = 0
        self.last_cutoff_reason = None
        self.cutoff_history = []
        self.decision_history = []
        self.step_counter = 0

    def get_history(self, last_n: Optional[int] = None) -> List[DecisionRecord]:
        """
        获取决策历史

        Args:
            last_n: 返回最近的N条记录，None表示全部

        Returns:
            决策记录列表
        """
        if last_n is None:
            return self.decision_history.copy()
        return self.decision_history[-last_n:]

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取决策统计信息

        Returns:
            统计信息字典
        """
        if not self.decision_history:
            return {"total_decisions": 0}

        cutoffs = [r for r in self.decision_history if r.cutoff]
        van_events = [r for r in self.decision_history if r.van_event]
        cooldowns = [r for r in self.decision_history if r.reason == "Cooldown"]

        return {
            "total_decisions": len(self.decision_history),
            "total_cutoffs": len(cutoffs),
            "total_van_events": len(van_events),
            "total_cooldowns": len(cooldowns),
            "cooldown_counter": self.cooldown_counter,
            "cutoff_ratio": len(cutoffs) / len(self.decision_history) if self.decision_history else 0
        }


class SimplifiedL3:
    """
    简化版L3元控制器 - 用于快速原型验证
    纯规则实现，无需学习
    """

    def __init__(
        self,
        entropy_threshold: float = 0.5,
        variance_threshold: float = 0.05
    ):
        self.entropy_threshold = entropy_threshold
        self.variance_threshold = variance_threshold

    def forward(
        self,
        entropy_stats: Dict[str, float],
        van_event: bool = False
    ) -> Dict[str, Any]:
        """
        简化版前向传播

        Returns:
            dict包含tau, theta, alpha, stability, cutoff, reason
        """
        mu_h = entropy_stats.get("mean", 0.0)
        sigma_h = entropy_stats.get("variance", 0.0) ** 0.5
        k_h = entropy_stats.get("trend", 0.0)

        if van_event:
            return {
                "tau": 0.1,
                "theta": 0.9,
                "alpha": 0.5,
                "stability": False,
                "cutoff": True,
                "reason": "VAN event"
            }

        if mu_h < self.entropy_threshold and sigma_h < self.variance_threshold and k_h < 0:
            return {
                "tau": 0.3,
                "theta": 0.8,
                "alpha": 0.3,
                "stability": False,
                "cutoff": True,
                "reason": "Low entropy"
            }

        return {
            "tau": 0.7,
            "theta": 0.7,
            "alpha": 0.1,
            "stability": True,
            "cutoff": False,
            "reason": None
        }


class AdaptiveL3Controller(nn.Module):
    """
    自适应L3元控制器
    可根据历史数据学习最优调控策略
    """

    def __init__(self, embed_dim: int = 128):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(4, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        self.tau_predictor = nn.Linear(embed_dim, 1)
        self.theta_predictor = nn.Linear(embed_dim, 1)
        self.alpha_predictor = nn.Linear(embed_dim, 1)

    def forward(self, entropy_stats: Dict[str, float]) -> ControlSignals:
        """
        自适应预测调控信号
        """
        features = torch.tensor([
            entropy_stats.get("mean", 0.0),
            entropy_stats.get("variance", 0.0) ** 0.5,
            entropy_stats.get("trend", 0.0),
            entropy_stats.get("current", 0.0)
        ], dtype=torch.float32).unsqueeze(0)

        encoded = self.encoder(features)

        tau = torch.sigmoid(self.tau_predictor(encoded)).item() * 1.9 + 0.1
        theta = torch.sigmoid(self.theta_predictor(encoded)).item() * 0.4 + 0.5
        alpha = torch.sigmoid(self.alpha_predictor(encoded)).item()

        return ControlSignals(
            tau=tau,
            theta=theta,
            alpha=alpha,
            stability=True,
            cutoff=False,
            reason=None
        )


class BayesianL3Controller(nn.Module):
    """
    贝叶斯L3元控制器 - 基于贝叶斯病因推断

    输入:
    - L2的熵统计 (μ_H, σ_H², k_H, p_harm_raw)

    输出:
    - 温度 τ ∈ [0.2, 2.0]
    - 稀疏阈值 θ ∈ [0.5, 0.9]
    - DMN系数 α ∈ [0.0, 1.0]
    - 稳定性标志 s ∈ {0, 1}
    - 截断信号 cutoff ∈ {0, 1}
    - 连续截断信心 p_harm ∈ [0, 1]
    - 病因后验概率 P(H | o_int)
    """

    def __init__(self, prior_probs=[0.5, 0.2, 0.2, 0.1]):
        super().__init__()
        import numpy as np
        self.p_H = np.array(prior_probs)
        self.models = {
            'normal': {
                'mu_std': 0.01,
                'obs_std': 0.02
            },
            'noise_injection': {
                'mu_std': 0.5,
                'obs_std': 0.1
            },
            'bias_injection': {
                'mu_std': 0.01,
                'obs_std': 0.02,
                'step_gain': 2.0
            },
            'self_reference': {
                'mu_std': 0.01,
                'obs_std': 0.01,
                'trend_gain': -0.2
            }
        }

        self.cutoff_cooldown = 10
        self.cooldown_counter = 0
        self.decision_history: List[DecisionRecord] = []
        self.step_counter = 0

        # 连续截断信心机制
        self.consecutive_confidence = ConsecutiveCutoffConfidence()

    def forward(
        self,
        entropy_stats: Dict[str, float],
        van_event: bool = False,
        p_harm: float = 0.0,
        task_embedding: Optional[torch.Tensor] = None
    ) -> ControlSignals:
        """
        L3层前向传播 - 贝叶斯元控制决策

        Args:
            entropy_stats: 来自L2的熵统计字典
                - mean: μ_H 熵均值
                - variance: σ_H² 熵方差
                - trend: k_H 趋势
                - current: 当前熵值
            van_event: 是否触发VAN事件
            p_harm: VAN检测的有害概率
            task_embedding: 任务嵌入向量

        Returns:
            ControlSignals: 调控信号
        """
        import numpy as np
        self.step_counter += 1

        o_int = {
            'mu_H': entropy_stats.get("mean", 0.0),
            'sigma_H2': entropy_stats.get("variance", 0.0),
            'k_H': entropy_stats.get("trend", 0.0),
            'p_harm_raw': p_harm
        }

        likelihoods = []
        for condition in ['normal', 'noise_injection', 'bias_injection', 'self_reference']:
            model = self.models[condition]
            if condition == 'normal':
                lik = np.exp(-0.5 * (o_int['sigma_H2']/0.05)**2) * \
                      np.exp(-0.5 * (abs(o_int['k_H'])/0.1)**2)
            elif condition == 'noise_injection':
                lik = np.exp(-0.5 * ((o_int['sigma_H2']-0.2)/0.1)**2)
            elif condition == 'bias_injection':
                mu_lik = np.exp(-0.5 * ((o_int['mu_H']-0.2)/0.05)**2)
                harm_lik = np.exp(-0.5 * ((o_int['p_harm_raw']-0.8)/0.1)**2)
                trend_lik = np.exp(-0.5 * ((o_int['k_H']+0.1)/0.05)**2)
                lik = mu_lik * harm_lik * trend_lik
            elif condition == 'self_reference':
                # 自指循环的特征：低熵、低方差、负趋势
                mu_lik = np.exp(-0.5 * ((o_int['mu_H']-0.1)/0.05)**2)
                var_lik = np.exp(-0.5 * ((o_int['sigma_H2']-0.01)/0.01)**2)
                trend_lik = np.exp(-0.5 * ((o_int['k_H']+0.15)/0.05)**2)
                lik = mu_lik * var_lik * trend_lik
            else:
                lik = 1.0
            likelihoods.append(lik)

        unnorm = self.p_H * np.array(likelihoods)
        if unnorm.sum() > 0:
            self.p_H = unnorm / unnorm.sum()

        # 计算温度 - 基于病因后验概率
        tau_default = 1.0
        # 偏见注入时降低温度，噪声注入时提高温度
        tau = tau_default * (1.0 - 0.6 * self.p_H[2]) * (1.0 + 0.4 * self.p_H[1])
        # 自指循环时提高温度以打破循环
        tau = tau * (1.0 + 0.5 * self.p_H[3])
        tau = np.clip(tau, 0.2, 2.0)

        # 计算有害概率
        p_harm = self.p_H[2] * 0.6 + o_int['p_harm_raw'] * 0.4

        # 连续截断信心更新
        should_cutoff_initial = van_event or p_harm > 0.6 or o_int['p_harm_raw'] > 0.8
        confidence_result = self.consecutive_confidence.update(
            should_cutoff=should_cutoff_initial,
            entropy_stats=entropy_stats,
            van_event=van_event,
            p_harm_raw=p_harm
        )

        cutoff = False
        reason = None
        if van_event:
            cutoff = True
            reason = "VAN event: sensitive content detected"
            self.cooldown_counter = self.cutoff_cooldown
        elif confidence_result['should_trust_cutoff']:
            cutoff = True
            reason = f"High harm probability: {p_harm:.2f}, confidence: {confidence_result['confidence']:.2f}"
            self.cooldown_counter = self.cutoff_cooldown
        elif self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            reason = "Cooldown"
        else:
            # 检查是否需要否决截断
            override, override_reason = self.consecutive_confidence.should_override_cutoff()
            if override:
                reason = f"Override: {override_reason}"

        self._record_decision(
            entropy_mean=o_int['mu_H'],
            entropy_variance=o_int['sigma_H2'],
            van_event=van_event,
            p_harm=p_harm,
            cutoff=cutoff,
            reason=reason
        )

        # 动态调整稀疏阈值和DMN系数
        theta = 0.7
        alpha = 0.1
        
        # 偏见注入时增加稀疏度
        if self.p_H[2] > 0.5:
            theta = min(0.9, theta + 0.2)
            alpha = min(0.5, alpha + 0.3)
        # 自指循环时降低稀疏度
        elif self.p_H[3] > 0.5:
            theta = max(0.5, theta - 0.2)
            alpha = min(0.3, alpha + 0.2)

        return ControlSignals(
            tau=float(tau),
            theta=theta,
            alpha=alpha,
            stability=not cutoff,
            cutoff=cutoff,
            reason=reason
        )

    def _record_decision(
        self,
        entropy_mean: float,
        entropy_variance: float,
        van_event: bool,
        p_harm: float,
        cutoff: bool,
        reason: Optional[str]
    ) -> None:
        """
        记录决策历史

        Args:
            entropy_mean: 熵均值
            entropy_variance: 熵方差
            van_event: VAN事件标志
            p_harm: 有害概率
            cutoff: 是否截断
            reason: 决策原因
        """
        record = DecisionRecord(
            step=self.step_counter,
            entropy_mean=entropy_mean,
            entropy_variance=entropy_variance,
            van_event=van_event,
            p_harm=p_harm,
            cutoff=cutoff,
            cooldown_remaining=self.cooldown_counter,
            reason=reason
        )
        self.decision_history.append(record)

        if len(self.decision_history) > 1000:
            self.decision_history.pop(0)

    def get_posterior(self) -> Dict[str, float]:
        """
        获取当前病因后验概率

        Returns:
            病因后验概率字典
        """
        return {
            'normal': float(self.p_H[0]),
            'noise_injection': float(self.p_H[1]),
            'bias_injection': float(self.p_H[2]),
            'self_reference': float(self.p_H[3])
        }

    def reset(self) -> None:
        """
        重置贝叶斯L3控制器状态
        """
        import numpy as np
        self.p_H = np.array([0.7, 0.1, 0.1, 0.1])  # 增加自指循环的先验概率
        self.cooldown_counter = 0
        self.decision_history = []
        self.step_counter = 0
        self.consecutive_confidence.reset()

    def get_history(self, last_n: Optional[int] = None) -> List[DecisionRecord]:
        """
        获取决策历史

        Args:
            last_n: 返回最近的N条记录，None表示全部

        Returns:
            决策记录列表
        """
        if last_n is None:
            return self.decision_history.copy()
        return self.decision_history[-last_n:]

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取决策统计信息

        Returns:
            统计信息字典
        """
        if not self.decision_history:
            return {
                "total_decisions": 0,
                "posterior": self.get_posterior()
            }

        cutoffs = [r for r in self.decision_history if r.cutoff]
        van_events = [r for r in self.decision_history if r.van_event]
        cooldowns = [r for r in self.decision_history if r.reason == "Cooldown"]

        return {
            "total_decisions": len(self.decision_history),
            "total_cutoffs": len(cutoffs),
            "total_van_events": len(van_events),
            "total_cooldowns": len(cooldowns),
            "cooldown_counter": self.cooldown_counter,
            "cutoff_ratio": len(cutoffs) / len(self.decision_history) if self.decision_history else 0,
            "posterior": self.get_posterior(),
            "confidence_statistics": self.consecutive_confidence.get_statistics()
        }


class SceneType:
    """场景类型枚举"""
    CREATIVE_WRITING = "creative_writing"
    CODE_GENERATION = "code_generation"
    QUESTION_ANSWERING = "question_answering"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    GENERAL = "general"


class TemperatureConfig:
    """温度配置"""
    def __init__(
        self,
        creative_range: Tuple[float, float] = (0.7, 1.2),
        code_range: Tuple[float, float] = (0.2, 0.5),
        qa_range: Tuple[float, float] = (0.3, 0.6),
        summary_range: Tuple[float, float] = (0.4, 0.7),
        translation_range: Tuple[float, float] = (0.3, 0.6),
        general_range: Tuple[float, float] = (0.5, 0.8),
        default_temperature: float = 0.7,
        smoothing_factor: float = 0.3,
        stability_threshold: float = 0.15
    ):
        self.creative_range = creative_range
        self.code_range = code_range
        self.qa_range = qa_range
        self.summary_range = summary_range
        self.translation_range = translation_range
        self.general_range = general_range
        self.default_temperature = default_temperature
        self.smoothing_factor = smoothing_factor
        self.stability_threshold = stability_threshold

    def get_range(self, scene_type: str) -> Tuple[float, float]:
        """获取场景对应的温度范围"""
        ranges = {
            SceneType.CREATIVE_WRITING: self.creative_range,
            SceneType.CODE_GENERATION: self.code_range,
            SceneType.QUESTION_ANSWERING: self.qa_range,
            SceneType.SUMMARIZATION: self.summary_range,
            SceneType.TRANSLATION: self.translation_range,
            SceneType.GENERAL: self.general_range
        }
        return ranges.get(scene_type, self.general_range)


class OutputStabilityMonitor:
    """输出稳定性监控器"""

    def __init__(
        self,
        window_size: int = 10,
        diversity_threshold: float = 0.3,
        repetition_threshold: float = 0.5,
        stability_threshold: float = 0.15
    ):
        self.window_size = window_size
        self.diversity_threshold = diversity_threshold
        self.repetition_threshold = repetition_threshold
        self.stability_threshold = stability_threshold

        self.output_history: List[str] = []
        self.temperature_history: List[float] = []
        self.diversity_history: List[float] = []
        self.repetition_history: List[float] = []

    def update(
        self,
        output: str,
        temperature: float
    ) -> Dict[str, float]:
        """
        更新监控状态

        Returns:
            包含稳定性指标的字典
        """
        self.output_history.append(output)
        self.temperature_history.append(temperature)

        if len(self.output_history) > self.window_size:
            self.output_history.pop(0)
            self.temperature_history.pop(0)

        diversity = self._compute_diversity(output)
        repetition = self._compute_repetition(output)

        self.diversity_history.append(diversity)
        self.repetition_history.append(repetition)

        if len(self.diversity_history) > self.window_size:
            self.diversity_history.pop(0)
            self.repetition_history.pop(0)

        return self._compute_stability_metrics()

    def _compute_diversity(self, text: str) -> float:
        """计算词汇多样性"""
        if not text or len(text.strip()) == 0:
            return 0.0

        tokens = text.split()
        if len(tokens) < 2:
            return 1.0

        unique_tokens = len(set(tokens))
        return unique_tokens / len(tokens)

    def _compute_repetition(self, text: str) -> float:
        """计算重复率"""
        if not text or len(text.strip()) == 0:
            return 0.0

        tokens = text.split()
        if len(tokens) < 2:
            return 0.0

        bigrams = [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)]
        if not bigrams:
            return 0.0

        unique_bigrams = len(set(bigrams))
        return 1.0 - (unique_bigrams / len(bigrams))

    def _compute_stability_metrics(self) -> Dict[str, float]:
        """计算稳定性指标"""
        if len(self.diversity_history) < 2:
            return {
                "stability_score": 1.0,
                "avg_diversity": 0.5,
                "avg_repetition": 0.0,
                "diversity_trend": 0.0,
                "repetition_trend": 0.0,
                "is_stable": True
            }

        avg_diversity = np.mean(self.diversity_history)
        avg_repetition = np.mean(self.repetition_history)

        diversity_trend = 0.0
        if len(self.diversity_history) >= 3:
            x = np.arange(len(self.diversity_history))
            slope, _ = np.polyfit(x, self.diversity_history, 1)
            diversity_trend = float(slope)

        repetition_trend = 0.0
        if len(self.repetition_history) >= 3:
            x = np.arange(len(self.repetition_history))
            slope, _ = np.polyfit(x, self.repetition_history, 1)
            repetition_trend = float(slope)

        diversity_stable = avg_diversity > self.diversity_threshold
        repetition_stable = avg_repetition < self.repetition_threshold
        trend_stable = abs(diversity_trend) < 0.05 and abs(repetition_trend) < 0.05

        stability_score = (
            0.4 * (1.0 if diversity_stable else 0.0) +
            0.3 * (1.0 if repetition_stable else 0.0) +
            0.3 * (1.0 if trend_stable else 0.0)
        )

        return {
            "stability_score": stability_score,
            "avg_diversity": avg_diversity,
            "avg_repetition": avg_repetition,
            "diversity_trend": diversity_trend,
            "repetition_trend": repetition_trend,
            "is_stable": stability_score > (1.0 - self.stability_threshold)
        }

    def should_adjust_temperature(self) -> bool:
        """判断是否需要调整温度以恢复稳定性"""
        if len(self.output_history) < 3:
            return False

        metrics = self._compute_stability_metrics()

        if not metrics["is_stable"]:
            if metrics["avg_diversity"] < self.diversity_threshold:
                return True
            if metrics["avg_repetition"] > self.repetition_threshold:
                return True

        return False

    def get_recommended_adjustment(self) -> float:
        """获取温度调整建议"""
        if len(self.temperature_history) < 2:
            return 0.0

        metrics = self._compute_stability_metrics()
        recent_temp = self.temperature_history[-1]

        if metrics["avg_diversity"] < self.diversity_threshold:
            return 0.1
        if metrics["avg_repetition"] > self.repetition_threshold:
            return -0.1
        if metrics["diversity_trend"] < -0.02:
            return 0.05

        return 0.0

    def reset(self) -> None:
        """重置监控状态"""
        self.output_history.clear()
        self.temperature_history.clear()
        self.diversity_history.clear()
        self.repetition_history.clear()


class ContextualTemperatureController:
    """
    基于上下文的动态温度调节器

    功能:
    1. 场景识别 - 根据输入/输出特征识别场景类型
    2. 平滑过渡 - 使用EMA平滑温度变化，避免剧烈跳跃
    3. 稳定性监控 - 监控温度变化对输出稳定性的影响
    4. 自适应调节 - 根据熵状态和稳定性指标动态调整温度

    设计原则:
    - 平衡稳定性与创造性
    - 温度变化应该渐进而非突变
    - 根据不同场景采用差异化策略
    """

    SCENE_KEYWORDS = {
        SceneType.CREATIVE_WRITING: [
            "写", "故事", "小说", "创作", "编写", "续写", "诗歌", "散文", "fiction", "novel",
            "creative", "write", "imagine", "设想", "假如", "如果", "scenario"
        ],
        SceneType.CODE_GENERATION: [
            "代码", "程序", "函数", "实现", "bug", "调试", "算法", "def ", "class ",
            "code", "function", "implement", "debug", "algorithm", "return", "import"
        ],
        SceneType.QUESTION_ANSWERING: [
            "什么", "为什么", "如何", "怎样", "多少", "谁", "解释", "回答",
            "what", "why", "how", "explain", "define", "describe"
        ],
        SceneType.SUMMARIZATION: [
            "总结", "概括", "摘要", "精简", "要点", "简述", "概述",
            "summarize", "summary", "brief", "condense", "key points"
        ],
        SceneType.TRANSLATION: [
            "翻译", "convert", "translate",
            "翻译成", "翻译为"
        ]
    }

    def __init__(
        self,
        config: Optional[TemperatureConfig] = None,
        tau_range: Tuple[float, float] = (0.1, 2.0)
    ):
        self.config = config or TemperatureConfig()
        self.tau_range = tau_range

        self.current_scene = SceneType.GENERAL
        self.current_temperature = self.config.default_temperature
        self.target_temperature = self.config.default_temperature
        self.smoothed_temperature = self.config.default_temperature

        self.stability_monitor = OutputStabilityMonitor()

        self.temperature_history: List[float] = []
        self.scene_history: List[str] = []
        self.entropy_history: List[float] = []

        self.decision_history: List[Dict] = []

    def detect_scene(self, prompt: str, context: Optional[str] = None) -> str:
        """
        检测场景类型

        Args:
            prompt: 输入提示
            context: 可选的上下文信息

        Returns:
            场景类型字符串
        """
        combined_text = f"{prompt} {context or ''}".lower()

        scores = {}
        for scene_type, keywords in self.SCENE_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw.lower() in combined_text)
            scores[scene_type] = score

        if max(scores.values()) > 0:
            detected = max(scores, key=scores.get)
            self.current_scene = detected
            self.scene_history.append(detected)
            if len(self.scene_history) > 100:
                self.scene_history.pop(0)
            return detected

        self.current_scene = SceneType.GENERAL
        return SceneType.GENERAL

    def compute_contextual_temperature(
        self,
        entropy_stats: Dict[str, float],
        van_event: bool = False,
        p_harm: float = 0.0,
        scene_type: Optional[str] = None
    ) -> float:
        """
        根据上下文计算目标温度

        Args:
            entropy_stats: 熵统计信息
            van_event: 是否触发VAN事件
            p_harm: 有害概率
            scene_type: 场景类型

        Returns:
            计算后的目标温度
        """
        scene = scene_type or self.current_scene
        range_min, range_max = self.config.get_range(scene)

        mu_h = entropy_stats.get("mean", 0.5)
        sigma_h = entropy_stats.get("variance", 0.02) ** 0.5
        k_h = entropy_stats.get("trend", 0.0)

        base_temp = (range_min + range_max) / 2.0

        if van_event or p_harm > 0.7:
            target_temp = range_min * 0.5
        elif p_harm > 0.4:
            target_temp = range_min
        elif mu_h < 0.3 and sigma_h < 0.05 and k_h < 0:
            target_temp = range_max * 0.8
        elif mu_h > 0.7 and sigma_h > 0.1:
            target_temp = range_min * 1.2
        elif k_h > 0.1:
            target_temp = min(base_temp * 1.1, range_max)
        elif k_h < -0.1:
            target_temp = max(base_temp * 0.9, range_min)
        else:
            target_temp = base_temp

        self.target_temperature = np.clip(target_temp, range_min, range_max)
        return self.target_temperature

    def smooth_temperature(
        self,
        target_temp: Optional[float] = None,
        smoothing_factor: Optional[float] = None
    ) -> float:
        """
        使用EMA平滑温度过渡

        Args:
            target_temp: 目标温度，如果为None则使用当前target_temperature
            smoothing_factor: 平滑因子，如果为None则使用配置的因子

        Returns:
            平滑后的温度
        """
        target = target_temp if target_temp is not None else self.target_temperature
        alpha = smoothing_factor if smoothing_factor is not None else self.config.smoothing_factor

        prev_smoothed = self.smoothed_temperature
        self.smoothed_temperature = alpha * target + (1 - alpha) * prev_smoothed

        self.smoothed_temperature = np.clip(
            self.smoothed_temperature,
            self.tau_range[0],
            self.tau_range[1]
        )

        return self.smoothed_temperature

    def adjust_for_stability(
        self,
        stability_metrics: Dict[str, float],
        current_temp: float
    ) -> float:
        """
        根据稳定性指标调整温度

        Args:
            stability_metrics: 稳定性指标
            current_temp: 当前温度

        Returns:
            调整后的温度
        """
        if stability_metrics["is_stable"]:
            return current_temp

        adjustment = 0.0

        if stability_metrics["avg_diversity"] < 0.3:
            adjustment += 0.1
        elif stability_metrics["avg_diversity"] > 0.7:
            adjustment -= 0.05

        if stability_metrics["avg_repetition"] > 0.4:
            adjustment += 0.1
        elif stability_metrics["avg_repetition"] < 0.2:
            adjustment -= 0.05

        if stability_metrics["diversity_trend"] < -0.03:
            adjustment += 0.08
        elif stability_metrics["diversity_trend"] > 0.03:
            adjustment -= 0.05

        adjusted_temp = current_temp + adjustment
        return float(np.clip(adjusted_temp, self.tau_range[0], self.tau_range[1]))

    def compute_temperature(
        self,
        entropy_stats: Dict[str, float],
        van_event: bool = False,
        p_harm: float = 0.0,
        output: Optional[str] = None,
        scene_type: Optional[str] = None,
        enable_stability_monitor: bool = True
    ) -> Dict[str, Any]:
        """
        综合计算温度

        流程:
        1. 检测场景类型
        2. 根据熵状态计算目标温度
        3. 平滑温度过渡
        4. 根据稳定性监控调整温度

        Args:
            entropy_stats: 熵统计信息
            van_event: 是否触发VAN事件
            p_harm: 有害概率
            output: 实际输出（用于稳定性监控）
            scene_type: 场景类型
            enable_stability_monitor: 是否启用稳定性监控

        Returns:
            包含温度信息和统计的字典
        """
        if scene_type:
            self.current_scene = scene_type

        self.compute_contextual_temperature(
            entropy_stats=entropy_stats,
            van_event=van_event,
            p_harm=p_harm,
            scene_type=scene_type
        )

        temp_before_smoothing = self.smoothed_temperature
        temp_after_smoothing = self.smooth_temperature()

        stability_metrics = {"is_stable": True, "stability_score": 1.0}
        stability_adjusted_temp = temp_after_smoothing

        if enable_stability_monitor and output:
            stability_metrics = self.stability_monitor.update(output, temp_after_smoothing)

            if self.stability_monitor.should_adjust_temperature():
                stability_adjusted_temp = self.adjust_for_stability(
                    stability_metrics,
                    temp_after_smoothing
                )

        final_temp = stability_adjusted_temp

        self.current_temperature = final_temp
        self.temperature_history.append(final_temp)

        mu_h = entropy_stats.get("mean", 0.0)
        self.entropy_history.append(mu_h)

        if len(self.temperature_history) > 1000:
            self.temperature_history.pop(0)
        if len(self.entropy_history) > 1000:
            self.entropy_history.pop(0)

        record = {
            "scene": self.current_scene,
            "target_temp": self.target_temperature,
            "temp_before_smoothing": temp_before_smoothing,
            "temp_after_smoothing": temp_after_smoothing,
            "stability_adjusted_temp": stability_adjusted_temp,
            "final_temp": final_temp,
            "entropy_mean": mu_h,
            "van_event": van_event,
            "p_harm": p_harm,
            "stability_metrics": stability_metrics
        }
        self.decision_history.append(record)

        if len(self.decision_history) > 1000:
            self.decision_history.pop(0)

        return {
            "temperature": final_temp,
            "scene": self.current_scene,
            "target_temperature": self.target_temperature,
            "smoothed_temperature": temp_after_smoothing,
            "stability_metrics": stability_metrics,
            "is_stable": stability_metrics["is_stable"]
        }

    def get_temperature_for_api(
        self,
        entropy_stats: Dict[str, float],
        van_event: bool = False,
        p_harm: float = 0.0,
        output: Optional[str] = None,
        scene_type: Optional[str] = None
    ) -> float:
        """
        获取适合API调用的温度参数

        Returns:
            温度值
        """
        result = self.compute_temperature(
            entropy_stats=entropy_stats,
            van_event=van_event,
            p_harm=p_harm,
            output=output,
            scene_type=scene_type
        )
        return result["temperature"]

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self.decision_history:
            return {
                "total_decisions": 0,
                "current_scene": self.current_scene,
                "current_temperature": self.current_temperature
            }

        temp_values = [d["final_temp"] for d in self.decision_history]
        scene_counts = {}
        for record in self.decision_history:
            scene = record["scene"]
            scene_counts[scene] = scene_counts.get(scene, 0) + 1

        stability_scores = [
            d["stability_metrics"]["stability_score"]
            for d in self.decision_history
            if "stability_score" in d.get("stability_metrics", {})
        ]

        return {
            "total_decisions": len(self.decision_history),
            "current_scene": self.current_scene,
            "current_temperature": self.current_temperature,
            "temperature_stats": {
                "mean": float(np.mean(temp_values)),
                "std": float(np.std(temp_values)),
                "min": float(np.min(temp_values)),
                "max": float(np.max(temp_values)),
                "recent_10": temp_values[-10:] if len(temp_values) >= 10 else temp_values
            },
            "scene_distribution": scene_counts,
            "stability_stats": {
                "mean_score": float(np.mean(stability_scores)) if stability_scores else 1.0,
                "stable_ratio": sum(1 for s in stability_scores if s > 0.8) / len(stability_scores) if stability_scores else 1.0
            }
        }

    def reset(self) -> None:
        """重置控制器状态"""
        self.current_scene = SceneType.GENERAL
        self.current_temperature = self.config.default_temperature
        self.target_temperature = self.config.default_temperature
        self.smoothed_temperature = self.config.default_temperature

        self.stability_monitor.reset()

        self.temperature_history.clear()
        self.scene_history.clear()
        self.entropy_history.clear()
        self.decision_history.clear()

    def get_scene_distribution(self) -> Dict[str, int]:
        """获取场景分布统计"""
        scene_counts = {}
        for scene in self.scene_history:
            scene_counts[scene] = scene_counts.get(scene, 0) + 1
        return scene_counts

    def get_recent_decisions(self, n: int = 10) -> List[Dict]:
        """获取最近的N个决策"""
        return self.decision_history[-n:]


class ConsecutiveCutoffConfidence:
    """
    连续截断信心机制

    核心功能：
    1. 追踪截断历史和连续截断长度
    2. 计算带衰减的连续截断信心
    3. 防止过度截断的保障机制

    信心计算原则：
    - 初始信心基于熵统计和VAN事件
    - 连续截断时信心累积（但有衰减）
    - 连续截断越长，信心越趋向保守
    - 提供信心阈值动态调整

    衰减机制：
    - 连续截断信心 = base_confidence * decay_factor^consecutive_count
    - decay_factor ∈ [0.5, 1.0]，避免信心无限累积
    """

    def __init__(
        self,
        initial_confidence: float = 0.5,
        decay_factor: float = 0.85,
        confidence_boost_per_cutoff: float = 0.15,
        max_consecutive_cutoffs: int = 5,
        over_cutoff_threshold: float = 0.75,
        confidence_threshold_increase: float = 0.1,
        min_confidence_threshold: float = 0.5,
        max_confidence_threshold: float = 0.9
    ):
        """
        初始化连续截断信心机制

        Args:
            initial_confidence: 初始信心值
            decay_factor: 衰减因子，每增加一次连续截断，信心增长幅度衰减
            confidence_boost_per_cutoff: 每次截断的信心增量
            max_consecutive_cutoffs: 最大允许连续截断次数
            over_cutoff_threshold: 超过此阈值认为进入"过度截断"风险
            confidence_threshold_increase: 过度截断时阈值增量
            min_confidence_threshold: 最小信心阈值
            max_confidence_threshold: 最大信心阈值
        """
        self.initial_confidence = initial_confidence
        self.decay_factor = decay_factor
        self.confidence_boost_per_cutoff = confidence_boost_per_cutoff
        self.max_consecutive_cutoffs = max_consecutive_cutoffs
        self.over_cutoff_threshold = over_cutoff_threshold
        self.confidence_threshold_increase = confidence_threshold_increase
        self.min_confidence_threshold = min_confidence_threshold
        self.max_confidence_threshold = max_confidence_threshold

        self.cutoff_history: List[bool] = []
        self.confidence_history: List[float] = []
        self.consecutive_cutoff_count = 0
        self.current_confidence = initial_confidence

        self._base_threshold = 0.6

    def update(
        self,
        should_cutoff: bool,
        entropy_stats: Optional[Dict[str, float]] = None,
        van_event: bool = False,
        p_harm_raw: float = 0.0
    ) -> Dict[str, float]:
        """
        更新连续截断信心状态

        Args:
            should_cutoff: 当前帧是否判定为应该截断
            entropy_stats: 熵统计字典
            van_event: 是否触发VAN事件
            p_harm_raw: 原始有害概率

        Returns:
            包含信心信息和统计的字典
        """
        self.cutoff_history.append(should_cutoff)

        if len(self.cutoff_history) > self.max_consecutive_cutoffs * 3:
            self.cutoff_history.pop(0)

        if should_cutoff:
            self.consecutive_cutoff_count += 1
        else:
            self.consecutive_cutoff_count = 0

        raw_confidence = self._compute_raw_confidence(
            entropy_stats=entropy_stats,
            van_event=van_event,
            p_harm_raw=p_harm_raw
        )

        self.current_confidence = self._compute_consecutive_confidence(
            raw_confidence=raw_confidence
        )

        self.confidence_history.append(self.current_confidence)
        if len(self.confidence_history) > 100:
            self.confidence_history.pop(0)

        dynamic_threshold = self._compute_dynamic_threshold()

        is_over_cutoff_risk = (
            self.consecutive_cutoff_count >= self.max_consecutive_cutoffs
        )

        return {
            'confidence': self.current_confidence,
            'raw_confidence': raw_confidence,
            'consecutive_cutoff_count': self.consecutive_cutoff_count,
            'dynamic_threshold': dynamic_threshold,
            'is_over_cutoff_risk': is_over_cutoff_risk,
            'should_trust_cutoff': self.current_confidence >= dynamic_threshold
        }

    def _compute_raw_confidence(
        self,
        entropy_stats: Optional[Dict[str, float]],
        van_event: bool,
        p_harm_raw: float
    ) -> float:
        """
        计算原始信心值

        基于：
        1. 熵统计 (mu_H, sigma_H2, k_H)
        2. VAN事件
        3. 原始有害概率

        Returns:
            原始信心值 ∈ [0, 1]
        """
        if entropy_stats is None:
            entropy_stats = {'mean': 0.5, 'variance': 0.02, 'trend': 0.0}

        mu_h = entropy_stats.get('mean', 0.5)
        sigma_h2 = entropy_stats.get('variance', 0.02)
        k_h = entropy_stats.get('trend', 0.0)

        if van_event:
            return min(1.0, p_harm_raw + 0.3)

        low_entropy_factor = 0.0
        if mu_h < 0.3:
            low_entropy_factor = (0.3 - mu_h) / 0.3
        elif mu_h > 0.7:
            low_entropy_factor = -(mu_h - 0.7) / 0.3

        low_variance_factor = 0.0
        if sigma_h2 < 0.05:
            low_variance_factor = (0.05 - sigma_h2) / 0.05

        negative_trend_factor = 0.0
        if k_h < -0.05:
            negative_trend_factor = min(1.0, abs(k_h) / 0.1)

        base_confidence = (
            0.3 * p_harm_raw +
            0.25 * low_entropy_factor +
            0.25 * low_variance_factor +
            0.2 * negative_trend_factor
        )

        return float(np.clip(base_confidence, 0.0, 1.0))

    def _compute_consecutive_confidence(self, raw_confidence: float) -> float:
        """
        计算带连续截断衰减的信心值

        公式：consecutive_confidence = raw_confidence + boost * decay^count

        其中：
        - boost = confidence_boost_per_cutoff
        - decay = decay_factor
        - count = consecutive_cutoff_count

        衰减机制确保：
        - 第1次截断：boost * 1.0
        - 第2次截断：boost * 0.85
        - 第3次截断：boost * 0.7225
        - 依次类推...

        Args:
            raw_confidence: 原始信心值

        Returns:
            连续截断衰减后的信心值
        """
        if self.consecutive_cutoff_count == 0:
            return raw_confidence

        decay_boost = (
            self.confidence_boost_per_cutoff *
            (self.decay_factor ** self.consecutive_cutoff_count) *
            min(self.consecutive_cutoff_count, self.max_consecutive_cutoffs)
        )

        consecutive_confidence = raw_confidence + decay_boost

        if self.consecutive_cutoff_count >= self.max_consecutive_cutoffs:
            over_cutoff_penalty = (
                (self.consecutive_cutoff_count - self.max_consecutive_cutoffs + 1) *
                0.1
            )
            consecutive_confidence -= over_cutoff_penalty

        return float(np.clip(consecutive_confidence, 0.0, 1.0))

    def _compute_dynamic_threshold(self) -> float:
        """
        计算动态信心阈值

        当连续截断过长时，提高阈值以防止过度截断

        Returns:
            动态信心阈值
        """
        base_threshold = self._base_threshold

        if self.consecutive_cutoff_count >= self.max_consecutive_cutoffs:
            increase = (
                (self.consecutive_cutoff_count - self.max_consecutive_cutoffs + 1) *
                self.confidence_threshold_increase
            )
            base_threshold += increase

        if len(self.confidence_history) >= 10:
            recent_confidence = np.mean(self.confidence_history[-10:])
            if recent_confidence > 0.7:
                stability_bonus = (recent_confidence - 0.7) * 0.2
                base_threshold += stability_bonus

        return float(np.clip(
            base_threshold,
            self.min_confidence_threshold,
            self.max_confidence_threshold
        ))

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取信心机制统计信息

        Returns:
            统计信息字典
        """
        total_cutoffs = sum(1 for c in self.cutoff_history if c)
        total_frames = len(self.cutoff_history) if self.cutoff_history else 1

        recent_10_cutoffs = self.cutoff_history[-10:] if len(self.cutoff_history) >= 10 else self.cutoff_history
        recent_cutoff_ratio = (
            sum(1 for c in recent_10_cutoffs if c) / len(recent_10_cutoffs)
            if recent_10_cutoffs else 0.0
        )

        return {
            'current_confidence': self.current_confidence,
            'consecutive_cutoff_count': self.consecutive_cutoff_count,
            'total_cutoffs': total_cutoffs,
            'cutoff_ratio': total_cutoffs / total_frames,
            'recent_cutoff_ratio': recent_cutoff_ratio,
            'dynamic_threshold': self._compute_dynamic_threshold(),
            'is_over_cutoff_risk': self.consecutive_cutoff_count >= self.max_consecutive_cutoffs,
            'history_length': len(self.cutoff_history)
        }

    def should_override_cutoff(self) -> Tuple[bool, Optional[str]]:
        """
        判断是否应该否决当前截断决策（防止过度截断）

        条件：
        1. 连续截断超过最大限制
        2. 当前信心低于动态阈值

        Returns:
            (是否否决, 原因)
        """
        if self.consecutive_cutoff_count >= self.max_consecutive_cutoffs:
            dynamic_threshold = self._compute_dynamic_threshold()

            if self.current_confidence < dynamic_threshold:
                return True, (
                    f"Over-cutoff protection: consecutive={self.consecutive_cutoff_count}, "
                    f"confidence={self.current_confidence:.2f} < threshold={dynamic_threshold:.2f}"
                )

        if len(self.confidence_history) >= 5:
            recent = self.confidence_history[-5:]
            if all(c < 0.4 for c in recent):
                return True, "Low confidence trend: all recent confidence < 0.4"

        return False, None

    def reset(self) -> None:
        """重置信心机制状态"""
        self.cutoff_history.clear()
        self.confidence_history.clear()
        self.consecutive_cutoff_count = 0
        self.current_confidence = self.initial_confidence


class EnhancedBayesianL3Controller(nn.Module):
    """
    增强版贝叶斯L3元控制器 - 改进的病因推断算法

    改进点:
    1. 时序贝叶斯推断 - 递归估计 + 历史信息利用
    2. 鲁棒似然模型 - 使用t分布替代高斯分布
    3. 自适应先验 - 基于历史决策更新
    4. 混合病因检测 - 检测组合异常模式
    5. 动态阈值调整 - 根据后验分布自适应
    6. 连续截断信心机制 - 防止过度截断

    病因假设 H = {H_normal, H_noise, H_bias, H_mixed}:
    - H_normal: 正常状态
    - H_noise: 噪声注入攻击
    - H_bias: 偏见/自指循环攻击
    - H_mixed: 混合型攻击（多种病因同时出现）
    """

    CAUSE_NAMES = ['normal', 'noise_injection', 'bias_injection', 'mixed']

    def __init__(
        self,
        prior_probs: List[float] = [0.7, 0.1, 0.1, 0.1],
        adaptive_learning_rate: float = 0.1,
        temporal_window: int = 5,
        robust_nu: float = 4.0,
        cutoff_cooldown: int = 10,
        use_consecutive_confidence: bool = True,
        consecutive_confidence_config: Optional[Dict] = None
    ):
        """
        初始化增强版贝叶斯L3控制器

        Args:
            prior_probs: 初始先验概率 [P(H_normal), P(H_noise), P(H_bias), P(H_mixed)]
            adaptive_learning_rate: 自适应学习率，用于先验更新
            temporal_window: 时序窗口大小，用于检测缓慢变化
            robust_nu: t分布的自由度参数，越小越鲁棒
            cutoff_cooldown: 截断后冷却步数
            use_consecutive_confidence: 是否启用连续截断信心机制
            consecutive_confidence_config: 连续截断信心机制配置
        """
        super().__init__()
        import numpy as np

        self.p_H = np.array(prior_probs)
        self.adaptive_learning_rate = adaptive_learning_rate
        self.temporal_window = temporal_window
        self.robust_nu = robust_nu
        self.cutoff_cooldown = cutoff_cooldown
        self.use_consecutive_confidence = use_consecutive_confidence

        consecutive_config = consecutive_confidence_config or {}
        self.consecutive_confidence = ConsecutiveCutoffConfidence(
            initial_confidence=consecutive_config.get('initial_confidence', 0.5),
            decay_factor=consecutive_config.get('decay_factor', 0.85),
            confidence_boost_per_cutoff=consecutive_config.get('confidence_boost_per_cutoff', 0.15),
            max_consecutive_cutoffs=consecutive_config.get('max_consecutive_cutoffs', 5),
            over_cutoff_threshold=consecutive_config.get('over_cutoff_threshold', 0.75),
            confidence_threshold_increase=consecutive_config.get('confidence_threshold_increase', 0.1),
            min_confidence_threshold=consecutive_config.get('min_confidence_threshold', 0.5),
            max_confidence_threshold=consecutive_config.get('max_confidence_threshold', 0.9)
        )

        self.temporal_history: List[Dict[str, float]] = []
        self.posterior_history: List[Dict[str, float]] = []

        self.cutoff_cooldown = 10
        self.cooldown_counter = 0
        self.decision_history: List[DecisionRecord] = []
        self.step_counter = 0

        self.causal_model_params = {
            'normal': {
                'mu_H_mean': 0.5,
                'mu_H_std': 0.15,
                'sigma_H2_mean': 0.02,
                'sigma_H2_std': 0.02,
                'k_H_mean': 0.0,
                'k_H_std': 0.05,
                'p_harm_mean': 0.0,
                'p_harm_std': 0.1
            },
            'noise_injection': {
                'mu_H_mean': 0.45,
                'mu_H_std': 0.2,
                'sigma_H2_mean': 0.25,
                'sigma_H2_std': 0.15,
                'k_H_mean': 0.0,
                'k_H_std': 0.1,
                'p_harm_mean': 0.2,
                'p_harm_std': 0.15
            },
            'bias_injection': {
                'mu_H_mean': 0.2,
                'mu_H_std': 0.1,
                'sigma_H2_mean': 0.03,
                'sigma_H2_std': 0.02,
                'k_H_mean': -0.05,
                'k_H_std': 0.03,
                'p_harm_mean': 0.7,
                'p_harm_std': 0.2
            },
            'mixed': {
                'mu_H_mean': 0.3,
                'mu_H_std': 0.15,
                'sigma_H2_mean': 0.15,
                'sigma_H2_std': 0.12,
                'k_H_mean': -0.03,
                'k_H_std': 0.04,
                'p_harm_mean': 0.5,
                'p_harm_std': 0.2
            }
        }

    def _robust_t_likelihood(self, x: float, mu: float, sigma: float, nu: float) -> float:
        """
        计算t分布的鲁棒似然值

        t分布比高斯分布对异常值更鲁棒，尾部更重

        Args:
            x: 观测值
            mu: 均值
            sigma: 尺度参数
            nu: 自由度

        Returns:
            似然值
        """
        import numpy as np
        from scipy.special import gamma as gamma_fn

        if sigma <= 0:
            sigma = 1e-6

        d = (x - mu) / sigma
        coefficient = gamma_fn((nu + 1) / 2) / (sigma * np.sqrt(nu * np.pi) * gamma_fn(nu / 2))
        likelihood = coefficient * (1 + d**2 / nu) ** (-(nu + 1) / 2)

        return max(likelihood, 1e-300)

    def _compute_temporal_features(self) -> Dict[str, float]:
        """
        从时序历史中提取特征

        用于检测缓慢变化的自指循环

        Returns:
            时序特征字典
        """
        import numpy as np

        if len(self.temporal_history) < 2:
            return {
                'mu_H_trend': 0.0,
                'mu_H_acceleration': 0.0,
                'sigma_H2_trend': 0.0,
                'k_H_consistency': 0.0,
                'temporal_stability': 1.0
            }

        recent = self.temporal_history[-self.temporal_window:]

        mu_H_values = [h.get('mu_H', 0.5) for h in recent]
        sigma_H2_values = [h.get('sigma_H2', 0.02) for h in recent]
        k_H_values = [h.get('k_H', 0.0) for h in recent]

        x = np.arange(len(mu_H_values))

        if len(x) >= 2:
            mu_H_slope, _ = np.polyfit(x, mu_H_values, 1)
            mu_H_accel = np.diff(mu_H_values).mean() if len(mu_H_values) > 1 else 0.0
            sigma_H2_slope, _ = np.polyfit(x, sigma_H2_values, 1)
            k_H_std = np.std(k_H_values)
            temporal_stability = 1.0 - min(np.std(mu_H_values) / (np.mean(mu_H_values) + 1e-6), 1.0)
        else:
            mu_H_slope = 0.0
            mu_H_accel = 0.0
            sigma_H2_slope = 0.0
            k_H_std = 0.0
            temporal_stability = 1.0

        return {
            'mu_H_trend': float(mu_H_slope),
            'mu_H_acceleration': float(mu_H_accel),
            'sigma_H2_trend': float(sigma_H2_slope),
            'k_H_consistency': float(k_H_std),
            'temporal_stability': float(temporal_stability)
        }

    def _compute_likelihood_vector(self, obs: Dict[str, float]) -> np.ndarray:
        """
        为所有病因假设计算似然向量

        使用鲁棒t分布似然，考虑时序特征

        Args:
            obs: 观测向量

        Returns:
            似然向量
        """
        import numpy as np

        temporal_features = self._compute_temporal_features()

        likelihoods = np.zeros(4)

        for i, cause in enumerate(self.CAUSE_NAMES):
            params = self.causal_model_params[cause]

            if cause == 'mixed':
                lik = self._compute_mixed_likelihood(obs, params, temporal_features)
            else:
                lik = 1.0

                lik *= self._robust_t_likelihood(
                    obs['mu_H'],
                    params['mu_H_mean'],
                    params['mu_H_std'],
                    self.robust_nu
                )

                lik *= self._robust_t_likelihood(
                    obs['sigma_H2'],
                    params['sigma_H2_mean'],
                    params['sigma_H2_std'],
                    self.robust_nu
                )

                lik *= self._robust_t_likelihood(
                    obs['k_H'],
                    params['k_H_mean'],
                    params['k_H_std'],
                    self.robust_nu
                )

                lik *= self._robust_t_likelihood(
                    obs['p_harm_raw'],
                    params['p_harm_mean'],
                    params['p_harm_std'],
                    self.robust_nu
                )

                if temporal_features['temporal_stability'] < 0.3:
                    lik *= 0.3

            likelihoods[i] = lik

        likelihoods = np.maximum(likelihoods, 1e-300)

        return likelihoods

    def _compute_mixed_likelihood(
        self,
        obs: Dict[str, float],
        params: Dict[str, float],
        temporal_features: Dict[str, float]
    ) -> float:
        """
        计算混合病因的似然

        混合病因是噪声+偏见的组合，需要同时满足两者的部分条件

        Args:
            obs: 观测向量
            params: 因果模型参数
            temporal_features: 时序特征

        Returns:
            混合病因似然
        """
        import numpy as np

        noise_lik = self._robust_t_likelihood(
            obs['sigma_H2'],
            self.causal_model_params['noise_injection']['sigma_H2_mean'],
            self.causal_model_params['noise_injection']['sigma_H2_std'],
            self.robust_nu
        )

        bias_lik = self._robust_t_likelihood(
            obs['mu_H'],
            self.causal_model_params['bias_injection']['mu_H_mean'],
            self.causal_model_params['bias_injection']['mu_H_std'],
            self.robust_nu
        ) * self._robust_t_likelihood(
            obs['p_harm_raw'],
            self.causal_model_params['bias_injection']['p_harm_mean'],
            self.causal_model_params['bias_injection']['p_harm_std'],
            self.robust_nu
        )

        mixed_lik = np.sqrt(noise_lik * bias_lik)

        if temporal_features['temporal_stability'] < 0.5:
            mixed_lik *= (1.0 - temporal_features['temporal_stability'])

        return mixed_lik

    def _update_adaptive_prior(self, posterior: np.ndarray) -> None:
        """
        基于后验分布自适应更新先验

        使用指数加权移动平均

        Args:
            posterior: 后验分布
        """
        import numpy as np

        if len(self.posterior_history) > 0:
            prev_posterior = np.array([
                self.posterior_history[-1].get('normal', 0.7),
                self.posterior_history[-1].get('noise_injection', 0.1),
                self.posterior_history[-1].get('bias_injection', 0.1),
                self.posterior_history[-1].get('mixed', 0.1)
            ])

            self.p_H = (1 - self.adaptive_learning_rate) * prev_posterior + \
                       self.adaptive_learning_rate * posterior

            self.p_H = np.maximum(self.p_H, 0.01)
            self.p_H = self.p_H / self.p_H.sum()

    def _compute_dynamic_threshold(self, posterior: np.ndarray) -> float:
        """
        动态计算截断阈值

        根据后验分布的熵调整阈值：
        - 后验分布集中 → 降低阈值，更确定可以截断
        - 后验分布分散 → 提高阈值，更谨慎

        Args:
            posterior: 后验分布

        Returns:
            动态阈值
        """
        import numpy as np

        entropy = -np.sum(posterior * np.log(posterior + 1e-10))

        max_entropy = np.log(4)
        normalized_entropy = entropy / max_entropy

        base_threshold = 0.6
        threshold_adjustment = 0.2 * (normalized_entropy - 0.5)

        return float(np.clip(base_threshold + threshold_adjustment, 0.4, 0.8))

    def _detect_slow_drift(self, obs: Dict[str, float]) -> bool:
        """
        检测缓慢漂移的自指循环

        识别特征：
        - 熵均值持续下降
        - 方差保持低位
        - 危险概率逐渐上升

        Args:
            obs: 当前观测

        Returns:
            是否检测到缓慢漂移
        """
        import numpy as np

        if len(self.temporal_history) < 3:
            return False

        recent = self.temporal_history[-3:]

        mu_H_decreasing = all(
            recent[i]['mu_H'] > recent[i+1]['mu_H']
            for i in range(len(recent)-1)
        )

        sigma_H2_low = all(
            h.get('sigma_H2', 0.1) < 0.05
            for h in recent
        )

        p_harm_increasing = all(
            recent[i].get('p_harm_raw', 0) < recent[i+1].get('p_harm_raw', 0)
            for i in range(len(recent)-1)
        )

        return mu_H_decreasing and sigma_H2_low and p_harm_increasing

    def forward(
        self,
        entropy_stats: Dict[str, float],
        van_event: bool = False,
        p_harm: float = 0.0,
        task_embedding: Optional[torch.Tensor] = None
    ) -> ControlSignals:
        """
        L3层前向传播 - 增强贝叶斯元控制决策

        Args:
            entropy_stats: 来自L2的熵统计字典
                - mean: μ_H 熵均值
                - variance: σ_H² 熵方差
                - trend: k_H 趋势
                - current: 当前熵值
            van_event: 是否触发VAN事件
            p_harm: VAN检测的有害概率
            task_embedding: 任务嵌入向量

        Returns:
            ControlSignals: 调控信号
        """
        import numpy as np
        self.step_counter += 1

        obs = {
            'mu_H': entropy_stats.get("mean", 0.5),
            'sigma_H2': entropy_stats.get("variance", 0.02),
            'k_H': entropy_stats.get("trend", 0.0),
            'p_harm_raw': p_harm
        }

        self.temporal_history.append(obs.copy())
        if len(self.temporal_history) > self.temporal_window * 2:
            self.temporal_history.pop(0)

        likelihoods = self._compute_likelihood_vector(obs)

        unnorm_posterior = self.p_H * likelihoods
        posterior = unnorm_posterior / unnorm_posterior.sum()

        self.posterior_history.append({
            'normal': float(posterior[0]),
            'noise_injection': float(posterior[1]),
            'bias_injection': float(posterior[2]),
            'mixed': float(posterior[3])
        })
        if len(self.posterior_history) > 100:
            self.posterior_history.pop(0)

        self._update_adaptive_prior(posterior)

        tau = self._compute_temperature(posterior, obs)
        dynamic_threshold = self._compute_dynamic_threshold(posterior)

        p_harm_composite = self._compute_composite_harm_probability(posterior, obs)

        cutoff, reason = self._make_cutoff_decision(
            van_event=van_event,
            p_harm=p_harm_composite,
            p_harm_raw=p_harm,
            posterior=posterior,
            obs=obs,
            dynamic_threshold=dynamic_threshold
        )

        self._record_decision(
            entropy_mean=obs['mu_H'],
            entropy_variance=obs['sigma_H2'],
            van_event=van_event,
            p_harm=p_harm_composite,
            cutoff=cutoff,
            reason=reason
        )

        return ControlSignals(
            tau=float(tau),
            theta=0.7,
            alpha=0.1,
            stability=not cutoff,
            cutoff=cutoff,
            reason=reason
        )

    def _compute_temperature(
        self,
        posterior: np.ndarray,
        obs: Dict[str, float]
    ) -> float:
        """
        根据后验分布计算温度参数

        Args:
            posterior: 后验分布
            obs: 观测向量

        Returns:
            温度参数 τ ∈ [0.2, 2.0]
        """
        import numpy as np

        base_tau = 1.0

        bias_weight = posterior[2]
        noise_weight = posterior[1]
        mixed_weight = posterior[3]

        tau = base_tau * (1.0 - 0.4 * bias_weight) * (1.0 + 0.2 * noise_weight)

        if mixed_weight > 0.3:
            tau *= 0.8

        if obs['p_harm_raw'] > 0.5:
            tau *= (1.0 - 0.3 * obs['p_harm_raw'])

        if self._detect_slow_drift(obs):
            tau *= 0.7

        return float(np.clip(tau, 0.2, 2.0))

    def _compute_composite_harm_probability(
        self,
        posterior: np.ndarray,
        obs: Dict[str, float]
    ) -> float:
        """
        计算综合有害概率

        结合后验分布和直接观测的有害概率

        Args:
            posterior: 后验分布
            obs: 观测向量

        Returns:
            综合有害概率 ∈ [0, 1]
        """
        import numpy as np

        posterior_harm = (
            posterior[1] * 0.3 +
            posterior[2] * 0.6 +
            posterior[3] * 0.5
        )

        observed_harm = obs['p_harm_raw']

        composite = 0.4 * posterior_harm + 0.6 * observed_harm

        if self._detect_slow_drift(obs):
            composite = max(composite, 0.5)

        return float(np.clip(composite, 0.0, 1.0))

    def _make_cutoff_decision(
        self,
        van_event: bool,
        p_harm: float,
        p_harm_raw: float,
        posterior: np.ndarray,
        obs: Dict[str, float],
        dynamic_threshold: float
    ) -> Tuple[bool, Optional[str]]:
        """
        做出截断决策（集成连续截断信心机制）

        Args:
            van_event: VAN事件标志
            p_harm: 综合有害概率
            p_harm_raw: 原始有害概率
            posterior: 后验分布
            obs: 观测向量
            dynamic_threshold: 动态阈值

        Returns:
            (是否截断, 原因)
        """
        import numpy as np

        entropy_stats = {
            'mean': obs.get('mu_H', 0.5),
            'variance': obs.get('sigma_H2', 0.02),
            'trend': obs.get('k_H', 0.0)
        }

        should_base_cutoff = (
            van_event or
            p_harm > dynamic_threshold or
            p_harm_raw > 0.8 or
            self._detect_slow_drift(obs)
        )

        if self.use_consecutive_confidence:
            confidence_info = self.consecutive_confidence.update(
                should_cutoff=should_base_cutoff,
                entropy_stats=entropy_stats,
                van_event=van_event,
                p_harm_raw=p_harm_raw
            )

            override, override_reason = self.consecutive_confidence.should_override_cutoff()

            if override:
                return False, override_reason

            if should_base_cutoff and confidence_info['should_trust_cutoff']:
                if van_event:
                    self.cooldown_counter = self.cutoff_cooldown
                    return True, "VAN event: sensitive content detected"

                if posterior[3] > 0.3:
                    return True, f"Mixed attack detected: P(mixed)={posterior[3]:.2f}"
                elif posterior[2] > 0.3:
                    return True, f"Bias injection detected: P(bias)={posterior[2]:.2f}"
                elif posterior[1] > 0.3:
                    return True, f"Noise injection detected: P(noise)={posterior[1]:.2f}"
                else:
                    return True, f"High harm probability: {p_harm:.2f} > threshold {dynamic_threshold:.2f}"

            return False, None

        if van_event:
            self.cooldown_counter = self.cutoff_cooldown
            return True, "VAN event: sensitive content detected"

        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return False, "Cooldown active"

        if p_harm > dynamic_threshold:
            dominant_cause_idx = np.argmax(posterior)
            cause_name = self.CAUSE_NAMES[dominant_cause_idx]

            if posterior[3] > 0.3:
                return True, f"Mixed attack detected: P(mixed)={posterior[3]:.2f}"
            elif posterior[2] > 0.3:
                return True, f"Bias injection detected: P(bias)={posterior[2]:.2f}"
            elif posterior[1] > 0.3:
                return True, f"Noise injection detected: P(noise)={posterior[1]:.2f}"
            else:
                return True, f"High harm probability: {p_harm:.2f} > threshold {dynamic_threshold:.2f}"

        if p_harm_raw > 0.8:
            self.cooldown_counter = self.cutoff_cooldown
            return True, f"Critical harm probability: {p_harm_raw:.2f}"

        if self._detect_slow_drift(obs):
            self.cooldown_counter = self.cutoff_cooldown
            return True, "Slow drift self-referential loop detected"

        return False, None

    def _record_decision(
        self,
        entropy_mean: float,
        entropy_variance: float,
        van_event: bool,
        p_harm: float,
        cutoff: bool,
        reason: Optional[str]
    ) -> None:
        """
        记录决策历史

        Args:
            entropy_mean: 熵均值
            entropy_variance: 熵方差
            van_event: VAN事件标志
            p_harm: 有害概率
            cutoff: 是否截断
            reason: 决策原因
        """
        record = DecisionRecord(
            step=self.step_counter,
            entropy_mean=entropy_mean,
            entropy_variance=entropy_variance,
            van_event=van_event,
            p_harm=p_harm,
            cutoff=cutoff,
            cooldown_remaining=self.cooldown_counter,
            reason=reason
        )
        self.decision_history.append(record)

        if len(self.decision_history) > 1000:
            self.decision_history.pop(0)

    def get_posterior(self) -> Dict[str, float]:
        """
        获取当前病因后验概率

        Returns:
            病因后验概率字典
        """
        import numpy as np
        if len(self.posterior_history) > 0:
            return self.posterior_history[-1].copy()

        return {
            'normal': float(self.p_H[0]),
            'noise_injection': float(self.p_H[1]),
            'bias_injection': float(self.p_H[2]),
            'mixed': float(self.p_H[3])
        }

    def get_temporal_features(self) -> Dict[str, float]:
        """
        获取当前时序特征

        Returns:
            时序特征字典
        """
        return self._compute_temporal_features()

    def get_causal_attribution(self) -> Dict[str, Any]:
        """
        获取当前因果归因分析

        包括后验分布、时序特征和建议

        Returns:
            因果归因字典
        """
        import numpy as np

        posterior = self.get_posterior()
        temporal_features = self.get_temporal_features()

        dominant_cause = max(posterior, key=posterior.get)
        confidence = posterior[dominant_cause]

        suggestions = []
        if posterior['bias_injection'] > 0.3:
            suggestions.append("Increase temperature to escape bias loop")
        if posterior['noise_injection'] > 0.3:
            suggestions.append("Apply denoising or increase smoothing")
        if posterior['mixed'] > 0.3:
            suggestions.append("Combined intervention needed")
        if temporal_features['temporal_stability'] < 0.3:
            suggestions.append("System showing instability, consider reset")

        return {
            'dominant_cause': dominant_cause,
            'confidence': confidence,
            'posterior': posterior,
            'temporal_features': temporal_features,
            'suggestions': suggestions
        }

    def reset(self) -> None:
        """
        重置增强版贝叶斯L3控制器状态
        """
        import numpy as np

        self.p_H = np.array([0.7, 0.1, 0.1, 0.1])
        self.temporal_history.clear()
        self.posterior_history.clear()
        self.cooldown_counter = 0
        self.decision_history = []
        self.step_counter = 0

        if self.use_consecutive_confidence:
            self.consecutive_confidence.reset()

    def get_history(self, last_n: Optional[int] = None) -> List[DecisionRecord]:
        """
        获取决策历史

        Args:
            last_n: 返回最近的N条记录，None表示全部

        Returns:
            决策记录列表
        """
        if last_n is None:
            return self.decision_history.copy()
        return self.decision_history[-last_n:]

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取决策统计信息

        Returns:
            统计信息字典
        """
        if not self.decision_history:
            result = {
                "total_decisions": 0,
                "posterior": self.get_posterior()
            }
            if self.use_consecutive_confidence:
                result["consecutive_confidence"] = self.consecutive_confidence.get_statistics()
            return result

        cutoffs = [r for r in self.decision_history if r.cutoff]
        van_events = [r for r in self.decision_history if r.van_event]
        cooldowns = [r for r in self.decision_history if r.reason == "Cooldown"]

        cause_distribution = {
            'normal': 0,
            'noise_injection': 0,
            'bias_injection': 0,
            'mixed': 0
        }

        for posterior_dict in self.posterior_history:
            max_cause = max(posterior_dict, key=posterior_dict.get)
            cause_distribution[max_cause] += 1

        result = {
            "total_decisions": len(self.decision_history),
            "total_cutoffs": len(cutoffs),
            "total_van_events": len(van_events),
            "total_cooldowns": len(cooldowns),
            "cooldown_counter": self.cooldown_counter,
            "cutoff_ratio": len(cutoffs) / len(self.decision_history) if self.decision_history else 0,
            "posterior": self.get_posterior(),
            "cause_distribution": cause_distribution,
            "temporal_features": self.get_temporal_features()
        }

        if self.use_consecutive_confidence:
            result["consecutive_confidence"] = self.consecutive_confidence.get_statistics()

        return result
