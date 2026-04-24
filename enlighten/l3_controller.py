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
from typing import Dict, Optional, Any, List
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

    def __init__(self, prior_probs=[0.6, 0.2, 0.2]):
        super().__init__()
        import numpy as np
        self.p_H = np.array(prior_probs)   # P(H1), P(H2), P(H3)
        # 三种假设的似然模型参数
        self.models = {
            'normal': {
                'mu_std': 0.01,  # 过程噪声
                'obs_std': 0.02   # 观测噪声
            },
            'noise_injection': {
                'mu_std': 0.5,    # 高过程噪声
                'obs_std': 0.1
            },
            'bias_injection': {
                'mu_std': 0.01,
                'obs_std': 0.02,
                'step_gain': 2.0  # 偏见阶跃增益
            }
        }

        self.cutoff_cooldown = 10
        self.cooldown_counter = 0
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

        # 构建观测向量
        o_int = {
            'mu_H': entropy_stats.get("mean", 0.0),
            'sigma_H2': entropy_stats.get("variance", 0.0),
            'k_H': entropy_stats.get("trend", 0.0),
            'p_harm_raw': p_harm
        }

        # 计算似然
        likelihoods = []
        for condition in ['normal', 'noise_injection', 'bias_injection']:
            model = self.models[condition]
            # 简化的似然计算
            if condition == 'normal':
                # 正常状态：低方差，稳定趋势
                lik = np.exp(-0.5 * (o_int['sigma_H2']/0.05)**2) * \
                      np.exp(-0.5 * (abs(o_int['k_H'])/0.1)**2)
            elif condition == 'noise_injection':
                # 噪声状态：高方差
                lik = np.exp(-0.5 * ((o_int['sigma_H2']-0.2)/0.1)**2)
            else:  # bias_injection
                # 偏见状态：低均值，高危险值
                # 低均值的似然
                mu_lik = np.exp(-0.5 * ((o_int['mu_H']-0.2)/0.05)**2)  # 以0.2为中心
                # 高危险值的似然
                harm_lik = np.exp(-0.5 * ((o_int['p_harm_raw']-0.8)/0.1)**2)  # 以0.8为中心
                # 负趋势的似然
                trend_lik = np.exp(-0.5 * ((o_int['k_H']+0.1)/0.05)**2)  # 以-0.1为中心
                lik = mu_lik * harm_lik * trend_lik
            likelihoods.append(lik)

        # 贝叶斯更新
        unnorm = self.p_H * np.array(likelihoods)
        self.p_H = unnorm / unnorm.sum()

        # 温度调节
        tau_default = 1.0
        tau = tau_default * (1.0 - 0.5 * self.p_H[2]) * (1.0 + 0.3 * self.p_H[1])
        tau = np.clip(tau, 0.2, 2.0)

        # 连续截断信心
        p_harm = self.p_H[2] * 0.5 + o_int['p_harm_raw'] * 0.5  # 增加原始危险值的权重

        # 截断决策
        cutoff = False
        reason = None
        if van_event:
            cutoff = True
            reason = "VAN event: sensitive content detected"
            self.cooldown_counter = self.cutoff_cooldown
        elif p_harm > 0.6 or o_int['p_harm_raw'] > 0.8:
            cutoff = True
            reason = f"High harm probability: {p_harm:.2f}"
            self.cooldown_counter = self.cutoff_cooldown
        elif self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            reason = "Cooldown"

        # 记录决策
        self._record_decision(
            entropy_mean=o_int['mu_H'],
            entropy_variance=o_int['sigma_H2'],
            van_event=van_event,
            p_harm=p_harm,
            cutoff=cutoff,
            reason=reason
        )

        # 生成调控信号
        return ControlSignals(
            tau=float(tau),
            theta=0.7,
            alpha=0.1,
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
            'bias_injection': float(self.p_H[2])
        }

    def reset(self) -> None:
        """
        重置贝叶斯L3控制器状态
        """
        import numpy as np
        self.p_H = np.array([0.8, 0.1, 0.1])
        self.cooldown_counter = 0
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
            "posterior": self.get_posterior()
        }
