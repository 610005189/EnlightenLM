# EnlightenLM 架构设计文档

> 版本: v1.0
> 更新日期: 2026-04-23

---

## 目录

1. [系统概述](#1-系统概述)
2. [设计动机](#2-设计动机)
3. [系统架构](#3-系统架构)
4. [核心组件详解](#4-核心组件详解)
5. [数据流与控制流](#5-数据流与控制流)
6. [安全机制](#6-安全机制)
7. [审计系统](#7-审计系统)
8. [配置管理](#8-配置管理)
9. [扩展性设计](#9-扩展性设计)

---

## 1. 系统概述

### 1.1 项目简介

EnlightenLM（觉悟三层架构）是一个基于认知神经科学的大模型安全推理与元认知框架。它将大模型推理过程解耦为三层网络结构，实现高效、可审计、可截断的安全推理。

### 1.2 设计目标

| 目标 | 描述 |
|------|------|
| **可控性** | 通过元控制器实时调节注意力方向，防止模型行为偏离 |
| **可解释性** | 通过工作记忆和元描述生成，提高模型透明度 |
| **可审计性** | 通过密码学方法确保审计日志不可篡改 |
| **安全性** | 通过多层防御机制，包括截断、隔离和威胁检测 |
| **高效性** | 通过稀疏注意力降低长上下文计算复杂度 |

### 1.3 技术指标

| 指标 | 值 |
|------|-----|
| 支持模型规模 | 7B - 70B 参数 |
| 最大上下文长度 | 128K tokens |
| 截断响应时间 | < 10ms |
| 审计日志存储效率 | 压缩率 70% |

---

## 2. 设计动机

### 2.1 传统大模型的三大结构性困境

#### 2.1.1 注意力不可控

**问题描述**: 标准Transformer的注意力机制是完全数据驱动的，模型可能忽略关键指令或被恶意提示词劫持。

**传统应对**: 提示词软引导、system prompt设定。

**EnlightenLM的解法**: 双流注意力 + 元控制器
- 目标驱动流（DAN）: 根据任务类型强制引导注意力
- 刺激驱动流（VAN）: 检测异常模式并触发中断

#### 2.1.2 自指递归与幻觉

**问题描述**: 模型在生成长文本时可能陷入自我参照循环，导致输出重复、逻辑混乱。

**传统应对**: 依赖训练数据分布的规律，无机制约束。

**EnlightenLM的解法**: DMN抑制 + 遗忘门
- 主动衰减内部噪声与KV缓存
- 检测自反循环并实时截断

#### 2.1.3 安全审计缺失

**问题描述**: 现有LLM的决策过程是不可追溯的，日志可篡改，无法进行事后审查。

**传统应对**: 简单的API日志记录。

**EnlightenLM的解法**: 密码学审计链 + 离线复盘
- 每一步注意力状态、调控动作均被签名存证
- 事后可生成可读报告

### 2.2 认知神经科学理论基础

EnlightenLM的设计灵感来自人脑的三个注意力网络：

| 网络 | 功能 | EnlightenLM对应 |
|------|------|----------------|
| **DAN** (Default Mode Network) | 目标驱动的主动聚焦 | 任务偏置流 |
| **VAN** (Ventral Attention Network) | 刺激驱动的自动重定向 | 显著性检测 |
| **DMN** (Default Mode Network) | 自发、自我参照的思维流 | DMN抑制 + 遗忘门 |

---

## 3. 系统架构

### 3.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              用户输入                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  L3 元注意力控制器（前额叶模拟）                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 输入:                                                                 │   │
│  │   - L2的熵统计 (μ_H, σ_H², k_H)                                      │   │
│  │   - VAN事件标志                                                       │   │
│  │   - 任务嵌入                                                          │   │
│  │                                                                       │   │
│  │ 输出:                                                                 │   │
│  │   - 温度 τ ∈ [0.1, 2.0]                                              │   │
│  │   - 稀疏阈值 θ ∈ [0.5, 0.9]                                          │   │
│  │   - DMN系数 α ∈ [0.0, 1.0]                                           │   │
│  │   - 稳定性标志 s ∈ {0, 1}                                            │   │
│  │   - 截断信号 cutoff ∈ {0, 1}                                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                             │
│                              │ 调控信号 (τ, θ, α, s, cutoff)                │
└──────────────────────────────┼─────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  L2 工作记忆层（背侧/腹侧注意网络的中介）                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 数据结构:                                                             │   │
│  │   - 记忆矩阵 M ∈ ℝ^(m×d), m=512, d=1024                             │   │
│  │   - 活跃索引集 A, |A| = m                                            │   │
│  │                                                                       │   │
│  │ 功能:                                                                 │   │
│  │   - 上下文压缩: n个token → m个活跃token                               │   │
│  │   - 熵统计计算: 均值μ_H, 方差σ_H², 趋势k_H                           │   │
│  │   - 稀疏键值提供: (K̃, Ṽ) 给L1                                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                             │
│                              │ 快照 + 熵统计 + 稀疏KV                       │
└──────────────────────────────┼─────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  L1 生成层（双流注意力 + DMN抑制 + 遗忘门）                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ DAN流 (目标驱动):                                                    │   │
│  │   - 输入: 任务偏置 B_DAN + Q, K, V                                  │   │
│  │   - 注意力: softmax(QK^T/√d + B_DAN)V                              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ VAN流 (刺激驱动):                                                    │   │
│  │   - 输入: 显著性检测结果 + 中断掩码 M_VAN                            │   │
│  │   - 功能: 异常模式识别、敏感词检测                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                             │
│                              │ 门控融合 g                                   │
│                              ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 融合与输出:                                                           │   │
│  │   Attn_fused = g·Attn_DAN + (1-g)·Attn_VAN                         │   │
│  │   Output = renormalize(softmax(Attn_fused/τ)·1_{>θ})V               │   │
│  │          + α·ξ (DMN抑制)                                            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                             │
│                              │ token序列 + 注意力权重                       │
└──────────────────────────────┼─────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  审计日志系统（实时写入） + 离线复盘服务（异步）                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 实时审计:                                                             │   │
│  │   - 哈希链: H_i = SHA256(H_{i-1} || data_i)                         │   │
│  │   - HMAC签名: 确保数据完整性                                          │   │
│  │   - 紧凑二进制格式存储                                               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 离线复盘:                                                             │   │
│  │   - 读取历史日志                                                      │   │
│  │   - 分析截断事件                                                      │   │
│  │   - 生成自然语言报告                                                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 三层职责划分

| 层级 | 名称 | 主要职责 | 比喻 |
|------|------|---------|------|
| **L1** | 生成层 | 双流注意力融合、token生成、DMN抑制 | 大脑皮层（执行） |
| **L2** | 工作记忆层 | 上下文压缩、熵统计、稀疏化 | 海马体（记忆） |
| **L3** | 元控制层 | 熵监控、截断决策、调控信号生成 | 前额叶（调控） |

---

## 4. 核心组件详解

### 4.1 L1 生成层组件

#### 4.1.1 双流注意力机制

**DAN (目标驱动注意力网络)**

```python
class DANAttention(nn.Module):
    """
    目标驱动的主动聚焦
    特点: 根据任务类型强制引导注意力方向
    """

    def forward(self, Q, K, V, task_bias):
        # 任务偏置投影
        bias = self.task_bias_proj(task_bias)  # [batch, num_heads]

        # 标准缩放点积注意力 + 任务偏置
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores + bias.unsqueeze(1)  # 应用任务偏置

        attn_weights = F.softmax(scores, dim=-1)
        attn_output = attn_weights @ V

        return attn_output, attn_weights
```

**VAN (刺激驱动注意力网络)**

```python
class VANAttention(nn.Module):
    """
    刺激驱动的自动重定向
    特点: 检测异常模式，触发硬中断
    """

    def __init__(self, vocab_size):
        self.saliency_detector = SaliencyDetector(vocab_size)
        self.mask_generator = InterruptMaskGenerator()

    def forward(self, tokens, hidden_states):
        # 显著性检测
        saliency_map = self.saliency_detector(hidden_states)

        # VAN事件检测
        van_event, event_type = self.detect_van_event(tokens, saliency_map)

        if van_event:
            mask = self.mask_generator.generate_mask(event_type, hidden_states.shape[1])
            return mask, True

        return None, False
```

#### 4.1.2 双流融合

```python
class AttentionFusion(nn.Module):
    """
    融合门控机制
    公式: Attn_fused = g·Attn_DAN + (1-g)·Attn_VAN
    """

    def forward(self, attn_dan, attn_van, tau, theta):
        # 预测融合权重 g ∈ [0, 1]
        g = torch.sigmoid(self.gate_predictor(
            torch.cat([attn_dan.mean(), attn_van.mean()])
        ))

        # 融合
        fused = g * attn_dan + (1 - g) * attn_van

        # 应用温度 (控制分布锐度)
        fused = F.softmax(fused / tau, dim=-1)

        # 应用稀疏截断 (只保留>θ的注意力)
        fused = fused * (fused > theta).float()

        return fused
```

#### 4.1.3 DMN抑制与遗忘门

**DMN抑制**

```python
class DMNInhibition(nn.Module):
    """
    默认模式网络抑制
    功能: 抑制内部噪声 ξ，防止无意义自循环
    """

    def forward(self, hidden_states, alpha):
        # 估计内部噪声
        noise = self.noise_estimator(hidden_states)

        # 应用DMN抑制: α · ξ
        inhibited = hidden_states - alpha * noise * self.inhibition_strength

        return inhibited
```

**遗忘门**

```python
class ForgetGate(nn.Module):
    """
    遗忘门机制
    功能: 提供指数衰减的KV缓存，防止"陷入"过去状态
    """

    def forward(self, prev_hidden, current_input, decay_rate=0.95):
        # 计算遗忘门值 f ∈ [0, 1]
        f = torch.sigmoid(self.forget_proj(
            torch.cat([prev_hidden, current_input])
        ))

        # 指数衰减
        decayed = f * decay_rate

        # 选择性遗忘
        return decayed * prev_hidden + (1 - decayed) * current_input
```

### 4.2 L2 工作记忆层组件

#### 4.2.1 工作记忆矩阵

```python
class WorkingMemory(nn.Module):
    """
    工作记忆矩阵 M (m × d)
    - m: 活跃token数量 (远小于n, 默认512)
    - d: 嵌入维度 (如1024)
    """

    def __init__(self, m=512, d=1024):
        self.M = nn.Parameter(torch.zeros(m, d))
        self.active_indices = []
        self.m = m
        self.d = d

    def update(self, key, value, importance_scores):
        """
        基于重要性分数更新记忆
        策略: 保留top-m个最重要的token
        """
        # 计算top-m索引
        topk_indices = torch.topk(importance_scores, self.m).indices

        # 更新记忆矩阵
        for i, idx in enumerate(topk_indices):
            self.M[i] = value[idx]
            self.active_indices[i] = idx.item()

    def get_sparse_kv(self):
        """
        返回稀疏的键值对 (K̃, Ṽ)
        用于L1的稀疏注意力计算
        """
        return self.M, self.M, self.active_indices
```

#### 4.2.2 熵统计追踪器

```python
class EntropyTracker:
    """
    注意力熵统计追踪器
    计算滑动窗口内的:
    - 均值 μ_H
    - 方差 σ_H²
    - 趋势 k_H
    """

    def __init__(self, window_size=100):
        self.window_size = window_size
        self.history = deque(maxlen=window_size)

    def update(self, attention_weights):
        """
        计算并记录当前时刻的注意力熵
        H = -Σ p_i log(p_i)
        """
        entropy = -torch.sum(
            attention_weights * torch.log(attention_weights + 1e-10),
            dim=-1
        ).mean()
        self.history.append(entropy.item())

    def get_statistics(self):
        """
        获取滑动统计
        """
        if len(self.history) < 2:
            return {"mean": 0, "variance": 0, "trend": 0, "current": 0}

        history = torch.tensor(self.history)

        return {
            "mean": history.mean().item(),        # μ_H
            "variance": history.var().item(),    # σ_H²
            "trend": self._compute_trend(history),  # k_H
            "current": history[-1].item()
        }

    def _compute_trend(self, history):
        """
        计算趋势 (简单线性回归斜率)
        k_H > 0: 注意力发散
        k_H < 0: 注意力聚焦
        """
        if len(history) < 2:
            return 0.0
        x = torch.arange(len(history)).float()
        return torch.polyfit(x, history, 1)[0].item()
```

### 4.3 L3 元控制器组件

#### 4.3.1 熵监控与截断决策

```python
class L3MetaController(nn.Module):
    """
    L3 元注意力控制器 (前额叶模拟)
    核心功能: 熵监控、截断决策、调控信号生成
    """

    def __init__(self, config):
        self.entropy_threshold = config.entropy_threshold  # 默认0.5
        self.variance_threshold = config.variance_threshold  # 默认0.05
        self.tau_range = config.tau_range  # (0.1, 2.0)
        self.theta_range = config.theta_range  # (0.5, 0.9)

    def forward(self, entropy_stats, van_event, task_embedding):
        """
        决策流程:
        1. 检查VAN事件 -> 立即截断
        2. 检查截断判据 -> 条件截断
        3. 正常调控
        """
        mu_h = entropy_stats["mean"]
        sigma_h = math.sqrt(entropy_stats["variance"])
        k_h = entropy_stats["trend"]

        # 1. VAN事件处理
        if van_event:
            return self._van_cutoff_response()

        # 2. 截断判据检查
        if self._should_cutoff(mu_h, sigma_h, k_h):
            return self._cutoff_response(mu_h)

        # 3. 正常调控
        return self._normal_control(task_embedding)

    def _should_cutoff(self, mu_h, sigma_h, k_h):
        """
        截断判据:
        - 低注意力熵 (μ_H < 0.5)
        - 低方差 (σ_H < 0.05)
        - 持续下降趋势 (k_H < 0)
        """
        return (mu_h < self.entropy_threshold and
                sigma_h < self.variance_threshold and
                k_h < 0)

    def _van_cutoff_response(self):
        """VAN事件触发的立即截断"""
        return {
            "tau": 0.1,    # 降低温度，快速收敛
            "theta": 0.9,  # 提高阈值，只保留最重要token
            "alpha": 0.5,  # 增强DMN抑制
            "s": False,    # 标记为不稳定
            "cutoff": True,
            "reason": "VAN event: sensitive content detected"
        }

    def _cutoff_response(self, mu_h):
        """自指循环检测触发的截断"""
        return {
            "tau": max(0.1, mu_h * 2),  # 与熵成反比
            "theta": 0.8,
            "alpha": 0.3,
            "s": False,
            "cutoff": True,
            "reason": "Self-referential loop detected"
        }

    def _normal_control(self, task_embedding):
        """正常调控"""
        return {
            "tau": 0.7,
            "theta": 0.7,
            "alpha": 0.1,
            "s": True,     # 标记为稳定
            "cutoff": False,
            "reason": None
        }
```

---

## 5. 数据流与控制流

### 5.1 推理流程时序图

```
用户输入 ──┐
           │
           ▼
┌──────────────────┐
│   L3 元控制器     │◄─────────────────────┐
│   (初始化参数)    │                      │
└────────┬─────────┘                      │
         │ 调控信号 (τ, θ, α, s)          │
         ▼                               │
┌──────────────────┐                     │
│   L2 工作记忆     │                     │
│   (压缩上下文)    │                     │
└────────┬─────────┘                     │
         │ 稀疏KV + 熵统计                │
         ▼                               │
┌──────────────────┐                      │
│   L1 双流注意力   │                     │
│   (生成token)    │─────────────────────►│
└────────┬─────────┘  反馈(熵,VAN事件)     │
         │                                  │
         ▼                                  │
    [生成token]                              │
         │                                  │
         ├──────────────────────────────────┤
         │                                  │
         ▼                                  ▼
┌──────────────────┐            ┌──────────────────┐
│   L2 更新记忆     │            │   审计日志写入    │
│   (更新M和A)      │            │   (哈希链+HMAC)   │
└──────────────────┘            └──────────────────┘
         │                                  │
         │ 熵统计                           │
         └──────────────────────────────────►
                    L3 反馈循环
```

### 5.2 状态机模型

```
┌─────────────┐    VAN事件     ┌─────────────┐
│   正常     │──────────────►│   VAN截断   │
│   状态     │               │   状态      │
└─────────────┘               └─────────────┘
       ▲                              │
       │                              │
       │ 熵恢复                        │ 截断完成
       │                              ▼
       │                       ┌─────────────┐
       └──────────────────────│   恢复      │
                               │   状态      │
                               └─────────────┘
       ▲                              │
       │ 持续低熵                      │
       ▼                              │
┌─────────────┐                        │
│   自指截断   │────────────────────────┘
│   状态      │
└─────────────┘
```

---

## 6. 安全机制

### 6.1 多层防御体系

| 层级 | 机制 | 作用 |
|------|------|------|
| **L1** | DAN任务偏置 | 强制引导注意力到安全方向 |
| **L1** | VAN显著性检测 | 识别并阻断敏感内容 |
| **L1** | DMN抑制 | 防止无意义自循环 |
| **L2** | 熵监控 | 实时检测异常模式 |
| **L3** | 截断决策 | 硬中断问题生成 |
| **L3** | 调控信号 | 动态调节生成行为 |
| **系统** | 审计日志 | 完整记录，可追溯 |

### 6.2 威胁检测类型

| 威胁类型 | 检测方法 | 响应措施 |
|---------|---------|---------|
| **越狱攻击** | 模式匹配 + 熵异常 | 立即截断 |
| **注入攻击** | 特殊字符序列检测 | 忽略+警告 |
| **隐私侵犯** | 敏感词匹配 | 过滤+记录 |
| **DOS攻击** | 请求频率检测 | 限流 |

---

## 7. 审计系统

### 7.1 哈希链设计

```python
class AuditHashChain:
    """
    审计哈希链
    特性:
    - 链接: H_i = SHA256(H_{i-1} || data_i)
    - 不可篡改: 任何data_i的修改都会导致后续哈希不匹配
    - 可验证: 从任意点可验证链的完整性
    """

    def append(self, data):
        # 计算当前数据哈希
        current_hash = sha256(self.serialize(data))

        # 链接到前一个哈希
        if self.chain:
            link_hash = sha256(
                self.serialize({
                    "prev": self.chain[-1].hash,
                    "current": current_hash
                })
            )
        else:
            link_hash = current_hash

        entry = AuditEntry(
            index=len(self.chain),
            hash=link_hash,
            data=data,
            timestamp=time.time()
        )
        self.chain.append(entry)
        return link_hash

    def verify(self):
        """从后向前验证"""
        for i in range(1, len(self.chain)):
            expected_prev = self.chain[i].data["prev_hash"]
            actual_prev = self.chain[i-1].hash
            if expected_prev != actual_prev:
                return False
        return True
```

### 7.2 离线复盘服务

```python
class OfflineReviewService:
    """
    离线复盘服务
    功能:
    1. 读取审计日志
    2. 分析截断事件
    3. 生成自然语言报告
    """

    def generate_report(self, session_id):
        # 获取日志和快照
        logs = self.get_session_logs(session_id)
        snapshots = self.get_snapshots(session_id)

        # 分析
        cutoff_events = self.analyze_cutoffs(logs)
        attention_patterns = self.analyze_attention(snapshots)
        safety_events = self.analyze_safety_events(logs)

        # 生成报告
        return self.format_report({
            "session_id": session_id,
            "cutoff_events": cutoff_events,
            "attention_patterns": attention_patterns,
            "safety_events": safety_events,
            "statistics": self.compute_statistics(logs)
        })
```

---

## 8. 配置管理

### 8.1 配置层次结构

```
configs/
├── core_rules.yaml          # 核心价值观敏感词表
├── task_embeddings.yaml     # 任务嵌入向量
└── hyperparameters.yaml     # 超参数配置
```

### 8.2 核心价值观配置 (core_rules.yaml)

```yaml
negative_vocab:
  - 暴力相关词
  - 色情相关词
  - 歧视相关词
  - 违法犯罪相关词

positive_vocab:
  - 和平
  - 友爱
  - 正义
  - 诚信

bias_strength:
  negative: -1e9   # 强负偏置
  positive: 2.0    # 弱正偏置
```

### 8.3 超参数配置 (hyperparameters.yaml)

```yaml
l1_generation:
  temperature:
    default: 0.7
    range: [0.1, 2.0]
  sparse_threshold:
    default: 0.7
    range: [0.5, 0.9]

l2_working_memory:
  memory_size: 512       # m: 活跃token数量
  embedding_dim: 1024    # d: 嵌入维度
  entropy_window: 100    # 滑动窗口大小

l3_controller:
  entropy_threshold: 0.5   # μ_H < 0.5 触发截断
  variance_threshold: 0.05 # σ_H < 0.05 触发截断
  tau_range: [0.1, 2.0]
  theta_range: [0.5, 0.9]
```

---

## 9. 扩展性设计

### 9.1 自定义注意力机制

```python
class CustomAttention(nn.Module):
    """
    用户可自定义注意力机制
    只需实现:
    1. forward() 方法
    2. get_attention_weights() 方法
    """

    def forward(self, Q, K, V):
        # 自定义注意力实现
        pass

    def get_attention_weights(self):
        return self.last_attention_weights
```

### 9.2 自定义截断判据

```python
class CustomCutoffStrategy:
    """
    用户可自定义截断策略
    只需实现 should_cutoff() 方法
    """

    def should_cutoff(self, entropy_stats, van_event, task_embedding):
        # 自定义截断逻辑
        pass
```

### 9.3 模型后端支持

| 后端 | 状态 | 说明 |
|------|------|------|
| **HuggingFace Transformers** | ✅ 支持 | 原型验证 |
| **vLLM** | ⚠️ 规划中 | 生产级优化 |
| **TensorRT-LLM** | 📋 待定 | 高性能推理 |
| **Ollama** | 📋 待定 | 本地部署 |

---

## 附录

### A. 术语表

| 术语 | 定义 |
|------|------|
| **DAN** | Default Attention Network，目标驱动注意力网络 |
| **VAN** | Ventral Attention Network，刺激驱动注意力网络 |
| **DMN** | Default Mode Network，默认模式网络 |
| **熵** | Attention Entropy，衡量注意力分布的混乱程度 |
| **截断** | Cutoff，元控制器触发的生成中断 |

### B. 参考资料

1. Attention Is All You Need (Vaswani et al., 2017)
2. The Human Attention Network (Corbetta & Shulman, 2002)
3. Default Mode Network (Raichle et al., 2001)

---

*文档版本: v1.0*
*最后更新: 2026-04-23*
