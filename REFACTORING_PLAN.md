# EnlightenLM 项目重构计划

> 基于认知神经科学的大模型安全推理与元认知框架

---

## 一、重构背景与目标

### 1.1 当前实现 vs 新版设计差距分析

| 模块 | 当前实现 | 新版设计要求 | 差距 |
|------|---------|-------------|------|
| **L1 生成层** | 单一注意力机制，使用模拟模型 | **双流注意力 (DAN+VAN)** + DMN抑制 + 遗忘门 | 需要完全重写 |
| **L2 工作记忆** | 简单的元描述生成 | **稀疏注意力** (m×d记忆矩阵)、熵统计、活跃索引集A | 需要完全重写 |
| **L3 元控制器** | 基础偏置生成 | **实时熵监控**、VAN事件检测、截断决策 | 需要大幅增强 |
| **注意力机制** | 无 | **稀疏截断 θ**、动态温度 τ、门控融合 g | 新增 |
| **审计系统** | 基础日志+哈希链 | **离线复盘服务**、HMAC签名、二进制压缩 | 需要增强 |
| **数学基础** | 无 | **形式化验证**、复杂度分析、收敛性证明 | 新增文档 |

### 1.2 重构目标

1. 实现人脑三层注意力网络（DAN/VAN/DMN）的工程映射
2. 建立形式化数学基础，确保框架可验证
3. 提供生产级工程可落地性
4. 完整的文档和测试覆盖

---

## 二、目标项目结构

```
EnlightenLM/
├── README.md
├── LICENSE
├── CONTRIBUTING.md
├── requirements.txt
├── pyproject.toml
├── enlighten/                    # 核心框架 (原src/)
│   ├── __init__.py
│   ├── l1_generation.py         # L1 双流注意力 + DMN抑制 + 遗忘门
│   ├── l2_working_memory.py     # L2 工作记忆、熵统计、稀疏注意力
│   ├── l3_controller.py         # L3 元控制、熵监控、截断决策
│   ├── attention/                # 注意力机制子模块
│   │   ├── __init__.py
│   │   ├── dan.py               # DAN (目标驱动注意力网络)
│   │   ├── van.py               # VAN (刺激驱动注意力网络)
│   │   ├── fusion.py            # 双流融合门控
│   │   └── sparse.py            # 稀疏注意力实现
│   ├── audit/                    # 审计系统
│   │   ├── __init__.py
│   │   ├── chain.py             # 哈希链
│   │   ├── hmac_sign.py         # HMAC签名
│   │   └── offline_review.py    # 离线复盘服务
│   ├── memory/                   # 工作记忆子模块
│   │   ├── __init__.py
│   │   ├── working_memory.py    # 工作记忆矩阵 M (m×d)
│   │   ├── entropy_tracker.py   # 熵统计追踪器
│   │   └── active_indices.py    # 活跃token索引集 A
│   ├── cutoff/                   # 截断控制子模块
│   │   ├── __init__.py
│   │   ├── dmn.py               # DMN 抑制机制
│   │   ├── forget_gate.py        # 遗忘门
│   │   └── cutoff_decision.py   # 截断决策逻辑
│   ├── utils.py                 # 通用工具函数
│   └── config.py                # 配置管理
├── configs/                      # 配置文件
│   ├── core_rules.yaml          # 核心价值观敏感词表
│   ├── task_embeddings.yaml      # 任务嵌入向量
│   └── hyperparameters.yaml     # 超参数配置
├── docs/                        # 文档
│   ├── architecture.md          # 完整架构设计文档
│   ├── math_verification.md     # 数学合理性证明
│   ├── integration_guide.md     # 集成指南
│   └── api_reference.md         # API参考文档
├── examples/                    # 示例代码
│   ├── demo_math_reasoning.py  # 数学推理演示
│   ├── demo_safety_cutoff.py    # 安全截断演示
│   └── demo_dual_stream.py      # 双流注意力演示
├── tests/                       # 测试套件
│   ├── unit/                   # 单元测试
│   │   ├── test_dan.py
│   │   ├── test_van.py
│   │   ├── test_working_memory.py
│   │   ├── test_entropy_tracker.py
│   │   └── test_cutoff.py
│   ├── integration/             # 集成测试
│   │   ├── test_l1_pipeline.py
│   │   ├── test_l2_pipeline.py
│   │   └── test_full_pipeline.py
│   └── benchmark/              # 性能基准测试
│       ├── benchmark_attention.py
│       └── benchmark_memory.py
├── scripts/                    # 脚本工具
│   ├── validate_math.py        # 数学验证脚本
│   └── generate_report.py      # 报告生成脚本
└── deployment/                 # 部署配置
    ├── docker/
    │   └── Dockerfile
    └── kubernetes/
        └── deployment.yaml
```

---

## 三、重构详细计划

### Phase 1：基础设施重构 (第1-2周)

#### 3.1.1 目录结构重组

| 任务 | 描述 | 优先级 | 工作量 |
|------|------|--------|--------|
| T1.1 | 创建新的目录结构 `enlighten/`、`configs/`、`docs/`、`examples/` | P0 | 0.5天 |
| T1.2 | 创建 `pyproject.toml` 替代 requirements.txt | P1 | 0.5天 |
| T1.3 | 迁移现有代码到新目录结构 | P0 | 1天 |
| T1.4 | 更新所有导入路径和包引用 | P0 | 1天 |
| T1.5 | 创建 CONTRIBUTING.md | P2 | 0.5天 |

#### 3.1.2 配置管理系统

```python
# enlighten/config.py - 配置管理设计
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class L1Config:
    model_name: str = "deepseek-ai/DeepSeek-V3"
    device: str = "cuda"
    max_memory: int = 16 * 1024**3  # 16GB

@dataclass
class L2Config:
    memory_size: int = 512  # m: 活跃token数量
    embedding_dim: int = 1024  # d: 嵌入维度

@dataclass
class L3Config:
    tau_range: Tuple[float, float] = (0.1, 2.0)  # 温度范围
    theta_default: float = 0.7  # 默认稀疏阈值
    entropy_threshold: float = 0.5  # 低熵阈值
    variance_threshold: float = 0.05  # 低方差阈值
```

### Phase 2：L1 生成层重构 (第3-5周)

#### 3.2.1 双流注意力机制 (DAN + VAN)

```python
# enlighten/attention/dan.py - 目标驱动注意力网络
class DANAttention:
    """
    DAN (Default Mode Network) - 目标驱动的主动聚焦
    输入: 任务偏置 B_DAN, 查询Q, 键K, 值V
    输出: 目标驱动的注意力权重
    """

    def __init__(self, task_bias_dim: int, num_heads: int = 12):
        self.task_bias_proj = nn.Linear(task_bias_dim, num_heads)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, Q: Tensor, K: Tensor, V: Tensor,
                task_bias: Tensor) -> Tuple[Tensor, Tensor]:
        # 任务偏置投影
        bias = self.task_bias_proj(task_bias)  # [batch, num_heads]

        # 标准注意力 + 任务偏置增强
        attn_output, attn_weights = self.attention(Q, K, V)

        # 应用任务偏置 (B_DAN 调节)
        attn_weights = attn_weights + bias.unsqueeze(-1)

        return attn_output, attn_weights
```

```python
# enlighten/attention/van.py - 刺激驱动注意力网络
class VANAttention:
    """
    VAN (Ventral Attention Network) - 刺激驱动的自动重定向
    功能: 显著性检测、异常模式识别、硬中断
    """

    def __init__(self, vocab_size: int):
        self.saliency_detector = SaliencyDetector(vocab_size)
        self.mask_generator = InterruptMaskGenerator()

    def forward(self, tokens: List[int],
                hidden_states: Tensor) -> Tuple[Tensor, bool]:
        # 显著性检测
        saliency_map = self.saliency_detector(hidden_states)

        # 检测VAN事件 (敏感词/异常模式)
        van_event, event_type = self.detect_van_event(tokens, saliency_map)

        # 生成中断掩码
        if van_event:
            mask = self.mask_generator.generate_mask(
                event_type, hidden_states.shape[1]
            )
            return mask, True  # 触发中断

        return None, False  # 无中断
```

```python
# enlighten/attention/fusion.py - 双流融合
class AttentionFusion:
    """
    融合门控: g · Attn_DAN + (1-g) · Attn_VAN
    动态温度 τ + 稀疏截断 θ
    """

    def __init__(self):
        self.gate_predictor = nn.Linear(256, 1)  # 门控预测器

    def forward(self, attn_dan: Tensor, attn_van: Tensor,
                tau: float, theta: float) -> Tensor:
        # 预测融合权重 g
        combined = torch.cat([attn_dan.mean(), attn_van.mean()])
        g = torch.sigmoid(self.gate_predictor(combined))

        # 融合注意力
        fused = g * attn_dan + (1 - g) * attn_van

        # 应用温度和稀疏截断
        fused = self.apply_temperature(fused, tau)
        fused = self.apply_sparse_threshold(fused, theta)

        return fused

    def apply_temperature(self, attn: Tensor, tau: float) -> Tensor:
        return softmax(attn / tau)

    def apply_sparse_threshold(self, attn: Tensor, theta: float) -> Tensor:
        return attn * (attn > theta)
```

#### 3.2.2 DMN 抑制与遗忘门

```python
# enlighten/cutoff/dmn.py - 默认模式网络抑制
class DMNInhibition:
    """
    DMN (Default Mode Network) - 自发、自我参照的思维流
    功能: 抑制内部噪声 ξ，防止无意义自循环
    """

    def __init__(self, hidden_dim: int):
        self.noise_estimator = nn.Linear(hidden_dim, hidden_dim)
        self.inhibition_strength = 0.1

    def forward(self, hidden_states: Tensor,
                alpha: float) -> Tensor:
        # 估计内部噪声
        noise = self.noise_estimator(hidden_states)

        # 应用DMN抑制: α · ξ
        inhibited = hidden_states - alpha * noise * self.inhibition_strength

        return inhibited
```

```python
# enlighten/cutoff/forget_gate.py - 遗忘门
class ForgetGate:
    """
    遗忘门 f_t: 提供指数衰减的KV缓存
    防止模型"陷入"过去的无效状态
    """

    def __init__(self, hidden_dim: int):
        self.forget_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, prev_hidden: Tensor,
                current_input: Tensor, decay_rate: float = 0.95) -> Tensor:
        # 计算遗忘门值
        combined = torch.cat([prev_hidden, current_input])
        f = self.sigmoid(self.forget_proj(combined))

        # 应用遗忘: 指数衰减
        decayed = f * decay_rate

        # 选择性遗忘
        return decayed * prev_hidden + (1 - decayed) * current_input
```

### Phase 3：L2 工作记忆层重构 (第6-8周)

#### 3.3.1 工作记忆矩阵

```python
# enlighten/memory/working_memory.py - 工作记忆
class WorkingMemory:
    """
    工作记忆矩阵 M (m × d)
    - m: 活跃token数量 (远小于n)
    - d: 嵌入维度
    """

    def __init__(self, m: int = 512, d: int = 1024):
        self.M = torch.zeros(m, d)  # 记忆矩阵
        self.active_indices = []      # 活跃token索引集 A
        self.m = m
        self.d = d

    def update(self, key: Tensor, value: Tensor,
               indices: List[int]):
        """更新工作记忆"""
        # 选择top-m个最重要的token
        importance = torch.norm(key, dim=-1)
        topk_indices = torch.topk(importance, self.m).indices

        # 更新记忆矩阵
        for i, idx in enumerate(topk_indices):
            self.M[i] = value[idx]
            self.active_indices[i] = indices[idx]

    def get_sparse_kv(self) -> Tuple[Tensor, Tensor, List[int]]:
        """返回稀疏的键值对 (K̃, Ṽ) 和索引"""
        return self.M, self.M, self.active_indices
```

#### 3.3.2 熵统计追踪器

```python
# enlighten/memory/entropy_tracker.py - 熵统计
class EntropyTracker:
    """
    实时计算注意力熵滑动统计
    - 均值 μ_H
    - 方差 σ_H²
    - 趋势 k_H (上升/下降)
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.entropy_history = deque(maxlen=window_size)

    def update(self, attention_weights: Tensor):
        """计算并记录当前时刻的注意力熵"""
        entropy = -torch.sum(
            attention_weights * torch.log(attention_weights + 1e-10),
            dim=-1
        ).mean()
        self.entropy_history.append(entropy.item())

    def get_statistics(self) -> Dict[str, float]:
        """获取滑动统计"""
        history = torch.tensor(self.entropy_history)

        return {
            "mean": history.mean().item(),      # μ_H
            "variance": history.var().item(),   # σ_H²
            "trend": self._compute_trend(history),  # k_H
            "current": history[-1].item() if len(history) > 0 else 0
        }

    def _compute_trend(self, history: Tensor) -> float:
        """计算趋势 (简单线性回归斜率)"""
        if len(history) < 2:
            return 0.0
        x = torch.arange(len(history)).float()
        return torch.polyfit(x, history, 1)[0].item()
```

### Phase 4：L3 元控制器重构 (第9-11周)

#### 3.4.1 熵监控与截断决策

```python
# enlighten/l3_controller.py - L3元控制器
class L3MetaController:
    """
    L3 元注意力控制器 (前额叶模拟)

    输入:
    - L2的熵统计 (μ_H, σ_H², k_H)
    - VAN事件
    - 任务嵌入

    输出:
    - 温度 τ
    - 稀疏阈值 θ
    - DMN系数 α
    - 稳定性标志 s
    - 截断信号 cutoff
    """

    def __init__(self, config: L3Config):
        self.config = config
        self.stability_tracker = StabilityTracker()

    def forward(self, entropy_stats: Dict[str, float],
                van_event: bool, task_embedding: Tensor) -> Dict:
        # 获取熵统计
        mu_h = entropy_stats["mean"]
        sigma_h = entropy_stats["variance"]
        k_h = entropy_stats["trend"]

        # 检查VAN事件 - 立即截断
        if van_event:
            return {
                "tau": 0.1,  # 降低温度，快速收敛
                "theta": 0.9,  # 提高阈值，只保留最重要token
                "alpha": 0.5,  # 增强DMN抑制
                "s": False,  # 不稳定
                "cutoff": True,
                "reason": "VAN event triggered"
            }

        # 截断判据检查
        should_cutoff = self.check_cutoff_criteria(
            mu_h, sigma_h, k_h
        )

        if should_cutoff:
            return {
                "tau": self.adjust_tau_for_cutoff(mu_h),
                "theta": self.adjust_theta(),
                "alpha": self.adjust_alpha(),
                "s": False,
                "cutoff": True,
                "reason": "Self-referential loop detected"
            }

        # 正常调控
        return self.normal_control(task_embedding)

    def check_cutoff_criteria(self, mu_h: float,
                               sigma_h: float,
                               k_h: float) -> bool:
        """
        截断判据:
        - 低注意力熵 (μ_H < 0.5)
        - 低方差 (σ_H < 0.05)
        - 持续下降趋势 (k_H < 0)
        """
        return (mu_h < self.config.entropy_threshold and
                sigma_h < self.config.variance_threshold and
                k_h < 0)
```

### Phase 5：审计系统增强 (第12-13周)

#### 3.5.1 哈希链与HMAC签名

```python
# enlighten/audit/chain.py - 审计哈希链
class AuditHashChain:
    """
    哈希链: 每个条目包含前一条的哈希
    提供不可篡改的审计日志
    """

    def __init__(self, algorithm: str = "sha256"):
        self.algorithm = algorithm
        self.chain = []  # [(index, hash, data), ...]

    def append(self, data: Dict) -> str:
        """追加新条目，返回其哈希"""
        # 计算当前数据的哈希
        current_hash = self._hash_data(data)

        # 如果不是第一个条目，链接到前一个
        if self.chain:
            prev_hash = self.chain[-1][1]
            link_hash = self._hash_link(prev_hash, current_hash)
        else:
            link_hash = current_hash

        entry = (len(self.chain), link_hash, data)
        self.chain.append(entry)

        return link_hash

    def verify(self) -> bool:
        """验证链的完整性"""
        for i in range(1, len(self.chain)):
            _, expected_prev_hash, _ = self.chain[i]
            actual_prev_hash = self.chain[i-1][1]

            if expected_prev_hash != actual_prev_hash:
                return False  # 链被篡改

        return True
```

#### 3.5.2 离线复盘服务

```python
# enlighten/audit/offline_review.py - 离线复盘
class OfflineReviewService:
    """
    离线复盘服务:
    1. 读取审计日志
    2. 读取L2快照
    3. 生成自然语言报告
    """

    def __init__(self, l2_model):
        self.l2_model = l2_model

    def generate_report(self, session_id: str) -> str:
        """生成会话复盘报告"""
        # 读取日志
        logs = self.audit_logger.get_session_logs(session_id)
        snapshots = self.l2_model.get_snapshots(session_id)

        # 分析截断事件
        cutoff_events = self._analyze_cutoffs(logs)

        # 分析注意力模式
        attention_patterns = self._analyze_attention(snapshots)

        # 生成报告
        report = self._format_report(
            session_id=session_id,
            cutoff_events=cutoff_events,
            attention_patterns=attention_patterns,
            stats=self._compute_stats(logs)
        )

        return report

    def _format_report(self, **kwargs) -> str:
        """格式化自然语言报告"""
        return f"""
        === EnlightenLM 会话复盘报告 ===

        会话ID: {kwargs['session_id']}

        一、截断事件分析
        {'; '.join(kwargs['cutoff_events'])}

        二、注意力模式分析
        {'; '.join(kwargs['attention_patterns'])}

        三、统计摘要
        - 总token数: {kwargs['stats']['total_tokens']}
        - 截断次数: {kwargs['stats']['cutoff_count']}
        - 平均注意力熵: {kwargs['stats']['avg_entropy']:.4f}

        报告生成时间: {datetime.now().isoformat()}
        """
```

### Phase 6：测试与文档 (第14周)

#### 3.6.1 测试覆盖

| 测试类型 | 测试内容 | 覆盖率目标 |
|---------|---------|-----------|
| 单元测试 | DAN/VAN/记忆/熵追踪/截断 | 90% |
| 集成测试 | L1→L2→L3完整流程 | 80% |
| 性能测试 | 延迟/显存/吞吐量 | - |
| 安全测试 | 截断有效性/审计完整性 | 100% |

#### 3.6.2 文档

- `docs/architecture.md` - 完整架构设计
- `docs/math_verification.md` - 数学证明
- `docs/integration_guide.md` - 集成指南
- `docs/api_reference.md` - API参考

---

## 四、数学基础验证清单

根据新版设计，需要验证以下数学性质：

### 4.1 工作记忆稀疏注意力

- [ ] 复杂度分析: \(O(n \cdot m \cdot d)\) 当 m << n 时远低于 \(O(n^2 d)\)
- [ ] 近似误差界: m=512 时 \(\varepsilon < 0.1\)

### 4.2 渐进双流稳定性

- [ ] 稳定性标志 s 控制 DAN/VAN 计算频率
- [ ] 稳定时仅单流，节省 40% FLOPs

### 4.3 元控制动态调节

- [ ] 温度 τ 的几何意义
- [ ] 稀疏截断 θ 的约束条件

### 4.4 DMN与遗忘门

- [ ] LayerNorm归一化确保扰动不累积
- [ ] 遗忘门提供指数衰减的KV缓存

### 4.5 截断判据

- [ ] 低注意力熵 (μ_H < 0.5) + 低方差 (σ_H < 0.05) + 持续时间
- [ ] 结合VAN事件，有效识别自反循环

---

## 五、性能预期

| 指标 | 标准Transformer 7B | EnlightenLM | 变化 |
|------|-------------------|-------------|------|
| 每token延迟 | 40ms | 56ms | +40% |
| 显存 | 14GB | 16GB | +2GB |
| 长上下文注意力 | \(O(n^2)\) | \(O(n \cdot 512)\) | 大幅降低 |
| 可审计性 | 无 | 密码级 | 新增 |
| 安全截断 | 无 | 实时 | 新增 |

---

## 六、风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| 双流注意力FLOPs增加 | 延迟+40% | 渐进式执行(稳定时单流) |
| vLLM深度集成难度 | 开发周期 | Phase 3专注研究，Phase 4引入专家 |
| 网络问题导致模型加载失败 | 无法测试 | 已实现mock回退机制 |
| 数学证明不完整 | 设计可信度 | 分模块验证，逐步完善 |

---

## 七、里程碑

| 里程碑 | 完成时间 | 交付物 |
|--------|---------|--------|
| M1: 目录结构重组 | 第2周末 | 新项目结构、可运行框架 |
| M2: L1双流注意力 | 第5周末 | DAN+VAN实现、基础测试 |
| M3: L2工作记忆 | 第8周末 | 稀疏注意力、熵追踪器 |
| M4: L3元控制 | 第11周末 | 熵监控、截断决策 |
| M5: 审计增强 | 第13周末 | 离线复盘服务 |
| M6: 测试文档 | 第14周末 | 完整测试套件、文档 |

---

*文档版本: v1.0*
*更新日期: 2026-04-23*
*维护者: EnlightenLM Team*
