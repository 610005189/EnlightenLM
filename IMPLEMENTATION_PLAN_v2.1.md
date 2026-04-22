# EnlightenLM v2.1 实现计划

> 基于 README v2.1 的完整实现任务清单
> 版本: v2.1.0
> 更新日期: 2026-04-23
> 状态: T1-T8已完成 ✅ | T9进行中 ⏳ | T10待执行 ⏳

---

## 目录

1. [实现概述](#1-实现概述)
2. [任务详情](#2-任务详情)
3. [优先级排序](#3-优先级排序)
4. [依赖关系](#4-依赖关系)
5. [时间估算](#5-时间估算)

---

## 1. 实现概述

### 1.1 核心功能清单

| 功能模块 | 描述 | 优先级 | 状态 |
|---------|------|--------|------|
| 配置模式系统 | full/balanced/lightweight 三种预设模式 | P0 | ✅ 已完成 |
| VAN 三级漏斗 | light/medium/full 三个级别 | P0 | ✅ 已完成 |
| L2 可配置刷新 | 滑动窗口 + 定期刷新策略 | P1 | ✅ 已完成 |
| L3 冷却机制 | 截断抖动防止 | P1 | ✅ 已完成 |
| 异步审核系统 | 独立进程 1.5B 审核模型 | P2 | ✅ 已完成 |
| 运行时切换 | 热切换运行模式 | P1 | ✅ 已完成 |
| 配置文件 | 三个预设模式 YAML | P0 | ✅ 已完成 |
| 性能测试 | 基准测试验证 | P2 | ✅ 已完成 |

### 1.2 性能目标

| 模式 | 延迟增加 | 显存增加 | 重复率 |
|------|---------|---------|--------|
| **full** | +15% | +1.5GB | <2% |
| **balanced** | +10% | +1GB | <3% |
| **lightweight** | +5% | +0.5GB | <5% |

---

## 2. 任务详情

### T1: 配置模式系统 [P0]

**负责人**: 开发者
**预计时间**: 2-3 天

#### 子任务

- [ ] **T1.1** 创建模式枚举类
```python
# enlighten/config/modes.py
class EnlightenMode(Enum):
    FULL = "full"
    BALANCED = "balanced"
    LIGHTWEIGHT = "lightweight"
```

- [ ] **T1.2** 创建模式配置类
```python
# enlighten/config/modes.py
@dataclass
class ModeConfig:
    mode: EnlightenMode
    van_level: str  # "light" | "medium" | "full"
    gate_fusion: bool
    dmn_noise: bool
    use_topk_refresh: bool
    refresh_interval: int
    async_review_enabled: bool
```

- [ ] **T1.3** 创建模式预设
```python
MODE_PRESETS = {
    "full": ModeConfig(
        mode=EnlightenMode.FULL,
        van_level="full",
        gate_fusion=True,
        dmn_noise=True,
        use_topk_refresh=True,
        refresh_interval=32,
        async_review_enabled=True
    ),
    # ... balanced, lightweight
}
```

- [ ] **T1.4** 环境变量覆盖支持
```python
# 环境变量
ENLIGHTEN_MODE=lightweight
ENLIGHTEN_VAN_LEVEL=light
ENLIGHTEN_WORKING_MEMORY_CAPACITY=256
```

#### 验收标准

```python
# 测试用例
config = load_config(mode="balanced")
assert config.mode == EnlightenMode.BALANCED
assert config.van_level == "medium"
assert config.gate_fusion == True
```

---

### T2: VAN 三级漏斗机制 [P0]

**负责人**: 开发者
**预计时间**: 3-4 天

#### 子任务

- [ ] **T2.1** 关键词匹配器（所有模式）
```python
# enlighten/attention/van.py
class KeywordMatcher:
    """关键词匹配器 - VAN Level light"""
    def __init__(self, keywords: List[str]):
        self.keywords = set(keywords)

    def detect(self, tokens: List[int]) -> bool:
        """检测敏感词"""
        # 返回是否触发VAN事件
        pass
```

- [ ] **T2.2** 轻量 MLP 分类器（medium/full）
```python
# enlighten/attention/van.py
class LightweightMLPClassifier:
    """轻量MLP分类器 - VAN Level medium/full"""
    def __init__(self, vocab_size: int, hidden_dim: int = 128):
        self.classifier = nn.Sequential(
            nn.Embedding(vocab_size, hidden_dim),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def predict(self, hidden_states: Tensor) -> float:
        """返回有害概率 p_harm"""
        pass
```

- [ ] **T2.3** 完整注意力检查（full 模式）
```python
# enlighten/attention/van.py
class FullAttentionChecker:
    """完整注意力检查 - VAN Level full"""
    def __init__(self, embed_dim: int, num_heads: int):
        self.attention = MultiheadAttention(embed_dim, num_heads)

    def check(self, hidden_states: Tensor) -> Tuple[float, bool]:
        """返回 p_harm 和是否触发"""
        pass
```

- [ ] **T2.4** VAN 漏斗协调器
```python
# enlighten/attention/van.py
class VANFunnel:
    """VAN 三级漏斗协调器"""
    def __init__(self, level: str):
        self.level = level  # "light" | "medium" | "full"
        self.keyword_matcher = KeywordMatcher()
        self.mlp_classifier = LightweightMLPClassifier() if level != "light" else None
        self.full_checker = FullAttentionChecker() if level == "full" else None

    def forward(self, tokens, hidden_states) -> Tuple[float, bool]:
        """
        返回 (p_harm, van_event)
        """
        # Level light: 关键词匹配
        # Level medium: + MLP分类器
        # Level full: + 完整注意力
        pass
```

#### 验收标准

```python
# 测试用例
van = VANFunnel(level="medium")

# light级别 - 只有关键词匹配
van_light = VANFunnel(level="light")
p, event = van_light.forward(tokens, hidden)
assert event == True  # 如果有敏感词

# medium级别 - 关键词 + MLP
p, event = van.forward(tokens, hidden)
assert 0 <= p <= 1  # p_harm 在 [0, 1]

# full级别 - 额外完整注意力
van_full = VANFunnel(level="full")
p, event = van_full.forward(tokens, hidden)
# 更精确的检测
```

---

### T3: L2 可配置刷新策略 [P1]

**负责人**: 开发者
**预计时间**: 2-3 天

#### 子任务

- [ ] **T3.1** 滑动窗口刷新（始终启用）
```python
# enlighten/memory/working_memory.py
class SlidingWindowRefresh:
    """滑动窗口刷新策略"""
    def refresh(self, memory, new_tokens, active_indices):
        # 始终启用，超出容量丢弃最旧的非敏感token
        pass
```

- [ ] **T3.2** 定期 Top-K 刷新
```python
# enlighten/memory/working_memory.py
class TopkRefresh:
    """基于注意力得分的定期刷新"""
    def __init__(self, interval: int = 32):
        self.interval = interval
        self.step_counter = 0

    def should_refresh(self) -> bool:
        self.step_counter += 1
        return self.step_counter >= self.interval

    def refresh(self, memory, attention_scores):
        # 基于 attention_scores 重新计算重要性
        # 替换低分 token
        pass
```

- [ ] **T3.3** VAN 敏感 token 保护
```python
# enlighten/memory/working_memory.py
def protect_sensitive_tokens(self, indices: Set[int]):
    """VAN 标记的敏感 token 永不淘汰"""
    self.sensitive_indices = indices
```

#### 验收标准

```python
# 测试用例
memory = WorkingMemory(capacity=512, use_topk_refresh=True, refresh_interval=32)

# 模拟 100 步
for step in range(100):
    memory.update(tokens[step], attention_scores[step])

    if step % 32 == 0:
        # 应该触发刷新
        assert memory.refresh_triggered

# 敏感词保护
memory.mark_sensitive([10, 20, 30])
memory.evict_oldest()
assert 10 in memory.active_indices  # 敏感词未被淘汰
```

---

### T4: L3 冷却机制增强 [P1]

**负责人**: 开发者
**预计时间**: 1-2 天

#### 子任务

- [ ] **T4.1** 截断冷却计数器
```python
# enlighten/l3_controller.py
class L3Controller:
    def __init__(self):
        self.cooldown_steps = 10
        self.cooldown_counter = 0

    def forward(self, entropy_stats, van_event):
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return ControlSignals(...)  # 返回默认信号
```

- [ ] **T4.2** 抖动检测与抑制
```python
# enlighten/l3_controller.py
def _detect_flickering(self, history: List[bool]) -> bool:
    """检测截断信号是否在抖动"""
    # 如果最近 N 次决策反复横跳，返回 True
    pass
```

#### 验收标准

```python
# 测试用例
controller = L3Controller(cooldown_steps=10)

# 触发截断后，10步内不应再次截断
controller.forward(entropy_stats, van_event=True)  # 触发
assert controller.cooldown_counter == 10

for _ in range(9):
    result = controller.forward(entropy_stats, False)
    assert result.cutoff == False  # 冷却中

result = controller.forward(entropy_stats, False)
assert result.cutoff == False  # 冷却结束
```

---

### T5: 异步审核系统 [P2]

**负责人**: 开发者
**预计时间**: 3-4 天

#### 子任务

- [ ] **T5.1** 审核进程管理器
```python
# enlighten/async_review.py
class AsyncReviewManager:
    """异步审核进程管理器"""
    def __init__(self, model_name: str, interval: int = 32):
        self.model_name = model_name
        self.interval = interval
        self.review_queue = Queue()
        self.review_process = None

    def start(self):
        """启动独立审核进程"""
        pass

    def submit_for_review(self, session_id: str, content: str):
        """提交审核任务"""
        self.review_queue.put((session_id, content))

    def get_review_result(self, session_id: str) -> Optional[ReviewResult]:
        """获取审核结果"""
        pass
```

- [ ] **T5.2** 审核结果写入审计日志
```python
# enlighten/async_review.py
class ReviewResult:
    session_id: str
    factuality_score: float
    safety_score: float
    issues: List[str]
    timestamp: float
```

#### 验收标准

```python
# 测试用例
manager = AsyncReviewManager(
    model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    interval=32
)
manager.start()

# 提交审核
manager.submit_for_review("session_123", "生成内容...")

# 获取结果
result = manager.get_review_result("session_123")
assert result.factuality_score >= 0
assert result.factuality_score <= 1
```

---

### T6: 配置文件 [P0]

**负责人**: 配置管理员
**预计时间**: 0.5 天

#### 子任务

- [ ] **T6.1** configs/full.yaml
```yaml
enlighten:
  mode: "full"

  components:
    van_stream:
      level: "full"  # light | medium | full
    gate_fusion: true
    dmn_noise: true

    working_memory:
      capacity: 512
      refresh_interval: 32
      use_topk_refresh: true

    entropy_monitor:
      window_size: 20
    cutoff:
      low_entropy_threshold: 0.5
      low_variance_threshold: 0.05
      min_duration: 5
      van_threshold: 0.9

    async_review:
      enabled: true
      model: "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
      interval: 32
```

- [ ] **T6.2** configs/balanced.yaml
```yaml
enlighten:
  mode: "balanced"

  components:
    van_stream:
      level: "medium"
    gate_fusion: true
    dmn_noise: false

    working_memory:
      capacity: 512
      refresh_interval: 32
      use_topk_refresh: true

    entropy_monitor:
      window_size: 20
    cutoff:
      low_entropy_threshold: 0.5
      low_variance_threshold: 0.05
      min_duration: 5
      van_threshold: 0.9

    async_review:
      enabled: false
```

- [ ] **T6.3** configs/lightweight.yaml
```yaml
enlighten:
  mode: "lightweight"

  components:
    van_stream:
      level: "light"
    gate_fusion: false
    dmn_noise: false

    working_memory:
      capacity: 256
      refresh_interval: 0  # 不刷新
      use_topk_refresh: false

    entropy_monitor:
      window_size: 20
    cutoff:
      low_entropy_threshold: 0.5
      low_variance_threshold: 0.05
      min_duration: 5
      van_threshold: 0.9

    async_review:
      enabled: false
```

- [ ] **T6.4** configs/custom.yaml.example
```yaml
# 自定义配置示例
enlighten:
  mode: "custom"  # 或指定 preset

  components:
    # 参考 full.yaml 全部选项
```

#### 验收标准

```bash
# 测试用例
ls configs/
# 应包含: full.yaml, balanced.yaml, lightweight.yaml, custom.yaml.example
```

---

### T7: 运行时模式切换 [P1]

**负责人**: 开发者
**预计时间**: 1-2 天

#### 子任务

- [ ] **T7.1** 热切换方法
```python
# enlighten/main.py
class EnlightenLM:
    def set_mode(self, mode: str):
        """
        热切换运行模式
        会重置 L2 和 L3 的状态
        """
        if mode not in ["full", "balanced", "lightweight"]:
            raise ValueError(f"Unknown mode: {mode}")

        self.config = load_config(mode=mode)
        self.l2.reset()  # 重置工作记忆
        self.l3.reset()  # 重置冷却计数器
        logger.info(f"Switched to {mode} mode")
```

#### 验收标准

```python
# 测试用例
model = EnlightenLM.from_pretrained(config="balanced.yaml")

# 生成一些内容
model.generate("Hello")

# 热切换到 lightweight
model.set_mode("lightweight")
assert model.config.mode == "lightweight"

# 确认状态已重置
assert model.l3.cooldown_counter == 0
```

---

### T8: 性能基准测试 [P2]

**负责人**: 测试工程师
**预计时间**: 2-3 天

#### 子任务

- [ ] **T8.1** 延迟基准测试
```python
# tests/benchmark/test_latency.py
def benchmark_latency(mode: str, num_runs: int = 100):
    """测量每种模式的平均延迟"""
    pass

# 预期结果
"""
| 模式 | 延迟增加 |
|------|---------|
| full | +15% |
| balanced | +10% |
| lightweight | +5% |
"""
```

- [ ] **T8.2** 显存基准测试
```python
# tests/benchmark/test_memory.py
def benchmark_memory(mode: str):
    """测量每种模式的显存占用"""
    pass
```

- [ ] **T8.3** 重复率测试
```python
# tests/benchmark/test_repetition.py
def test_repetition_rate(mode: str) -> float:
    """测试生成内容的重复率"""
    pass

# 预期结果
"""
| 模式 | 重复率 |
|------|--------|
| full | <2% |
| balanced | <3% |
| lightweight | <5% |
"""
```

---

### T9: 文档一致性 [P2]

**负责人**: 文档管理员
**预计时间**: 1 天

#### 子任务

- [ ] **T9.1** README 与实现一致性检查
```python
# scripts/check_consistency.py
def check_readme_implementation_match():
    """检查 README 中的描述与实际实现是否一致"""
    checks = [
        ("VAN 三级漏斗", "enlighten.attention.van.VANFunnel"),
        ("L2 可配置刷新", "enlighten.memory.working_memory.TopkRefresh"),
        # ...
    ]
```

- [ ] **T9.2** API 文档更新
- [ ] **T9.3** 示例代码验证

---

## 3. 优先级排序

### 第一阶段：核心功能 (P0)

| 优先级 | 任务 | 依赖 | 预计时间 |
|--------|------|------|---------|
| P0-1 | T1: 配置模式系统 | - | 2-3 天 |
| P0-2 | T2: VAN 三级漏斗 | T1 | 3-4 天 |
| P0-3 | T6: 配置文件 | T1 | 0.5 天 |

### 第二阶段：重要功能 (P1)

| 优先级 | 任务 | 依赖 | 预计时间 |
|--------|------|------|---------|
| P1-1 | T3: L2 可配置刷新 | T1, T2 | 2-3 天 |
| P1-2 | T4: L3 冷却机制 | T1 | 1-2 天 |
| P1-3 | T7: 运行时切换 | T1, T6 | 1-2 天 |

### 第三阶段：增强功能 (P2)

| 优先级 | 任务 | 依赖 | 预计时间 |
|--------|------|------|---------|
| P2-1 | T5: 异步审核系统 | T1, T2 | 3-4 天 |
| P2-2 | T8: 性能基准测试 | T1-T7 | 2-3 天 |
| P2-3 | T9: 文档一致性 | T1-T8 | 1 天 |

---

## 4. 依赖关系

```
T1 (配置模式系统)
├── T2 (VAN 三级漏斗)
├── T3 (L2 可配置刷新)
├── T4 (L3 冷却机制)
└── T6 (配置文件)
    └── T7 (运行时切换)

T5 (异步审核系统) ──┬── T1
                    └── T2

T8 (性能基准测试) ───┬── T1
                    ├── T2
                    ├── T3
                    ├── T4
                    ├── T5
                    ├── T6
                    └── T7

T9 (文档一致性) ─────┴── T1-T8
```

---

## 5. 时间估算

### 总体估算

| 阶段 | 任务 | 预计时间 |
|------|------|---------|
| 第一阶段 | T1 + T2 + T6 | 5.5-7.5 天 |
| 第二阶段 | T3 + T4 + T7 | 4-7 天 |
| 第三阶段 | T5 + T8 + T9 | 6-8 天 |
| **总计** | **所有任务** | **15.5-22.5 天** |

### 里程碑

| 里程碑 | 完成任务 | 预计时间 |
|--------|---------|---------|
| M1: MVP | T1 + T2 + T6 | 第 2 周 |
| M2: Beta | T1-T7 完成 | 第 3-4 周 |
| M3: RC | 所有任务 + 初步测试 | 第 5 周 |
| M4: Release | 文档 + 最终测试 | 第 6 周 |

---

## 6. 验收标准清单

### 功能验收

- [ ] 三种模式可切换
- [ ] VAN 三级漏斗正常工作
- [ ] L2 刷新策略可配置
- [ ] L3 冷却机制防止抖动
- [ ] 异步审核可独立运行
- [ ] 配置文件格式正确

### 性能验收

- [ ] lightweight: 延迟 +5%, 显存 +0.5GB, 重复率 <5%
- [ ] balanced: 延迟 +10%, 显存 +1GB, 重复率 <3%
- [ ] full: 延迟 +15%, 显存 +1.5GB, 重复率 <2%

### 文档验收

- [ ] README 与实现一致
- [ ] API 文档完整
- [ ] 示例代码可运行
- [ ] CHANGELOG 更新

---

*计划版本: v1.0*
*创建日期: 2026-04-23*
*维护者: EnlightenLM Team*
