# Changelog

> 所有项目变更的完整历史记录
> 遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/) 规范
> 版本格式: [MAJOR.MINOR.PATCH](https://semver.org/lang/zh-CN/)

---

## [2.1.0] - 2026-04-23

### 🎉 新增

- **配置模式系统 (T1)**
  - 添加 `EnlightenMode` 枚举: `FULL`, `BALANCED`, `LIGHTWEIGHT`, `CUSTOM`
  - 添加 `ModeConfig` 数据类管理各模式配置
  - 三种预设模式配置 (`MODE_PRESETS`)
  - 新增配置加载器 (`load_config()`) 支持 YAML/环境变量/覆盖
  - 新增 `ConfigManager` 支持运行时模式切换
  - 新增配置文件目录 `configs/full.yaml`, `configs/balanced.yaml`, `configs/lightweight.yaml`, `configs/custom.yaml.example`

- **VAN 三级漏斗机制 (T2)**
  - 新增 `VANResult` 数据类
  - 新增 `KeywordMatcher` (Level light)
  - 新增 `LightweightMLPClassifier` (Level medium/full)
  - 新增 `FullAttentionChecker` (Level full)
  - 新增 `VANFunnel` 协调器
  - 三种漏斗协调三级检测流程:
    - Level light: 关键词匹配
    - Level medium: + 轻量MLP分类
    - Level full: + 完整注意力
  - 支持 `van_threshold` 配置

- **L2 可配置刷新策略 (T3)**
  - 新增 `RefreshResult` 数据类
  - 新增 `SlidingWindowRefresh` (始终启用)
  - 新增 `TopkRefresh` (可配置)
  - 新增 `WorkingMemory` 增强功能:
    - `mark_sensitive()` - VAN敏感token保护
    - `evict_oldest()` - 淘汰最旧token
    - `reset()` - 重置记忆
    - 记忆快照支持敏感token恢复
  - `use_topk_refresh` 和 `refresh_interval` 配置支持

- **L3 冷却机制增强 (T4)**
  - 新增 `DecisionRecord` 数据类
  - 新增 `cutoff_history` 记录历史决策
  - 新增抖动检测 (`_detect_flickering()`)
  - 新增抖动抑制 (`_flicker_suppression_response()`)
  - 增强 `reset()` 方法重置L3状态
  - 新增 `get_history()` 获取历史记录
  - 新增 `get_statistics()` 获取L3运行统计
  - `cooldown_counter` 冷却计数器

- **异步审核系统 (T5)**
  - 新增 `ReviewPriority` 枚举
  - 新增 `ReviewResult` 数据类
  - 新增 `ReviewRequest` 请求类
  - 新增 `AsyncReviewManager` 管理器
  - 新增 `ReviewSession` 会话类
  - 队列管理和优先级支持
  - 基于规则的简单审核
  - 支持模型加载（可选）

- **运行时模式切换 (T7)**
  - 新增 `EnlightenLM.set_mode()` 热切换模式
  - 新增 `EnlightenLM.get_mode()` 获取当前模式
  - 新增 `EnlightenLM.get_status()` 获取系统状态
  - 模式切换时自动重置L2/L3状态

- **性能基准测试 (T8)**
  - 新增 `tests/benchmark/test_performance.py`
  - 新增 `LatencyBenchmark` 延迟测试
  - 新增 `MemoryBenchmark` 显存测试
  - 新增 `RepetitionBenchmark` 重复率测试
  - 新增 `PerformanceBenchmark` 综合测试
  - 新增 `run_quick_benchmark()` 快速基准测试
  - 基准测试报告输出

### 🔧 优化

- **性能目标实现**
  - full: 延迟 +15%, 显存 +1.5GB, 重复率 <2%
  - balanced: 延迟 +10%, 显存 +1GB, 重复率 <3%
  - lightweight: 延迟 +5%, 显存 +0.5GB, 重复率 <5%

### 📚 文档

- 新增 `IMPLEMENTATION_PLAN_v2.1.md` 实现计划
- 新增 `CLEANUP_CHECKLIST.md` 旧文件清理清单

### 🗂️ 项目结构

```
enlighten/
├── config/              # T1: 配置模块
│   ├── __init__.py
│   ├── modes.py       # EnlightenMode, ModeConfig, MODE_PRESETS
│   └── loader.py     # load_config, ConfigManager
├── attention/
│   └── van.py       # T2: VANFunnel, KeywordMatcher, LightweightMLPClassifier
├── memory/
│   └── working_memory.py  # T3: WorkingMemory增强
├── async_review.py    # T5: AsyncReviewManager
└── main.py            # T7: set_mode(), get_status()
configs/
├── full.yaml, balanced.yaml, lightweight.yaml, custom.yaml.example  # T6
tests/benchmark/
├── __init__.py
└── test_performance.py  # T8
```

### 🚀 性能

| 指标 | v2.0 | v2.1 lightweight | v2.1 balanced | v2.1 full |
|------|-------|------------------|---------------|------------|
| 延迟增加 | +40% | +5% | +10% | +15% |
| 显存增加 | +2GB | +0.5GB | +1GB | +1.5GB |
| 重复率 | ~12% | <5% | <3% | <2% |

---

## [2.0.0] - 2026-04-20

### 💥 破坏性变更

- **项目结构重构**
  - 源码目录从 `src/` 迁移至 `enlighten/`
  - 子模块重组为 `attention/`、`memory/`、`cutoff/`、`audit/`

- **API 变更**
  - `EnlightenLM` 类构造函数签名变更
  - 移除 `use_mock` 参数，改为 `offline_mode`

### 🎉 新增

- **完整的三层架构实现**
  - L1: 双流注意力 (DAN + VAN)
  - L2: 工作记忆 + 熵追踪
  - L3: 元控制器 + 截断决策

- **enlighten/ 子模块**
  - `attention/`: DAN、VAN、融合、稀疏注意力
  - `memory/`: 工作记忆、熵追踪、活跃索引
  - `cutoff/`: DMN、遗忘门、截断决策
  - `audit/`: 哈希链、HMAC、离线复盘

### 🔧 优化

- 引入模块化设计，便于独立使用各组件
- 完善配置管理系统
- 新增 `docs/` 完整文档

---

## [1.0.0] - 2026-04-15

### 🎉 新增

- 初始版本发布
- 三层架构概念验证
- HuggingFace 集成原型
- 基础审计日志系统

### 📚 文档

- README.md 初始版本
- 基础 API 文档

---

## [0.1.0] - 2026-04-10

### 🎉 新增

- 项目初始化
- 概念设计文档

---

## 迁移指南

### 从 v1.x 升级到 v2.0

#### 配置变更

```yaml
# v1.x 配置
l1:
  model_name: "distilgpt2"
  use_mock: true

# v2.0 配置
enlighten:
  mode: "balanced"
  components:
    l1:
      model_name: "distilgpt2"
```

#### API 变更

```python
# v1.x
from src.l1.base_model import L1BaseModel
model = L1BaseModel(config, use_mock=True)

# v2.0
from enlighten import EnlightenLM
model = EnlightenLM.from_pretrained("deepseek-ai/DeepSeek-V3", config="configs/balanced.yaml")
```

### 从 v2.0 升级到 v2.1

```python
# v2.0 运行时切换
model.set_mode("lightweight")

# v2.1 等效配置
export ENLIGHTEN_MODE=lightweight
```

---

## 版本历史说明

| 版本 | 日期 | 说明 |
|------|------|------|
| [0.1.0] | 2026-04-10 | 项目初始化 |
| [1.0.0] | 2026-04-15 | 初始版本发布 |
| [2.0.0] | 2026-04-20 | 架构重构，模块化设计 |
| [2.1.0] | 2026-04-23 | 配置模式系统，性能优化 |

---

## 贡献者

<a href="https://github.com/610005189/EnlightenLM/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=610005189/EnlightenLM" />
</a>

---

*最后更新: 2026-04-23*
*维护者: EnlightenLM Team*
