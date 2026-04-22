# 旧文件清理清单

> 本文档列出完成验证后需要删除的旧文件
> 仅在所有新功能验证通过后执行清理
> **最后更新**: 2026-04-23 (T1-T8已完成)

---

## 🗑️ 待删除文件清单

### Phase 1: MVP 完成后删除 (T1-T3验证通过)

| 文件路径 | 说明 | 验证状态 |
|---------|------|---------|
| `enlighten/config.py` | 旧版配置管理类 (已被 `enlighten/config/modes.py` + `loader.py` 替代) | ✅ 已验证可删除 |
| `src/l1/base_model.py` | 旧L1基座模型 (已重构到 `enlighten/l1_generation.py`) | ✅ 已验证可删除 |
| `src/l1/__init__.py` | | ✅ 已验证可删除 |
| `src/l2/self_description_model.py` | 旧L2自描述模型 (已重构到 `enlighten/l2_working_memory.py`) | ✅ 已验证可删除 |
| `src/l2/__init__.py` | | ✅ 已验证可删除 |
| `src/l3/meta_attention_controller.py` | 旧L3元控制器 (已重构到 `enlighten/l3_controller.py`) | ✅ 已验证可删除 |
| `src/l3/__init__.py` | | ✅ 已验证可删除 |

### Phase 2: Beta 完成后删除 (T4-T7验证通过)

| 文件路径 | 说明 | 验证状态 |
|---------|------|---------|
| `src/audit/` | 旧审计系统 (已重构到 `enlighten/audit/`) | ✅ 已验证可删除 |
| `src/main.py` | 旧主入口 (已重构到 `enlighten/main.py`) | ✅ 已验证可删除 |
| `src/utils.py` | 旧工具函数 | ✅ 已验证可删除 |
| `src/__init__.py` | | ✅ 已验证可删除 |

### Phase 3: Release 准备时删除

| 文件路径 | 说明 | 验证状态 |
|---------|------|---------|
| `readme - old1.md` | 旧版 README 备份 | ✅ 已验证可删除 |
| `readme - old2.md` | 更旧的 README 备份 | ✅ 已验证可删除 |
| `REFACTORING_PLAN.md` | 已过期的重构计划 (已被 `IMPLEMENTATION_PLAN_v2.1.md` 替代) | ✅ 已验证可删除 |

---

## 🔍 验证检查清单

在删除任何文件前，确保以下验证通过：

### 配置系统验证 ✅

```python
from enlighten.config import EnlightenMode, ModeConfig, load_config, get_mode_preset

# ✅ 新配置系统
config = load_config(mode="full")
assert config.van_level == "full"
assert config.mode == EnlightenMode.FULL

# ✅ 预设模式
preset = get_mode_preset("balanced")
assert preset.gate_fusion == True
```

### 模块导入验证 ✅

```python
# ✅ 新模块
from enlighten.config import EnlightenMode, ModeConfig, load_config
from enlighten.attention.van import VANFunnel
from enlighten.memory import WorkingMemory, SlidingWindowRefresh, TopkRefresh, RefreshResult
from enlighten.l3_controller import L3Controller, DecisionRecord
from enlighten.async_review import AsyncReviewManager, ReviewResult, ReviewPriority

# ✅ EnlightenLM主类支持set_mode
from enlighten.main import EnlightenLM
```

### 配置文件验证 ✅

```bash
ls configs/
# 应包含: full.yaml, balanced.yaml, lightweight.yaml, custom.yaml.example
```

### 性能基准测试验证 ✅

```python
# ✅ 性能基准测试模块
from tests.benchmark import run_quick_benchmark
```

---

## ⚠️ 删除命令（谨慎执行）

### Phase 1 删除命令:

```bash
# 删除旧config.py
rm enlighten/config.py

# 删除旧src/l1目录
rm -rf src/l1

# 删除旧src/l2目录
rm -rf src/l2

# 删除旧src/l3目录
rm -rf src/l3
```

### Phase 2 删除命令:

```bash
# 删除旧src/audit目录
rm -rf src/audit

# 删除旧src/main.py和utils.py
rm src/main.py
rm src/utils.py
rm src/__init__.py
```

### Phase 3 删除命令:

```bash
# 删除旧README备份
rm "readme - old1.md"
rm "readme - old2.md"

# 删除过期计划文件
rm REFACTORING_PLAN.md
```

---

## 📋 风险评估

| 文件/目录 | 删除风险 | 缓解措施 |
|-----------|---------|---------|
| `src/l1/` | 中 | 确保 `enlighten/l1_generation.py` 已完整 |
| `src/l2/` | 中 | 确保 `enlighten/l2_working_memory.py` 已完整 |
| `src/l3/` | 中 | 确保 `enlighten/l3_controller.py` 已完整 |
| `enlighten/config.py` | 低 | 新 `enlighten/config/` 模块已完整 |
| `readme - old*.md` | 极低 | 仅文档备份 |

---

## ✅ 当前进度

| 阶段 | 完成任务 | 状态 |
|------|---------|------|
| Phase 1 | T1-T3 + 验证 | ✅ 完成 |
| Phase 2 | T4-T7 + 验证 | ✅ 完成 |
| Phase 3 | T8-T9 + 验证 | ⏳ 进行中 |
| 清理执行 | T10 | ⏳ 待执行 |

**核心实现已完成**:
- ✅ T1: 配置模式系统
- ✅ T2: VAN三级漏斗机制
- ✅ T3: L2可配置刷新策略
- ✅ T4: L3冷却机制增强
- ✅ T5: 异步审核系统
- ✅ T6: 配置文件
- ✅ T7: 运行时模式切换
- ✅ T8: 性能基准测试
- ⏳ T9: README和文档一致性
- ⏳ T10: 旧文件清理

---

*最后更新: 2026-04-23*
*维护者: EnlightenLM Team*