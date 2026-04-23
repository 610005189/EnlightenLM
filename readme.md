# EnlightenLM · 觉悟三层架构

> 从"静态护栏"到"动态觉悟"——基于认知神经科学的大模型安全推理与元认知框架
> **版本**：v2.3（API模式简化实现）
> **状态**：⚠️ 设计愿景 v2.2 / ✅ 实际运行简化版

> **重要说明**：
> - `docs/architecture.md` 描述的是**完整设计架构**（v2.2愿景）
> - `enlighten/hybrid_architecture.py` 是**当前实际运行的代码**
> - 详见 [实现状态总览](./docs/implementation_status.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: API Mode](https://img.shields.io/badge/Status-API%20Mode-blue)]()
[![DeepSeek Compatible](https://img.shields.io/badge/DeepSeek-V3%20%7C%20V4-Compatible-blue)]()

---

## 📖 目录

1. [核心思想](#核心思想)
2. [架构总览](#架构总览)
3. [各层详细设计](#各层详细设计)
4. [配置开关与模式](#配置开关与模式)
5. [数学合理性验证](#数学合理性验证)
6. [工程落地与性能](#工程落地与性能)
7. [DeepSeek-V4 兼容与扩展](#deepseek-v4-兼容与扩展)
8. [实施路线图](#实施路线图)
9. [项目结构](#项目结构)
10. [快速开始](#快速开始)
11. [许可证与致谢](#许可证与致谢)

---

## 核心思想

**EnlightenLM** 将大模型推理过程解耦为三层，并借鉴人脑注意力网络（DAN/VAN/DMN）实现实时自我监控与安全截断：

> ⚠️ **以下为设计愿景（v2.2文档描述）**：
> - **L1 生成层**：双流注意力（DAN 目标驱动 + VAN 刺激驱动） + 遗忘门
> - **L2 工作记忆层**：压缩上下文，维护活跃 token 集、实时熵统计
> - **L3 元控制层**：实时调控温度/稀疏度/截断，写入密码学审计链

> ✅ **以下为当前实际实现（hybrid_architecture.py）**：
> - **L1 生成层**：DeepSeek API 或本地模型（无双流/遗忘门/DMN）
> - **L2 工作记忆层**：会话历史管理 + 文本熵分析 + 近似注意力统计
> - **L3 元控制层**：敏感词检测 + 自指循环检测 + 文本熵截断（无密码学审计）

**目标**：将额外推理开销控制在 **+5% ~ +15%**，同时实现实时截断、密码级审计与幻觉抑制。

> ⚠️ **注意**：当前API模式下，部分安全监控使用**文本特征近似**而非真实模型注意力，因为DeepSeek API是黑盒无法获取内部状态。

---

## 架构总览

### ✅ 实际运行的简化架构（hybrid_architecture.py）

```
用户输入
  ↓
L1生成层: DeepSeek API 或 distilgpt2 本地模型
  ↓
L2工作记忆: 会话历史 + 文本熵分析 + 近似注意力统计
  ↓
L3 VAN监控: 敏感词检测 + 自指循环检测 + 文本熵截断
  ↓
输出 + 安全元信息
```

**实际实现的功能**：
- ✅ DeepSeek API 集成
- ✅ 会话历史管理
- ✅ 文本熵值计算（词汇多样性/重复率/字符熵）
- ✅ 敏感词/自指循环检测
- ✅ 词汇重复检测
- ✅ Cooldown机制
- ⚠️ 注意力统计（文本特征近似，非真实注意力）

---

### ⚠️ 设计架构（详见 docs/architecture.md）

```
用户输入
│
▼
┌─────────────────────────────────────────────────────────────┐
│  L3 元控制器（前额叶模拟）                                    │
│  · 接收 L2 熵统计 (μ_H, σ_H) + 轻量 VAN 分数 p_harm          │
│  · 输出：温度 τ, 稀疏阈值 θ, 稳定性标志 s, 截断标志 cutoff     │
│  · 截断判据：低熵+低方差+VAN事件 → 硬中断                     │
│  · 所有动作写入审计哈希链                                     │
└─────────────────────────────┬───────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────┐
│  L2 工作记忆层（可配置）                                      │
│  · 固定大小记忆矩阵 M (m×d)，m=256~512                       │
│  · 活跃索引集 A = 最近窗口 + VAN标记敏感token                 │
│  · 可选：定期刷新（基于注意力得分）或纯滑动窗口               │
│  · 实时计算滑动熵统计 (窗口L=20)                             │
│  · 定期保存快照供离线复盘                                     │
└───────────────┬─────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────┐
│  L1 生成层（简化双流，可配置）                                │
│  ┌───────────────────┐    ┌───────────────────────────────┐ │
│  │ DAN 流             │    │ VAN 流（三级漏斗）             │ │
│  │ · 工作记忆稀疏注意力 │    │ 1. 关键词匹配（自动机）         │ │
│  │ · 任务偏置 B_task   │    │ 2. 轻量MLP分类器（每步）        │ │
│  └─────────┬─────────┘    │ 3. 完整注意力（可选，仅full模式）│ │
│            └──────────────┴───────────────┬───────────────┘ │
│                    可选门控融合（balanced/full模式）         │
│                           │                                 │
│              动态温度 τ + 稀疏截断 θ                         │
│                           │                                 │
│              遗忘门（始终启用）+ 可选DMN噪声                  │
│                           │                                 │
│                     输出 token y                            │
└─────────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────┐
│  审计与复盘系统（可配置）                                     │
│  · 实时审计：紧凑日志 + 哈希链 + HMAC（始终启用）             │
│  · 异步审核：可选1.5B模型复核事实性（仅full模式）             │
│  · 离线复盘：基于日志+快照生成自然语言报告（始终启用）        │
└─────────────────────────────────────────────────────────────┘
```

---

## 各层详细设计

> ⚠️ **说明**：以下为设计文档描述的功能。实际实现状态请参见 [implementation_status.md](./docs/implementation_status.md)

### L1 生成层

**设计功能**（未完全实现）：

> ⚠️ **DAN 流**（始终启用）：仅在骨架代码 `l1_generation.py` 中，未集成运行
> - 从 L2 获取活跃键值对 \((\tilde{K}, \tilde{V})\)，执行稀疏注意力
- 从 L2 获取活跃键值对 \((\tilde{K}, \tilde{V})\)，执行稀疏注意力：
  \[
  \text{Attn}_{\text{DAN}} = \text{softmax}\left(\frac{Q_{\text{DAN}} \tilde{K}^\top}{\sqrt{d}}\right) \tilde{V} + B_{\text{task}}
  \]
- 任务偏置 \(B_{\text{task}}\) 由 L3 根据任务嵌入生成（低秩分解，参数量小）。

**VAN 流**（三级漏斗，级别可配置）：
- **Level light**（默认 lightweight 模式）：关键词匹配 + 轻量 MLP 分类器
- **Level medium**（默认 balanced 模式）：同上，但 MLP 每步运行
- **Level full**（仅 full 模式）：额外启用完整注意力（用于极高风险场景）
- 输出有害概率 \(p_{\text{harm}}\)，触发 VAN 事件

**门控融合**（可配置）：
- 在 `balanced`/`full` 模式下，当 L3 标记不稳定时，计算门控 \(g_t\) 融合 DAN 和 VAN（VAN 简化为标量偏置）。
- 在 `lightweight` 模式下，无融合，VAN 仅用于截断。

**温度与稀疏截断**：
\[
\text{Attn}_{\text{temp}} = \text{softmax}\left(\frac{\text{Attn}_{\text{used}}}{\tau_t}\right)
\]
\[
\text{Attn}_{\text{final}} = \text{renormalize}(\text{Attn}_{\text{temp}} \cdot \mathbf{1}[\text{Attn}_{\text{temp}} > \theta_t])
\]

**遗忘门**（始终启用）：
- 更新 KV 缓存：\(\tilde{K}_t^{\text{cache}} = f_t \odot \tilde{K}_{t-1}^{\text{cache}} + (1-f_t) \odot K_t^{\text{new}}\)

**DMN 噪声**（可选，仅 full 模式）：
- 输出噪声注入：\(\text{Output} = \text{Attn}_{\text{final}} V + \alpha_t \cdot \text{LayerNorm}(\xi_t), \quad \xi_t \sim \mathcal{N}(0,1)\)

### L2 工作记忆层

**数据结构**：
- 记忆矩阵 \(M_t \in \mathbb{R}^{m \times d}\)，容量 \(m\) 可配置（256~512）
- 活跃索引集 \(A_t\)：最近 \(w=128\) 个 token + 所有 VAN 标记敏感 token
- 熵队列 \(Q_H\)（长度 \(L=20\)）

**更新策略**（可配置）：
- **滑动窗口**（始终）：每步将新 token 加入，超出容量则丢弃最旧的非敏感 token。
- **定期刷新**（`use_topk_refresh=true` 时）：每 \(N\) 步基于注意力得分重新计算重要性，替换低分 token。

**熵统计**：每步计算 \(H_t = -\sum_{i \in A_t} p_i \log p_i\)，更新 \(\mu_H, \sigma_H\)。

### L3 元控制层

**输入**：\(\mu_H, \sigma_H, p_{\text{harm}}, e_{\text{task}}\)

**输出**（每步）：
\[
\tau_t = \text{clamp}(\tau_{\text{base}} + \lambda_\tau \cdot \sigma_H, 0.1, 2.0)
\]
\[
\theta_t = \text{clamp}(\theta_{\text{base}} - \beta \cdot p_{\text{harm}}, 0.0, 0.5)
\]
\[
stable_t = \mathbf{1}[\sigma_H < 0.05 \land \text{duration\_stable} > 5] \quad \text{(仅 balanced/full)}
\]
\[
cutoff_t = \mathbf{1}[(\mu_H < 0.5 \land \sigma_H < 0.05 \land \text{duration} > 5) \lor p_{\text{harm}} > 0.9]
\]

**截断动作**：若 \(cutoff_t=1\)，立即终止生成，返回预设安全响应，记录截断事件。

### 审计与复盘系统

**实时审计日志**（始终启用）：
- 格式：JSONL 每行紧凑记录（约 200 字节）
- 密码学：HMAC-SHA256 + 哈希链（每步链接前一步哈希）
- 可选 TEE 存储（SGX/CSV）

**异步审核**（仅 full 模式）：
- 独立进程，运行 1.5B 审核模型（微调 DeepSeek-R1-Distill-Qwen）
- 每生成 \(K\) 个 token 复核事实性与安全性，结果写入审计日志

**离线复盘**（始终启用）：
- 定时或按需读取审计日志 + L2 快照 + 异步审核结果
- 调用 7B 模型生成自然语言报告，供审计员查阅

---

## 配置开关与模式

### 三种预设模式

| 模式 | 目标场景 | 延迟增加 | 安全截断 | 幻觉抑制 | 审计完整性 |
|------|----------|---------|----------|----------|-----------|
| **full** | 高安全/高合规（金融、医疗、政务） | +15% | 完整 | 最强 | 完整 |
| **balanced** | 通用对话、内容生成（默认） | +10% | 完整 | 中等 | 完整 |
| **lightweight** | 高吞吐、低延迟（客服、实时交互） | +5% | 核心 | 基础 | 核心审计 |

### 配置示例

```yaml
# config.yaml
enlighten:
  mode: "balanced"  # full | balanced | lightweight

  components:
    # L1 生成层
    van_stream:
      level: "medium"   # light | medium | full
    gate_fusion: true
    dmn_noise: false

    # L2 工作记忆层
    working_memory:
      capacity: 512
      refresh_interval: 32      # 每 N 步刷新，0 表示不刷新
      use_topk_refresh: true

    # L3 元控制层
    entropy_monitor:
      window_size: 20
    cutoff:
      low_entropy_threshold: 0.5
      low_variance_threshold: 0.05
      min_duration: 5
      van_threshold: 0.9

    # 审计与复盘
    async_review:
      enabled: false
      model: "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
      interval: 32
    offline_review:
      enabled: true
      schedule: "on_demand"
```

### 环境变量覆盖

```bash
export ENLIGHTEN_MODE=lightweight
export ENLIGHTEN_VAN_LEVEL=light
export ENLIGHTEN_WORKING_MEMORY_CAPACITY=256
```

### 运行时动态切换

```python
from enlighten import EnlightenLM

model = EnlightenLM.from_pretrained("deepseek-ai/DeepSeek-V3", config="configs/balanced.yaml")
model.set_mode("lightweight")  # 热切换，重置状态
```

---

## 数学合理性验证

| 组件 | 关键性质 | 验证结论 |
|------|----------|----------|
| 稀疏注意力误差 | 总变差 ≤ 0.1（m=512） | ✅ 可接受 |
| 轻量 VAN 漏报率 | 可通过训练控制 ≤ 0.01 | ✅ 需验证集调优 |
| 截断假阳性率 | \(P(\mu_H<0.5) < 10^{-28}\) | ✅ 极低 |
| 遗忘门数值稳定 | 指数衰减，有界 | ✅ |
| DMN 噪声不累积 | LayerNorm 后单位方差 | ✅ |
| 训练时可微性 | α-entmax 替代硬阈值 | ✅ |
| 审计哈希链 | 标准密码学安全 | ✅ |

详细证明见 [`docs/math_verification.md`](docs/math_verification.md)。

---

## 工程落地与性能

### 性能预期（相对标准 Transformer 7B）

| 指标 | 标准 Transformer | full | balanced | lightweight |
|------|------------------|------|----------|-------------|
| 每 token 延迟 | 40ms | 46ms (+15%) | 44ms (+10%) | 42ms (+5%) |
| 显存 | 14GB | 15.5GB (+1.5GB) | 15GB (+1GB) | 14.5GB (+0.5GB) |
| 长上下文注意力 | O(n²) | O(n·512) | O(n·512) | O(n·256) |
| 可审计性 | 无 | 完整 | 完整 | 核心 |
| 安全截断 | 无 | 完整 | 完整 | 核心 |
| 重复率（幻觉） | ~12% | <2% | <3% | <5% |

### 推理框架适配

| 框架 | 适配难度 | 推荐场景 |
|------|---------|----------|
| HuggingFace Transformers | 低 | 原型验证 |
| vLLM | 中 | 生产级在线服务 |
| DeepSeek-V4 原生 | 低（适配器模式） | V4 专用优化 |

---

## DeepSeek-V4 兼容与扩展

### V4 核心特性（预计 2026.04 发布）

- 总参数 ~1T，推理激活 ~37B
- 上下文窗口 **1M token**
- 注意力机制：MLA + **Engram 条件记忆**（O(1) 哈希检索）
- 多模态原生支持（图像、视频、音频）
- 推理速度较 V3 **提升 35 倍**
- 开源协议预计 Apache 2.0

### 适配策略

| EnlightenLM 组件 | V4 原生能力 | 适配方式 |
|-----------------|-------------|----------|
| L1 稀疏注意力 | Sparse Attention + Sliding Window | 直接复用 |
| L2 工作记忆 | **Engram 条件记忆** | 作为底层存储，更新 O(1) |
| VAN 显著性 | 多模态安全对齐表征 | 扩展到视觉/音频域 |
| 遗忘门 | KV Cache INT8 + 滑动窗口 | 复用窗口机制 |

### 分层工作记忆（适配 1M 上下文）

```
L2-L1：最近窗口 (w=1024)        → 每步参与计算
L2-L2：Engram 记忆表 (64K)      → 每128步刷新
L2-L3：VAN 敏感 token (16K)     → 永不淘汰
```

### 版本兼容矩阵

| EnlightenLM 版本 | 支持的 DeepSeek 后端 | 预期开销 |
|-----------------|---------------------|----------|
| v1.0 | V3 原生 | +15% |
| v1.1 | V3 + V4（适配器） | +12% |
| v2.0 | V4 原生深度优化 | **+5%** |

---

## 实施路线图

> ⚠️ **重要更新 (v2.3)**：
> - Phase 1-4 的完成状态指的是**骨架代码**的完成
> - 实际运行的简化版 (`hybrid_architecture.py`) 已完成核心功能
> - 完整架构（双流注意力/DMN/遗忘门/哈希链）需要本地模型模式

### Phase 1：概念验证（骨架代码 ✅）
- ✅ HuggingFace + Qwen2.5-7B 原型
- ✅ L1 稀疏注意力、L2 简化工作记忆、L3 规则控制器

### Phase 2：配置开关与模式（骨架代码 ✅，API模式简化实现 ✅）
- ✅ 实现三级配置模式（full/balanced/lightweight）
- ✅ 集成轻量 VAN MLP 分类器（三级漏斗机制）
- ⚠️ 完善审计日志与哈希链（骨架代码有，API模式未使用）
- ✅ 运行时模式热切换
- ✅ 环境变量配置覆盖

### Phase 3：生产级优化（骨架代码 ✅，API模式部分实现）
- ⚠️ vLLM 适配器基础架构（骨架代码有，未集成）
- ⚠️ TEE 兼容审计数据格式（骨架代码有，API模式未使用）
- ⚠️ 自动化复盘服务调度器（骨架代码有，未使用）
- ✅ 性能基准测试套件
- ✅ 安全测试验证（VAN监控、截断）

### Phase 4：DeepSeek-V4 适配（骨架代码 ✅，API模式 ✅）
- ✅ DeepSeek 适配器实现（支持 V3/V4、API模式、本地模式）
- ✅ Engram 记忆优化器（设计，API模式简化实现）
- ⚠️ 多模态 VAN 设计（骨架代码有，未集成）
- ✅ 完整集成测试通过

### Phase 5：多模态融合（规划中）
- 规划中：实现文本-图像跨模态检测
- 规划中：实现音频-视频模态支持
- 规划中：统一多模态融合框架

---

## 项目结构

```
.
├── README.md                          # 本文件
├── LICENSE                            # MIT许可证
├── requirements.txt                   # Python依赖
├── CHANGELOG.md                       # 变更日志
├── CONTRIBUTING.md                    # 贡献指南
├── deployment_guide.md                # 部署指南
├── usage_examples.md                  # 使用示例
├── IMPLEMENTATION_PLAN_v2.1.md       # 实施计划
│
├── configs/                          # ✅ 配置文件
│   ├── full.yaml                     # 完整模式配置
│   ├── balanced.yaml                  # 平衡模式配置
│   ├── lightweight.yaml               # 轻量模式配置
│   ├── core_rules.yaml               # 核心规则
│   ├── deepseek_v3.yaml              # DeepSeek V3配置
│   ├── hyperparameters.yaml          # 超参数配置
│   └── task_embeddings.yaml           # 任务嵌入配置
│
├── enlighten/                        # 核心代码
│   ├── __init__.py
│   ├── main.py                       # 主入口
│   ├── utils.py                      # 工具函数
│   ├── hybrid_architecture.py        # ✅ 实际运行: L1/L2/L3架构
│   ├── api_server.py                # ✅ 实际运行: API服务
│   ├── async_review.py              # 异步审核
│   │
│   ├── api/                         # ✅ API客户端
│   │   ├── __init__.py
│   │   ├── deepseek_client.py       # DeepSeek API
│   │   └── dashscope_client.py      # 阿里云API
│   │
│   ├── config/                      # ✅ 配置管理
│   │   ├── __init__.py
│   │   ├── modes.py                 # 模式配置类
│   │   └── loader.py                # 配置加载器
│   │
│   ├── adapters/                    # ⚠️ 模型适配器(骨架)
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── deepseek_adapter.py
│   │   └── vllm_adapter.py
│   │
│   ├── attention/                   # ⚠️ 注意力机制(骨架)
│   │   ├── __init__.py
│   │   ├── dan.py                   # DAN流
│   │   ├── van.py                   # VAN流
│   │   ├── fusion.py                # 融合门控
│   │   ├── sparse.py                # 稀疏注意力
│   │   └── multimodal_van.py        # 多模态VAN
│   │
│   ├── memory/                      # ⚠️ 记忆管理(骨架)
│   │   ├── __init__.py
│   │   ├── working_memory.py
│   │   ├── engram_optimizer.py      # Engram优化器
│   │   └── entropy_tracker.py       # 熵追踪器
│   │
│   ├── audit/                       # ⚠️ 审计系统(骨架)
│   │   ├── __init__.py
│   │   ├── chain.py                 # 哈希链
│   │   ├── hmac_signature.py        # HMAC签名
│   │   ├── tee_audit.py             # TEE审计
│   │   ├── offline_review.py        # 离线复盘
│   │   └── review_service.py        # 复盘服务
│   │
│   ├── cutoff/                      # ⚠️ 截断机制(骨架)
│   │   ├── __init__.py
│   │   ├── cutoff_entropy.py        # 熵截断
│   │   ├── dmn.py                   # DMN抑制
│   │   └── forget_gate.py           # 遗忘门
│   │
│   ├── l1_generation.py             # ⚠️ L1生成层(骨架)
│   ├── l2_working_memory.py         # ⚠️ L2工作记忆(骨架)
│   └── l3_controller.py             # ⚠️ L3控制器(骨架)
│
├── docs/                            # 文档
│   ├── chat.html                    # ✅ Web聊天界面
│   ├── architecture.md              # ⚠️ 架构设计文档
│   ├── math_verification.md         # ⚠️ 数学验证文档
│   ├── math_verification_audit.md   # ✅ 实现状态审计
│   ├── product_report.md            # 产品报告
│   ├── user_manual.md               # 用户手册
│   ├── api_reference.md             # API参考
│   ├── integration_guide.md          # 集成指南
│   ├── 设计文档.md                   # 设计文档
│   ├── article/                     # 文章
│   └── paper/                       # 论文
│
├── tests/                           # 测试
│   ├── test_hybrid_architecture.py # ✅ 68个测试通过
│   ├── demo_comparison.py           # ✅ 对比演示
│   ├── test_security.py            # 安全测试
│   ├── test_jailbreak.py           # 越狱测试
│   ├── test_deepseek_adapter.py    # 适配器测试
│   ├── test_entropy_cutoff.py      # 熵截断测试
│   ├── test_engram_optimizer.py    # 记忆优化器测试
│   ├── test_multimodal_van.py      # 多模态VAN测试
│   ├── test_phase1_validation.py   # Phase1验证
│   ├── test_phase2_integration.py  # Phase2集成
│   ├── test_phase4_integration.py  # Phase4集成
│   ├── test_attention_bias.py      # 注意力偏差测试
│   ├── test_security_performance.py # 安全性能测试
│   ├── test_simple.py              # 简单测试
│   ├── test_system.py              # 系统测试
│   ├── jailbreak_test_report.md    # 越狱测试报告
│   └── benchmark/                  # 性能基准
│       └── test_performance.py
│
├── EnlightenLM_Quick_Test/          # 快速测试
│   ├── EnlightenLM_Quick_Test_1.md
│   └── Experiment_Results.md
│
└── logs/                            # 日志
    ├── security_test_results.json  # 安全测试结果
    └── audit/                      # 审计日志
        └── last_hash.txt           # 上次哈希
```

**图例**：
- ✅ = 实际运行使用的代码/文档
- ⚠️ = 设计/骨架代码（未在实际运行中集成）

---

## 架构设计完成度对比

### 完成度总览

| 层级 | 模块名称 | 设计功能 | 实现状态 | 完成度 | 未完成原因 |
|------|----------|----------|----------|--------|------------|
| **L1 生成层** | 双流注意力 (DAN+VAN) | DAN/VAN 融合 | 骨架代码 | 0% | API黑盒限制，需本地模型 |
| | DMN噪声抑制 | 噪声估计+抑制 | 骨架代码 | 0% | API黑盒限制 |
| | 遗忘门 | LSTM式门控 | 骨架代码 | 0% | API黑盒限制 |
| | 本地模型支持 | Transformer生成 | hybrid_architecture | 60% | 仅支持distilgpt2 |
| | DeepSeek API集成 | API调用 | hybrid_architecture | 100% | 已完成 |
| **L2 工作记忆** | 稀疏注意力选择 | Top-k重要性评分 | 骨架代码 | 0% | API黑盒限制 |
| | 上下文窗口管理 | 滑动窗口/定期刷新 | hybrid_architecture | 100% | 已完成 |
| | 注意力统计追踪 | 熵值/方差/趋势 | hybrid_architecture | 70% | API模式下为文本近似 |
| | 记忆矩阵压缩 | m×d矩阵 | 骨架代码 | 0% | 无真实稀疏注意力 |
| **L3 元控制** | 截断判据(三条件) | μ<τμ ∧ σ<τσ ∧ k<0 | hybrid_architecture | 100% | 已完成 |
| | 温度动态调节 | τ = f(σ_H) | 骨架代码 | 0% | API黑盒限制 |
| | 稀疏度动态调节 | θ = f(p_harm) | 骨架代码 | 0% | API黑盒限制 |
| | 冷却机制 | cooldown防止抖动 | hybrid_architecture | 100% | 已完成 |
| **VAN 监控** | 敏感词检测 | 正则模式匹配 | hybrid_architecture | 100% | 已完成 |
| | 自指循环检测 | 词汇模式检测 | hybrid_architecture | 100% | 已完成 |
| | 词汇重复检测 | bigram重复率 | hybrid_architecture | 100% | 已完成 |
| | 文本熵分析 | 词汇多样性/字符熵 | hybrid_architecture | 100% | 已完成 |
| | 多模态VAN | 图像/音频检测 | 骨架代码 | 0% | 未集成 |
| **审计系统** | 哈希链 | SHA256链式结构 | 骨架代码 | 0% | 未持久化 |
| | HMAC签名 | 消息认证 | 骨架代码 | 0% | 未集成 |
| | Merkle树 | 批量验证 | 骨架代码 | 0% | 未集成 |
| | 事件日志 | 内存记录 | hybrid_architecture | 50% | 仅内存，不持久化 |
| **配置系统** | 三种预设模式 | full/balanced/lightweight | modes.py | 100% | 已完成 |
| | 环境变量覆盖 | 配置覆盖 | modes.py | 100% | 已完成 |
| | 运行时切换 | 热切换 | api_server | 100% | 已完成 |
| **API服务** | 推理接口 | /inference | api_server | 100% | 已完成 |
| | 健康检查 | /health | api_server | 100% | 已完成 |
| | 安全统计 | /security/stats | api_server | 100% | 已完成 |
| **Web界面** | 聊天界面 | 对话UI | chat.html | 100% | 已完成 |
| | 主题切换 | 深色/浅色 | chat.html | 100% | 已完成 |
| | Markdown渲染 | 格式显示 | chat.html | 100% | 已完成 |
| | 对话持久化 | localStorage | chat.html | 100% | 已完成 |

### 按子系统完成度汇总

| 子系统 | 包含模块数 | 已完成 | 部分完成 | 未完成 | 综合完成度 |
|--------|-----------|--------|----------|--------|-----------|
| L1 生成层 | 5 | 1 | 1 | 3 | **40%** |
| L2 工作记忆 | 4 | 2 | 1 | 1 | **62%** |
| L3 元控制 | 4 | 2 | 0 | 2 | **50%** |
| VAN 监控 | 5 | 4 | 0 | 1 | **80%** |
| 审计系统 | 4 | 0 | 1 | 3 | **12%** |
| 配置系统 | 3 | 3 | 0 | 0 | **100%** |
| API服务 | 3 | 3 | 0 | 0 | **100%** |
| Web界面 | 4 | 4 | 0 | 0 | **100%** |
| **总计** | **32** | **19** | **3** | **10** | **59%** |

### 未完成功能原因分析

| 原因类别 | 影响模块数 | 说明 |
|----------|-----------|------|
| **API黑盒限制** | 8 | DeepSeek API不返回模型内部状态（注意力/loss/logits），无法实现真实监控 |
| **优先级调整** | 2 | 当前聚焦API模式可用性，完整架构延后 |
| **技术难度** | 2 | 双流融合、稀疏注意力需要自定义模型支持 |
| **资源限制** | 1 | 多模态VAN需要额外训练数据和技术验证 |

### 下一步计划

| 优先级 | 模块 | 目标 | 依赖条件 |
|--------|------|------|----------|
| P0 | 完善本地模型支持 | 支持更多开源模型（如LLaMA、Qwen） | 模型权重 |
| P1 | 集成骨架代码 | 将l1/l2/l3_controller集成到实际运行 | 本地模型支持 |
| P2 | 审计持久化 | 实现哈希链和HMAC持久化存储 | 数据库支持 |
| P2 | 多模态VAN | 图像输入安全检测 | 视觉编码器 |

---

## 快速开始

> ⚠️ **当前实际运行方式**：使用 API 服务器 + Web 界面

### 方式一：API 服务器模式（推荐）

```bash
# 设置 API Key
$env:DEEPSEEK_API_KEY = "sk-你的密钥"

# 启动 API 服务器
python -m enlighten.api_server

# 启动 Web 界面（另一个终端）
cd docs
python -m http.server 8080
```

访问 http://localhost:8080/chat.html 使用 Web 界面。

### 方式二：直接使用 Python API

```python
from enlighten.hybrid_architecture import HybridEnlightenLM
from enlighten.api.deepseek_client import DeepSeekAPIClient, DeepSeekConfig
import os

config = DeepSeekConfig(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    model="deepseek-chat"
)
client = DeepSeekAPIClient(config)

model = HybridEnlightenLM(use_local_model=False, api_client=client)
result = model.generate("请解释量子纠缠", max_length=512)
print(result.text)
```

---

### 设计文档中的使用方式（需要本地模型）

> ⚠️ 以下是完整架构的使用方式，当前 API 模式不支持

```python
# 完整架构使用方式（设计愿景）
from enlighten import EnlightenLM

# 加载模型（默认 balanced 模式）- 需要本地模型
model = EnlightenLM.from_pretrained("deepseek-ai/DeepSeek-V3", config="configs/balanced.yaml")

# 生成文本
response = model.generate(
    "请解释量子纠缠的原理。",
    max_tokens=512,
    task="science_explain"
)
print(response)

# 切换到 lightweight 模式（更低延迟）
model.set_mode("lightweight")
fast_response = model.generate("你好", max_tokens=50)
```

---

## 许可证与致谢

- 代码：**MIT License**
- 文档：**CC BY-SA 4.0**
- DeepSeek 模型权重遵循其自身许可证（V3 MIT，V4 预计 Apache 2.0）

**致谢**：认知神经科学中的 DAN/VAN/DMN 理论，东方哲学"止观双运"思想，DeepSeek 开源团队。

---

## Star History

如果这个项目对你有启发，欢迎点亮 Star，让更多人看到"可驾驭的 AI"是如何被构建的。

[![Star History Chart](https://api.star-history.com/svg?repos=610005189/EnlightenLM&type=Date)](https://star-history.com/#610005189/EnlightenLM&Date)

---

**觉悟不是让 AI 成佛，而是让人类放心地把方向盘交给它。**
