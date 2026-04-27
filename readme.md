# EnlightenLM

**模型无关的三层推理安全框架 —— 实时监控、幻觉检测与偏见缓解**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-51%20passed-green)]()
[![Docker](https://img.shields.io/badge/Docker-ready-blue)]()

> **核心理念**：不推翻主流架构，而是通过在模型外围构建元认知监控层，为任何大模型注入"自知之明"——让模型在推理时实时觉察自身的幻觉与偏见风险，并主动干预。

---

## 评估指标

### 核心性能指标

| 指标 | 说明 | 目标值 | 测量方法 |
|------|------|--------|----------|
| **幻觉检测率** | 正确识别并截断幻觉内容的比例 | >85% | TruthfulQA / HaluEval 测试集 |
| **偏见拦截率** | 正确拦截偏见性内容的比例 | >90% | BBQ 偏见测试集 |
| **误报率** | 正常内容被错误截断的比例 | <5% | 随机采样正常对话 |
| **VAN 事件检测** | 敏感词/自指检测准确率 | >95% | 合成测试集 |
| **响应延迟增量** | 安全监控带来的额外延迟 | <20% | 端到端延迟对比 |
| **截断循环率** | 触发冷却机制的比例 | <10% | 生产环境统计 |

### 安全监控指标

| 指标 | 说明 | 测量方法 |
|------|------|----------|
| **截断率** | 被截断的生成请求比例 | 截断次数 / 总请求数 |
| **VAN 事件率** | 触发 VAN 安全监控的比例 | VAN 事件数 / 总请求数 |
| **冷却激活率** | 触发智能冷却机制的比例 | 冷却次数 / 总截断次数 |
| **平均响应长度** | 正常输出的平均 token 数 | 总输出 token / 成功请求数 |
| **审计链完整率** | 哈希链验证通过的比例 | 验证通过数 / 验证总数 |

### 风险等级分布

| 风险等级 | 条件 | 预期分布 |
|----------|------|----------|
| 🟢 低风险 | p_hall < 0.3，无偏见触发 | 70-80% |
| 🟡 中风险 | 0.3 ≤ p_hall < 0.7 | 15-25% |
| 🔴 高风险 | p_hall ≥ 0.7 或偏见高置信度 | 5-10% |

---

## 目录

- [项目简介](#项目简介)
- [项目结构](#项目结构)
- [架构设计](#架构设计)
- [核心功能](#核心功能)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [API参考](#api参考)
- [测试](#测试)
- [下一步计划](#下一步计划)
- [贡献指南](#贡献指南)
- [相关资源](#相关资源)
- [许可证](#许可证)

---

## 项目简介

EnlightenLM 是一个**模型无关**的三层推理安全框架。它不修改底层大模型的权重，而是通过在模型外部（API 侧）构建 L2 工作记忆层和 L3 元控制层，实现对生成过程的实时安全监控、幻觉截断与偏见缓解。

当前版本已实现完整的 L3 安全监控体系（敏感词检测、自指循环检测、词汇重复检测、智能冷却机制、元认知自检）、L2 会话历史与统计特征提取、贝叶斯病因推断器（区分"先验偏见"与"感知噪声"）、Engram锚点和rewind机制（记忆状态保存和回退）、Merkle树审计验证、密码学审计链、轻量级 MLP 幻觉判别器（基于规则的实时幻觉风险预测）。项目支持 Ollama、DeepSeek API 等多种模型后端，提供 REST API 和 Web 界面，已通过 51+ 个单元测试。

---

## 项目结构

```
EnlightenLM/
├── static/                 # Web UI 资源
│   └── chat.html          # 聊天界面（默认首页）
├── configs/                # 配置文件
│   ├── balanced.yaml       # 平衡模式配置
│   ├── full.yaml          # 完整模式配置
│   ├── lightweight.yaml    # 轻量模式配置
│   └── core_rules.yaml     # 核心规则配置
├── deploy/                  # 部署配置
│   ├── docker-compose.yml  # Docker 部署
│   ├── Dockerfile          # Docker 镜像
│   ├── grafana/           # Grafana 仪表板配置
│   ├── k8s/               # Kubernetes 配置
│   ├── nginx/              # Nginx 配置
│   └── prometheus/         # Prometheus 配置
├── docs/                   # 项目文档
│   ├── architecture_v3.0.md    # 架构设计
│   ├── api_reference.md         # API 参考
│   ├── admin_guide.md          # 管理员指南
│   ├── user_manual.md          # 用户手册
│   ├── security.md             # 安全体系
│   ├── implementation_status.md # 实现状态
│   ├── ADR-001-model-agnostic-approach.md  # 架构决策
│   ├── archives/               # 归档文档
│   └── plans/                  # 实施计划
├── enlighten/               # 核心源代码
│   ├── api_server.py      # FastAPI 服务
│   ├── main.py            # 主入口
│   ├── metacognition.py    # 元认知模块
│   ├── l2_working_memory.py  # L2 工作记忆
│   ├── l3_controller.py   # L3 控制器
│   ├── adapters/          # 模型适配器
│   ├── api/              # API 客户端
│   ├── attention/        # 注意力机制
│   ├── audit/            # 审计模块
│   ├── cutoff/           # 截断模块
│   ├── interfaces/       # 模型接口
│   └── memory/           # 记忆模块
├── tests/                 # 测试文件 (51+)
│   ├── test_*.py         # 单元测试
│   └── benchmark/         # 性能测试
├── train/                 # 训练脚本
├── readme.md             # 项目文档
├── requirements.txt      # Python 依赖
└── pyproject.toml        # 项目配置
```

---

## 设计理念

### 技术路线声明

EnlightenLM 坚持**模型无关的外围监控路线**，不修改底层大模型的权重，而是通过在模型外部（API 侧）构建监控层实现安全能力。项目已移除所有内部修改代码（如双流注意力实现），明确专注于通用、可插拔的外围监控架构。

### 问题背景

当前大语言模型在实用化部署中普遍面临以下核心问题，EnlightenLM 通过外围监控架构系统性解决：

| 问题 | 具体表现 | EnlightenLM 解决方案 |
|------|----------|---------------------|
| **事实性幻觉** | 模型生成与已知事实不符的内容 | L2 熵监控 + L3 动态截断 |
| **价值性偏见** | 模型输出存在性别、种族等偏见 | VAN 偏见词库 + 探针引导 |
| **安全护栏绕过** | 精心构造的提示词绕过安全限制 | 敏感词检测 + 冷却机制 |
| **自指循环** | 模型无限引用自身输出，耗尽上下文 | 自指循环检测 + 强制截断 |
| **审计缺失** | 输出争议时无法回溯决策依据 | 哈希链 + HMAC签名 + Merkle树验证 |
| **黑盒推理** | 模型无法说明"依据什么得出" | 贝叶斯病因推断 + 熵监控 |

### 核心优势

- **不推翻主流架构**：基于现有模型 API 构建，不依赖特定模型
- **实时监控**：在生成过程中实时检测风险并干预
- **可解释性**：贝叶斯病因推断提供风险归因
- **可审计**：完整生成轨迹记录，支持事后验证（Merkle树）
- **分级干预**：根据风险等级动态调整干预强度（continue / halt / rewind）
- **记忆回退**：Engram锚点支持记忆状态的保存和回退

---

## 架构设计

EnlightenLM 将推理过程划分为三个层次，安全能力完全由 L2/L3 层提供：

```
┌──────────────────────────────────────────────┐
│         L3 元控制层 (Meta-Control)            │
│  · VAN 安全监控 (敏感词/自指/重复/冷却)        │
│  · 贝叶斯病因推断 (偏见 vs 幻觉)               │
│  · 元认知自检 (自动追问 + 自我修正)             │
│  · Engram锚点 (记忆保存/回退)                   │
│  · 动态截断与温度调节                          │
│  · 密码学审计链 (哈希链 + Merkle树)            │
└────────────┬─────────────────────────────────┘
             │
┌────────────▼─────────────────────────────────┐
│         L2 工作记忆层 (Working Memory)         │
│  · 会话历史管理                                │
│  · 文本统计特征提取 (熵/重复率/多样性)           │
│  · 模型 Logits 实时监控 (本地模型模式)           │
│  · 信号自适应预处理 (FFT/Laplace/Z变换)        │
└────────────┬─────────────────────────────────┘
             │
┌────────────▼─────────────────────────────────┐
│         L1 生成层 (Generation)                 │
│  · 透明调用底层模型 (Ollama / DeepSeek / vLLM)  │
│  · 无需修改模型权重                             │
└──────────────────────────────────────────────┘
```

**关键设计原则：**

- **模型无关**：支持任何提供标准 API 的大模型，无需侵入模型内部
- **可插拔**：监控模块可独立开启/关闭，不影响主推理管线
- **开销可控**：安全计算主要为文本特征与轻量级规则，增量开销低
- **智能分级**：根据风险等级动态调整干预强度（continue / halt / rewind）

---

## 核心功能

### 已实现 ✅

- **VAN 安全监控引擎**
  实时检测敏感关键词、自指循环（模型引用自身输出）、词汇过度重复、回答长度异常。触发后自动冷却，防止无限截断循环。

- **贝叶斯病因推断器**
  将安全事件归类为两类病因：
  - 先验偏见（模型固有的倾向性）
  - 感官噪声（输入中的干扰/误导）
  根据不同病因动态调整生成温度与截断信心。

- **连续截断与智能冷却**
  避免"一刀切"截断，支持在置信度低于阈值时连续降低温度，或在多次触发后强制中止。冷却窗口内对同类事件降级处理。

- **元认知自检提示**
  自动追问模型"你是否有偏见？"，支持自动追问机制和自我修正流程，提升模型自我反思能力。

- **混合架构调度**
  统一管理 Ollama、DeepSeek API、vLLM 等多后端，自动切换或级联。支持流式与非流式生成。

- **会话与工作记忆**
  维护对话历史，计算文本统计特征（词汇多样性、重复率、字符熵），作为 L3 的部分输入信号。

- **Engram锚点和rewind机制**
  支持记忆状态的保存和回退，可以在关键时刻创建锚点，需要时回退到之前的状态。

- **审计日志与密码学骨架**
  预置哈希链与 HMAC 签名模块，以及 Merkle树验证模块，可审计每一步生成决策（已集成激活）。

- **Logits 级实时熵监控**
  逐 Token 获取模型输出的 logits，计算后验熵。构建基于 logits 的特征向量（熵、置信度、top-1与top-2概率比），提升幻觉预警精度。

- **偏见检测与缓解**
  集成多领域敏感词库（性别、种族、职业等），基于正则表达式实现高效匹配。触发后追加元认知自检提示。

- **REST API + Web 界面**
  提供 `/inference`、`/health`、`/status`、`/l3/stats` 等接口，以及 WebSocket 实时监控。

- **Docker 部署 + Kubernetes 配置**
  提供完整的 Docker 部署方案，包括 K8s 配置文件（Deployment、Service、ConfigMap、Secret、HPA），支持 sidecar 模式。

- **Prometheus/Grafana 监控**
  集成完整的监控指标，包括请求率、响应时间、VAN 事件、截断率、元认知自检率、Engram使用率等。

- **WebSocket 实时监控**
  实现实时监控数据传输，每 100ms 推送一次监控数据。

- **信号自适应预处理**
  支持状态分类（收敛/发散/离散）和多种变换（FFT/Laplace/Z变换），提升信号处理能力。

- **轻量级 MLP 幻觉判别器**
  基于 8 维特征（熵、置信度、重复率、多样性、字符熵、方差、趋势、干预次数）的实时幻觉风险预测，支持在输入和输出两个阶段进行检测，检测到高风险时自动截断并记录审计日志。可通过配置文件或环境变量控制启用/禁用。

- **MLP 模型训练**
  完成了模型训练流程，包括数据预处理、模型训练和评估。训练好的模型已集成到系统中，用于实时幻觉检测。

### 开发中 🚧

- **Agent 协同验证**（集成多 Agent 辩论框架）

---

## 快速开始

### 环境要求

- Python 3.10+
- Ollama 服务（使用本地模型时）或 DeepSeek API（使用API模式时）

### 1. 克隆仓库

```bash
git clone https://github.com/610005189/EnlightenLM.git
cd EnlightenLM
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

复制示例配置文件并修改：

```bash
cp .env.example .env
```

编辑 `.env`，配置模型选择：

```ini
# -----------------------------------------------------------------------------
# 模型配置 - 本地模式或API模式
# -----------------------------------------------------------------------------
# 使用本地模型
ENLIGHTEN_MODEL_USE_LOCAL=true
ENLIGHTEN_MODEL_LOCAL_NAME=qwen2.5:7b

# 或使用API模式
# ENLIGHTEN_MODEL_USE_LOCAL=false
# ENLIGHTEN_MODEL_API_PROVIDER=ollama
# ENLIGHTEN_MODEL_API_MODEL=qwen2.5:14b
```

### 4. 启动Ollama服务（本地模式）

```bash
ollama serve
ollama pull qwen2.5:7b
```

### 5. 启动服务

```bash
python -m enlighten.api_server
```

服务默认运行在 `http://localhost:8000`。

### 6. 使用 Docker（可选）

```bash
docker-compose --profile production up -d
```

### 7. API 调用示例

```bash
# 健康检查
curl http://localhost:8000/health

# 状态查询
curl http://localhost:8000/status

# L3统计
curl http://localhost:8000/l3/stats

# 推理请求
curl -X POST http://localhost:8000/inference \
  -H "Content-Type: application/json" \
  -d '{"text": "太阳系有几颗行星？", "max_length": 500}'
```

---

## 配置说明

EnlightenLM 支持多种配置方式，优先级从低到高：

1. **预设模式**（balanced/full/lightweight）
2. **YAML配置文件**（`configs/*.yaml`）
3. **环境变量**（`.env`）
4. **代码参数**

### 环境变量配置

```ini
# 配置文件路径
ENLIGHTEN_CONFIG_PATH=configs/balanced.yaml
ENLIGHTEN_MODE=balanced

# 模型配置
ENLIGHTEN_MODEL_USE_LOCAL=true
ENLIGHTEN_MODEL_LOCAL_NAME=qwen2.5:7b
ENLIGHTEN_MODEL_API_PROVIDER=ollama
ENLIGHTEN_MODEL_API_MODEL=qwen2.5:14b
```

### 预设模式

| 模式 | 说明 |
|------|------|
| `full` | 完整功能，所有组件启用 |
| `balanced` | 平衡模式，推荐日常使用 |
| `lightweight` | 轻量模式，最小资源占用 |

### MLP 幻觉判别器训练

#### 1. 生成示例数据

```bash
python generate_sample_data.py
```

#### 2. 训练模型

```bash
python train/train_mlp.py
```

#### 3. 评估模型

```bash
python train/evaluate.py
```

#### 4. 模型集成

训练好的模型会自动集成到系统中，用于实时幻觉检测。系统会在输入和输出两个阶段进行幻觉风险检测，检测到高风险时自动截断并记录审计日志。

#### 5. 配置幻觉判别器

可以通过配置文件或环境变量控制幻觉判别器的启用/禁用：

**在配置文件中设置**：
```yaml
# 在 configs/balanced.yaml 中添加
use_hallucination_discriminator: true  # 启用幻觉判别器
```

**通过环境变量设置**：
```ini
# 在 .env 文件中添加
ENLIGHTEN_USE_HALLUCINATION_DISCRIMINATOR=true
```

---

## API 参考

### 健康检查

`GET /health`

响应：
```json
{
  "status": "healthy",
  "model_loaded": true,
  "mode": "api"
}
```

### 模型状态

`GET /status`

响应：
```json
{
  "mode": "api",
  "model": "ollama",
  "use_bayesian_l3": false,
  "use_l1_adapter": false,
  "use_skeleton_l2": true,
  "use_l3_controller": true
}
```

### L3 统计

`GET /l3/stats`

响应：
```json
{
  "confidence": 0.0,
  "entropy": 1.0,
  "stability": 1.0,
  "bayesian": {"normal": 1.0, "noise": 0, "bias": 0.0},
  "temperature": 0.7,
  "sceneType": "general",
  "van": {"total_requests": 0, "van_events": 0}
}
```

### 推理请求

`POST /inference`

请求体：
```json
{
  "text": "解释量子计算的基本原理",
  "max_length": 500
}
```

---

## 测试

运行核心测试（当前 36+ 个用例全部通过）：

```bash
pytest tests/test_merkle_tree.py tests/test_metacognition.py tests/test_engram.py tests/test_l3_hallucination_detection.py -v
```

### 测试覆盖

| 测试类别 | 测试文件 | 用例数 | 覆盖内容 |
|----------|----------|--------|----------|
| Merkle树审计 | `test_merkle_tree.py` | 8 | 哈希链、Merkle树构建、验证、完整性检查 |
| 元认知自检 | `test_metacognition.py` | 8 | 自检提示、自动追问、信心值计算、阈值判断 |
| Engram锚点 | `test_engram.py` | 6 | 创建锚点、回退、最大数量限制、重置 |
| 幻觉检测 | `test_l3_hallucination_detection.py` | 13 | 事实性幻觉、自相矛盾、无根据猜测、重复循环、逻辑不一致、敏感内容检测 |

---

## 下一步计划

### 短期（1~2 周）

1. ✅ **架构定位收拢与文档更新** - 已完成，明确技术路线为模型无关外围监控
2. ✅ **审计链激活** - 已完成，哈希链与 HMAC 签名模块已接入实际推理管线
3. ✅ **MLP 幻觉判别器** - 已完成基于规则的实时幻觉风险预测，待收集标注数据离线训练

### 中期（2~3 周）

4. **偏见检测与缓解（P0）** - 集成多领域敏感词库，触发后追加元认知自检提示
5. **偏见检测与缓解（P1）** - 构建有偏/无偏语句对比数据集，训练偏见探针
6. ✅ **干预策略精细化** - 已完成三级策略：continue / halt / rewind

### 长期（3~4 周）

7. **Agent 协同验证** - 集成多 Agent 辩论框架
8. **监控面板与生产化** - 提供可视化控制台
9. **偏见与幻觉统一评估管线** - 集成标准化基准
10. **收集标注数据** - 收集真实推理数据（含人工标注的幻觉/非幻觉 token 级标签），训练 MLP 幻觉判别器

---

## 贡献指南

欢迎提交 Issue 和 Pull Request！

### 开发环境设置

```bash
# 克隆仓库
git clone https://github.com/610005189/EnlightenLM.git
cd EnlightenLM

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 代码规范

- **代码风格**：遵循 `black` 格式规范
- **导入排序**：使用 `isort` 整理导入顺序
- **类型注解**：鼓励添加类型注解提升可读性
- **文档字符串**：公开 API 应包含 docstring

```bash
# 格式化代码
black enlighten/ tests/

# 检查导入顺序
isort --check-only enlighten/ tests/
```

### 提交规范

**提交信息格式**：
```
<类型>: <简短描述>

[可选的详细说明]

[可选的关联 Issue]
```

**类型标识**：
| 标识 | 说明 |
|------|------|
| `feat` | 新功能 |
| `fix` | Bug 修复 |
| `docs` | 文档更新 |
| `test` | 测试相关 |
| `refactor` | 代码重构 |
| `security` | 安全相关 |

---

## 相关资源

- [架构设计文档](docs/architecture_v3.0.md)
- [API 参考文档](docs/api_reference.md)
- [安全体系文档](docs/security.md)
- [实现状态](docs/implementation_status.md)
- [用户手册](docs/user_manual.md)
- [管理员指南](docs/admin_guide.md)
- [用户待办任务](docs/user_tasks.md)
- [实施计划](docs/implementation_plan.md)
- [架构决策记录](docs/ADR-001-model-agnostic-approach.md)

---

## 许可证

本项目基于 [MIT License](LICENSE) 开源。

---

**维护者**：EnlightenLM 团队
**最后更新**：2026-04-27
**版本**：v2.5.0 - 三层安全监控 + 贝叶斯推断 + 元认知自检 + Engram锚点 + Merkle树审计 + 实时熵监控 + MLP幻觉判别器 + WebSocket 实时监控 + Prometheus/Grafana 监控已就绪