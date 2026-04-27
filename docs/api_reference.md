# EnlightenLM API 参考文档

> 版本: v2.5.0
> 更新日期: 2026-04-26

---

## 目录

1. [EnlightenLM 主类](#1-enlightenlm-主类)
2. [L1 生成层API](#2-l1-生成层api) - **模型无关透明调用**
3. [L2 工作记忆API](#3-l2-工作记忆api)
4. [L3 元控制器API](#4-l3-元控制器api)
5. [审计系统API](#5-审计系统api)
6. [REST API端点](#6-rest-api端点)
7. [WebSocket端点](#7-websocket端点)
8. [数据类型定义](#8-数据类型定义)

---

## 1. EnlightenLM 主类

### `class HybridEnlightenLM`

主入口类，协调L1、L2、L3组件完成推理。

#### 初始化

```python
HybridEnlightenLM(
    model_backend: str = "ollama",
    model_name: str = "qwen2.5:7b",
    use_bayesian_l3: bool = True,
    use_l3_controller: bool = True,
    config: Dict = None
) -> HybridEnlightenLM
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| model_backend | str | "ollama" | 模型后端 ("ollama" \| "deepseek" \| "vllm") |
| model_name | str | "qwen2.5:7b" | 模型名称 |
| use_bayesian_l3 | bool | True | 是否使用贝叶斯L3控制器 |
| use_l3_controller | bool | True | 是否使用L3控制器 |
| config | Dict | None | 配置字典 |

#### 方法

##### `generate()`

执行推理并返回结果。

```python
def generate(
    self,
    prompt: str,
    max_tokens: int = 2048,
    temperature: float = 0.7,
    stream: bool = False,
    **kwargs
) -> Tuple[str, int]
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| prompt | str | - | 输入提示词 |
| max_tokens | int | 2048 | 最大生成长度 |
| temperature | float | 0.7 | 温度参数 |
| stream | bool | False | 是否流式输出 |

**返回**: `(generated_text, token_count)`

##### `chat()`

执行对话生成。

```python
def chat(
    self,
    messages: List[Dict[str, str]],
    max_tokens: int = 2048,
    temperature: float = 0.7,
    stream: bool = False,
    **kwargs
) -> ChatResult
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| messages | List[Dict] | - | 消息列表 [{"role": "user", "content": "..."}] |
| max_tokens | int | 2048 | 最大生成长度 |
| temperature | float | 0.7 | 温度参数 |
| stream | bool | False | 是否流式输出 |

**返回**: `ChatResult`

##### `get_status()`

获取系统状态。

```python
def get_status(self) -> Dict[str, Any]
```

**返回**: 包含模式、步骤计数、L3控制器状态等状态信息的字典

##### `get_entropy_stats()`

获取L2层的熵统计信息。

```python
def get_entropy_stats(self) -> Dict[str, float]
```

**返回**: 包含熵均值、方差、趋势等信息的字典

##### `get_van_stats()`

获取VAN监控统计信息。

```python
def get_van_stats(self) -> Dict[str, Any]
```

**返回**: 包含VAN事件、截断次数等信息的字典

##### `get_l3_trace_signals()`

获取L3追踪信号。

```python
def get_l3_trace_signals(self) -> Dict[str, float]
```

**返回**: 包含L3控制器输入信号的字典

##### `reset()`

重置系统状态。

```python
def reset(self) -> None
```

---

## 2. L1 生成层API（模型无关透明调用）

### 概述

L1生成层负责透明调用底层大模型，**不进行任何内部计算图修改**。安全能力完全由外围L2/L3提供。

### 支持的后端

| 后端 | 说明 | 状态 |
|------|------|------|
| Ollama | 本地部署，通过HTTP API调用 | ✅ 已实现 |
| DeepSeek API | 云端API，支持流式输出 | ✅ 已实现 |
| vLLM | 高性能推理引擎 | 🚧 开发中 |

### `class OllamaClient`

Ollama模型客户端。

#### 初始化

```python
OllamaClient(
    model_name: str = "qwen2.5:7b",
    base_url: str = "http://localhost:11434",
    timeout: int = 120
) -> OllamaClient
```

#### 方法

##### `chat()`

发送聊天请求。

```python
def chat(
    self,
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    max_tokens: int = 2048,
    stream: bool = False
) -> Dict[str, Any]
```

##### `generate()`

发送生成请求。

```python
def generate(
    self,
    prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    stream: bool = False
) -> Dict[str, Any]
```

### `class DeepSeekClient`

DeepSeek API客户端。

#### 初始化

```python
DeepSeekClient(
    api_key: str,
    model: str = "deepseek-chat",
    base_url: str = "https://api.deepseek.com"
) -> DeepSeekClient
```

#### 方法

##### `chat()`

发送聊天请求。

```python
def chat(
    self,
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    max_tokens: int = 2048,
    stream: bool = False
) -> Dict[str, Any]
```

---

## 3. L2 工作记忆API

### `class L2WorkingMemory`

L2工作记忆层，负责上下文压缩和熵统计。

#### 初始化

```python
L2WorkingMemory(
    memory_size: int = 512,
    embedding_dim: int = 1024,
    config_path: str = None
) -> L2WorkingMemory
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| memory_size | int | 512 | 活跃token数量m |
| embedding_dim | int | 1024 | 嵌入维度d |
| config_path | str | None | 配置文件路径 |

#### 方法

##### `update()`

更新工作记忆。

```python
def update(
    self,
    key: Tensor,
    value: Tensor,
    importance_scores: Tensor
) -> None
```

##### `get_sparse_kv()`

获取稀疏键值对。

```python
def get_sparse_kv(self) -> Tuple[Tensor, Tensor, List[int]]
```

**返回**: `(K_tilde, V_tilde, active_indices)`

##### `get_entropy_stats()`

获取熵统计。

```python
def get_entropy_stats(self) -> EntropyStats
```

---

## 4. L3 元控制器API

### `class L3Controller`

L3元控制器，负责熵监控和截断决策。

#### 初始化

```python
L3Controller(
    entropy_threshold: float = 0.5,
    variance_threshold: float = 0.05,
    config_path: str = None
) -> L3Controller
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| entropy_threshold | float | 0.5 | 熵阈值μ_H |
| variance_threshold | float | 0.05 | 方差阈值σ_H |

#### 方法

##### `forward()`

执行元控制决策。

```python
def forward(
    self,
    entropy_stats: EntropyStats,
    van_event: bool,
    task_embedding: Tensor = None
) -> ControlSignals
```

**返回**: `ControlSignals`

##### `should_cutoff()`

判断是否应该截断。

```python
def should_cutoff(
    self,
    mu_h: float,
    sigma_h: float,
    k_h: float
) -> bool
```

##### `compute_control_signals()`

计算调控信号。

```python
def compute_control_signals(
    self,
    entropy_stats: EntropyStats,
    task_type: str = None
) -> ControlSignals
```

---

## 5. 审计系统API

### `class AuditLogger`

审计日志记录器。

#### 初始化

```python
AuditLogger(
    storage_path: str = "logs/audit",
    hash_algorithm: str = "sha256"
) -> AuditLogger
```

#### 方法

##### `log_session()`

记录会话。

```python
def log_session(
    self,
    session_id: str,
    input_text: str,
    output_text: str,
    meta_description: str,
    core_rules: Dict,
    user_params: Dict,
    attention_stats: Dict,
    cutoff_event: Dict = None
) -> str
```

| 参数 | 类型 | 说明 |
|------|------|------|
| session_id | str | 会话ID |
| input_text | str | 输入文本 |
| output_text | str | 输出文本 |
| meta_description | str | 元描述 |
| core_rules | Dict | 核心规则信息 |
| user_params | Dict | 用户参数 |
| attention_stats | Dict | 注意力统计 |
| cutoff_event | Dict | 截断事件 |

**返回**: 哈希链节点ID

##### `verify_hash_chain()`

验证哈希链完整性。

```python
def verify_hash_chain(self) -> bool
```

##### `get_session_summary()`

获取会话摘要。

```python
def get_session_summary(self, session_id: str) -> Dict
```

---

## 6. REST API端点

### `POST /api/v1/chat/completions`

聊天补全端点（兼容OpenAI格式）。

**请求体**:
```json
{
    "model": "qwen2.5:7b",
    "messages": [
        {"role": "system", "content": "你是一个有帮助的助手。"},
        {"role": "user", "content": "太阳系有几颗行星？"}
    ],
    "temperature": 0.7,
    "max_tokens": 2048,
    "stream": false,
    "safety": {
        "enable_monitoring": true,
        "intervention_level": "halt"
    }
}
```

**响应**:
```json
{
    "id": "chatcmpl-xxx",
    "object": "chat.completion",
    "created": 1714153600,
    "model": "qwen2.5:7b",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "太阳系有8颗行星..."
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 50,
        "completion_tokens": 120,
        "total_tokens": 170
    },
    "safety": {
        "risk_level": "low",
        "hallucination_risk": 0.12,
        "bias_flag": false,
        "interventions": [],
        "van_events": 0,
        "cutoff_triggered": false,
        "hash_signature": "sha256:xxxx",
        "entropy_stats": {
            "mean": 0.65,
            "variance": 0.02,
            "trend": 0.01
        }
    }
}
```

### `GET /l3/stats`

获取L3统计信息。

**响应**:
```json
{
    "confidence": 95.2,
    "entropy": 0.65,
    "stability": 0.88,
    "selfReferential": 0.12,
    "bayesian": {
        "normal": 0.85,
        "noise": 0.05,
        "bias": 0.10
    },
    "temperature": 0.7,
    "sceneType": "通用",
    "generation": {
        "progress": 100,
        "speed": 5.0,
        "tokens": 170,
        "preview": "生成完成"
    },
    "l2": {
        "entropy": 0.62,
        "entropyTrend": "上升",
        "entropyDistribution": [0.65, 0.68, 0.70, 0.68, 0.66, 0.64, 0.63, 0.62, 0.61, 0.60]
    },
    "attention": {
        "concentration": 0.85,
        "area": "全局",
        "heatmap": [0.1, 0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.2, 0.1]
    }
}
```

### `GET /health`

健康检查。

**响应**:
```json
{
    "status": "healthy",
    "model_loaded": true,
    "model_backend": "ollama",
    "model_name": "qwen2.5:7b",
    "use_bayesian_l3": true,
    "use_l3_controller": true,
    "version": "3.0.0"
}
```

### `GET /api/v1/audit/logs`

获取审计日志。

**查询参数**:
| 参数 | 类型 | 说明 |
|------|------|------|
| session_id | str | 会话ID（可选） |
| limit | int | 返回数量限制 |

**响应**:
```json
{
    "logs": [
        {
            "session_id": "xxx",
            "timestamp": 1714153600,
            "input_text": "...",
            "output_text": "...",
            "cutoff_count": 0,
            "total_tokens": 150,
            "risk_level": "low"
        }
    ]
}
```

### `GET /api/v1/safety/config`

获取当前监控阈值配置。

**响应**:
```json
{
    "van_sensitivity": 0.7,
    "self_reference_threshold": 0.8,
    "max_repetition_ratio": 0.3,
    "cooling_window_seconds": 60,
    "bayesian_prior_bias": 0.5
}
```

### `PUT /api/v1/safety/config`

动态调整监控强度。

**请求体**:
```json
{
    "van_sensitivity": 0.8,
    "cooling_window_seconds": 30
}
```

## 7. WebSocket端点

### `ws://host/ws/l3/stats`

L3实时监控WebSocket端点。

**连接建立后**：服务器将每100ms推送一次L3统计数据。

**推送数据格式**:
```json
{
    "confidence": 95.2,
    "entropy": 0.65,
    "stability": 0.88,
    "selfReferential": 0.12,
    "bayesian": {
        "normal": 0.85,
        "noise": 0.05,
        "bias": 0.10
    },
    "temperature": 0.7,
    "sceneType": "通用",
    "generation": {
        "progress": 100,
        "speed": 5.0,
        "tokens": 170,
        "preview": "生成完成"
    },
    "l2": {
        "entropy": 0.62,
        "entropyTrend": "上升",
        "entropyDistribution": [0.65, 0.68, 0.70, 0.68, 0.66, 0.64, 0.63, 0.62, 0.61, 0.60]
    },
    "attention": {
        "concentration": 0.85,
        "area": "全局",
        "heatmap": [0.1, 0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.2, 0.1]
    },
    "van": {
        "total_requests": 100,
        "van_events": 5,
        "blocked_requests": 1,
        "block_ratio": 0.01
    },
    "timestamp": 1714153600.123
}
```

**JavaScript客户端示例**:
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/l3/stats');

ws.onopen = function() {
    console.log('WebSocket连接已建立');
};

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    updateL3Metrics(data);
};

ws.onerror = function(error) {
    console.error('WebSocket错误:', error);
};

ws.onclose = function() {
    console.log('WebSocket连接已关闭');
};
```

---

## 8. 数据类型定义

### `InferenceResult`

推理结果。

```python
@dataclass
class InferenceResult:
    session_id: str
    output: str
    meta_description: str
    user_params: Dict
    cutoff: bool
    cutoff_reason: str
    attention_stats: Dict
    tokens: List[int]
    generated_at: datetime
```

### `EntropyStats`

熵统计。

```python
@dataclass
class EntropyStats:
    mean: float          # μ_H
    variance: float      # σ_H²
    trend: float         # k_H
    current: float       # 当前熵值
```

### `ControlSignals`

调控信号。

```python
@dataclass
class ControlSignals:
    tau: float           # 温度
    theta: float         # 稀疏阈值
    alpha: float         # DMN系数
    stability: bool      # 稳定性标志
    cutoff: bool         # 截断信号
    reason: str          # 决策原因
```

### `L1Output`

L1输出。

```python
@dataclass
class L1Output:
    output_ids: Tensor
    attention_weights: Tensor
    hidden_states: Tensor
    entropy_stats: EntropyStats
    van_event: bool
```

### `AuditEntry`

审计条目。

```python
@dataclass
class AuditEntry:
    index: int
    hash: str
    data: Dict
    timestamp: float
    signature: str
```

---

## 错误代码

| 代码 | 名称 | 说明 |
|------|------|------|
| 400 | Bad Request | 请求参数错误 |
| 401 | Unauthorized | 认证失败 |
| 403 | Forbidden | 权限不足 |
| 404 | Not Found | 资源不存在 |
| 429 | Too Many Requests | 请求过于频繁 |
| 500 | Internal Error | 服务器内部错误 |
| 503 | Service Unavailable | 服务不可用 |

---

*文档版本: v2.5.0*
*最后更新: 2026-04-23*
