# EnlightenLM API 参考文档

> 版本: v1.0
> 更新日期: 2026-04-23

---

## 目录

1. [EnlightenLM 主类](#1-enlightenlm-主类)
2. [L1 生成层API](#2-l1-生成层api)
3. [L2 工作记忆API](#3-l2-工作记忆api)
4. [L3 元控制器API](#4-l3-元控制器api)
5. [审计系统API](#5-审计系统api)
6. [REST API端点](#6-rest-api端点)
7. [数据类型定义](#7-数据类型定义)

---

## 1. EnlightenLM 主类

### `class EnlightenLM`

主入口类，协调L1、L2、L3组件完成推理。

#### 初始化

```python
EnlightenLM(
    model_name: str = "deepseek-ai/DeepSeek-V3",
    device: str = "cuda",
    config_path: str = None
) -> EnlightenLM
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| model_name | str | "deepseek-ai/DeepSeek-V3" | 模型名称或路径 |
| device | str | "cuda" | 运行设备 |
| config_path | str | None | 配置文件路径 |

#### 方法

##### `generate()`

执行推理并返回结果。

```python
def generate(
    self,
    text: str,
    user_params: Dict = None,
    task_type: str = None,
    max_length: int = 2048
) -> InferenceResult
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| text | str | - | 输入文本 |
| user_params | Dict | None | 用户参数 |
| task_type | str | None | 任务类型 |
| max_length | int | 2048 | 最大生成长度 |

**返回**: `InferenceResult`

**示例**:
```python
result = model.generate(
    text="解释量子计算",
    user_params={
        "creativity": 0.5,
        "detail": 0.8,
        "safety": 0.7
    }
)
```

##### `batch_generate()`

批量推理。

```python
def batch_generate(
    self,
    texts: List[str],
    user_params_list: List[Dict] = None,
    max_length: int = 2048
) -> List[InferenceResult]
```

---

## 2. L1 生成层API

### `class L1Generation`

L1生成层，负责双流注意力融合和token生成。

#### 初始化

```python
L1Generation(
    model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    config: L1Config = None
) -> L1Generation
```

#### 方法

##### `forward()`

```python
def forward(
    self,
    input_ids: Tensor,
    attention_mask: Tensor = None,
    control_signals: ControlSignals = None
) -> L1Output
```

| 参数 | 类型 | 说明 |
|------|------|------|
| input_ids | Tensor | 输入token IDs |
| attention_mask | Tensor | 注意力掩码 |
| control_signals | ControlSignals | 来自L3的调控信号 |

**返回**: `L1Output`

##### `apply_dan_attention()`

应用DAN（目标驱动注意力）。

```python
def apply_dan_attention(
    self,
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    task_bias: Tensor
) -> Tuple[Tensor, Tensor]
```

##### `apply_van_attention()`

应用VAN（刺激驱动注意力）。

```python
def apply_van_attention(
    self,
    tokens: List[int],
    hidden_states: Tensor
) -> Tuple[Tensor, bool]
```

##### `fuse_attention()`

融合双流注意力。

```python
def fuse_attention(
    self,
    attn_dan: Tensor,
    attn_van: Tensor,
    tau: float,
    theta: float
) -> Tensor
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

### `POST /inference`

推理端点。

**请求体**:
```json
{
    "text": "输入文本",
    "user_params": {
        "creativity": 0.5,
        "detail": 0.8,
        "safety": 0.7,
        "role": 1
    },
    "task_type": "math"
}
```

**响应**:
```json
{
    "session_id": "uuid",
    "output": "生成的文本",
    "meta_description": "元描述",
    "user_params": {...},
    "cutoff": false,
    "cutoff_reason": "",
    "attention_stats": {...}
}
```

### `GET /health`

健康检查。

**响应**:
```json
{
    "status": "healthy",
    "model_loaded": true,
    "version": "1.0.0"
}
```

### `GET /audit/session/{session_id}/summary`

获取会话摘要。

**响应**:
```json
{
    "session_id": "uuid",
    "input_text": "...",
    "output_text": "...",
    "cutoff_count": 0,
    "total_tokens": 150
}
```

### `GET /audit/verify`

验证哈希链。

**响应**:
```json
{
    "verified": true,
    "message": "Hash chain is valid"
}
```

### `GET /audit/threat/statistics`

威胁检测统计。

**响应**:
```json
{
    "total_requests": 1000,
    "threats_detected": 5,
    "jailbreak_attempts": 2,
    "injection_attempts": 3
}
```

---

## 7. 数据类型定义

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

*文档版本: v1.0*
*最后更新: 2026-04-23*
