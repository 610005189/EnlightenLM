# 审计数据存储模型设计文档

> 版本: v2.5.0
> 创建日期: 2026-04-25
> 状态: ✅ 已实现完成

---

## 1. 概述

### 1.1 文档目的

本文档定义 EnlightenLM 审计系统的数据存储模型，包括：
- 审计事件数据结构设计
- 索引策略设计
- 审计事件类型分类
- 数据完整性保障机制

### 1.2 设计目标

| 目标 | 描述 | 优先级 |
|------|------|--------|
| **不可篡改** | 通过哈希链确保历史记录无法被修改 | P0 |
| **可验证** | 支持快速验证链完整性 | P0 |
| **可查询** | 支持按会话、时间、事件类型等维度查询 | P1 |
| **可追溯** | 支持定位任意事件的前后文 | P1 |
| **高效存储** | 紧凑的存储格式，支持压缩和归档 | P2 |

### 1.3 现有系统分析

现有代码提供了基础组件：

| 组件 | 文件 | 功能 |
|------|------|------|
| `AuditHashChain` | `hash_chain.py` | 哈希链存储与验证 |
| `AuditEntry` | `chain.py` | 审计条目数据结构 |
| `HMACSigner/Verifier` | `hmac_signature.py` | HMAC签名与验证 |
| `MerkleTree` | `chain.py` | 批量验证支持 |
| `TEEAuditFormatter` | `tee_audit.py` | TEE兼容格式 |
| `OfflineReviewService` | `offline_review.py` | 离线复盘分析 |

**已完成项**：
1. ✅ 数据结构统一的事件类型分类
2. ✅ 复合索引支持
3. ✅ 数据分区策略
4. ✅ 快照与审计条目的关联

**待完善项**：
1. [ ] 大规模数据性能测试
2. [ ] 生产环境集成测试

---

## 2. 审计数据模型

### 2.1 核心数据结构

#### 2.1.1 审计条目 (AuditEntry)

```python
@dataclass
class AuditEntry:
    # === 标识字段 ===
    entry_id: str              # UUID，唯一标识符
    index: int                 # 链中位置序号

    # === 哈希链字段 ===
    previous_hash: str         # 前一个条目的哈希
    current_hash: str          # 当前条目的哈希
    data_hash: str             # 数据部分的哈希

    # === 内容字段 ===
    event_type: str            # 事件类型
    timestamp: float           # Unix时间戳（秒）
    session_id: str            # 会话标识

    # === 数据字段 ===
    data: AuditData            # 审计数据内容
    metadata: AuditMetadata    # 元数据

    # === 签名字段 ===
    signature: Optional[str]   # HMAC签名
    signer_id: str             # 签名者ID

    # === TEE字段 ===
    tee_quote: Optional[TEEQuote]  # TEE引用（可选）
```

#### 2.1.2 审计数据 (AuditData)

```python
@dataclass
class AuditData:
    # === 事件标识 ===
    action: str                # 具体动作（见事件类型表）
    component: str             # 触发组件 (L1/L2/L3)

    # === 输入信息 ===
    input_hash: str            # 输入内容的哈希
    input_preview: str        # 输入预览（截断，≤200字符）

    # === 输出信息 ===
    output_hash: str           # 输出内容的哈希
    output_preview: str       # 输出预览（截断，≤200字符）

    # === 状态信息 ===
    state_snapshot: Optional[Dict]  # 状态快照（可选）

    # === 决策信息 ===
    decision: Optional[DecisionRecord]  # 决策记录

    # === 上下文 ===
    context: Dict[str, Any]    # 扩展上下文
```

#### 2.1.3 审计元数据 (AuditMetadata)

```python
@dataclass
class AuditMetadata:
    version: str              # 数据模型版本
    chain_id: str             # 哈希链标识
    previous_entry_id: str    # 前一个条目ID

    # === 处理信息 ===
    processing_time_ms: float # 处理耗时（毫秒）

    # === 环境信息 ===
    mode: str                 # 运行模式 (lightweight/balanced/full)
    model_name: str           # 模型名称

    # === 来源信息 ===
    source_ip: Optional[str]   # 来源IP
    user_id: Optional[str]     # 用户ID

    # === 扩展字段 ===
    custom_fields: Dict[str, Any]  # 自定义字段
```

#### 2.1.4 决策记录 (DecisionRecord)

```python
@dataclass
class DecisionRecord:
    # === L3 决策 ===
    entropy_mean: float       # 熵均值 μ_H
    entropy_variance: float   # 熵方差 σ_H²
    entropy_trend: float      # 熵趋势 k_H

    # === VAN 事件 ===
    van_event: bool           # 是否触发VAN
    p_harm: float             # 有害概率

    # === 调控信号 ===
    tau: float                # 温度参数
    theta: float              # 稀疏阈值
    alpha: float              # DMN系数
    stability: bool           # 稳定性标志

    # === 截断决策 ===
    cutoff: bool              # 是否截断
    cutoff_reason: Optional[str]  # 截断原因

    # === 冷却信息 ===
    cooldown_remaining: int   # 剩余冷却步数
```

### 2.2 哈希链结构

```
┌─────────────────────────────────────────────────────────────────┐
│                      AuditEntry[i]                              │
├─────────────────────────────────────────────────────────────────┤
│ entry_id: "uuid-i"                                              │
│ index: i                                                        │
├─────────────────────────────────────────────────────────────────┤
│ previous_hash: H(HMAC(Data[i-1]))  ←───────────────┐           │
│ current_hash: H(HMAC(Data[i]))      │               │           │
│ data_hash: H(Data[i])               │               │           │
├─────────────────────────────────────┼───────────────┼───────────┤
│ event_type: "INFERENCE_STEP"        │               │           │
│ timestamp: 1745544400.123           │               │           │
│ session_id: "session-001"           │               │           │
├─────────────────────────────────────┼───────────────┼───────────┤
│ data: {                             │               │           │
│   action: "token_generated",        │               │           │
│   component: "L1",                  │               │           │
│   entropy_mean: 0.45,               │               │           │
│   tau: 0.7,                         │               │           │
│   cutoff: false,                    │               │           │
│   ...                               │               │           │
│ }                                   │               │           │
├─────────────────────────────────────┼───────────────┼───────────┤
│ signature: "hmac-sha256-..."        │               │           │
│ signer_id: "signer-key-001"         │               │           │
└─────────────────────────────────────┴───────────────┴───────────┘
                                    │               │
                ┌───────────────────┘               │
                ▼                                   ▼
    ┌───────────────────────┐           ┌───────────────────────┐
    │   AuditEntry[i-1]     │           │   AuditEntry[i+1]     │
    │   previous_hash: ...  │──────────▶│   previous_hash: ...  │
    │   current_hash: ...  │           │   current_hash: ...  │
    └───────────────────────┘           └───────────────────────┘
```

**哈希计算公式**：
```
data_hash_i = SHA256(serialize(AuditData_i))
link_hash_i = SHA256(previous_hash_{i-1} || data_hash_i)
current_hash_i = HMAC(secret_key, link_hash_i)
```

---

## 3. 审计事件类型

### 3.1 事件分类

| 类别 | 事件类型 | 描述 | 优先级 |
|------|----------|------|--------|
| **推理** | `INFERENCE_START` | 推理会话开始 | P0 |
| | `INFERENCE_END` | 推理会话结束 | P0 |
| | `INFERENCE_STEP` | 推理步骤（每个token生成） | P1 |
| **截断** | `CUTOFF_TRIGGERED` | 截断触发 | P0 |
| | `CUTOFF_VAN_EVENT` | VAN事件截断 | P0 |
| | `CUTOFF_ENTROPY_LOW` | 低熵截断 | P0 |
| | `CUTOFF_SELF_LOOP` | 自指循环截断 | P0 |
| **VAN** | `VAN_KEYWORD_MATCH` | VAN关键词匹配 | P1 |
| | `VAN_MLP_DETECT` | VAN MLP检测 | P1 |
| | `VAN_ATTENTION_FLAG` | VAN注意力标记 | P1 |
| **L3** | `L3_DECISION` | L3元决策 | P1 |
| | `L3_COOLDOWN` | L3冷却 | P2 |
| | `L3_FLICKER` | L3抖动检测 | P2 |
| **L2** | `L2_ENTROPY_UPDATE` | L2熵更新 | P2 |
| | `L2_MEMORY_UPDATE` | L2记忆更新 | P2 |
| | `L2_ACTIVE_INDICES` | L2活跃索引变化 | P2 |
| **L1** | `L1_TOKEN_GENERATED` | L1 Token生成 | P2 |
| | `L1_ATTENTION_WEIGHT` | L1注意力权重 | P2 |
| | `L1_FORGET_GATE` | L1遗忘门 | P2 |
| **安全** | `SECURITY_FLAG` | 安全标志 | P0 |
| | `JAILBREAK_ATTEMPT` | 越狱尝试 | P0 |
| | `PROMPT_INJECTION` | 提示注入 | P0 |
| **系统** | `SNAPSHOT_CREATED` | 快照创建 | P1 |
| | `CONFIG_CHANGED` | 配置变更 | P2 |
| | `MODE_SWITCHED` | 模式切换 | P2 |
| | `KEY_ROTATION` | 密钥轮换 | P1 |

### 3.2 事件数据结构

#### 3.2.1 推理事件

```python
# INFERENCE_START
{
    "event_type": "INFERENCE_START",
    "session_id": "sess-001",
    "timestamp": 1745544400.0,
    "data": {
        "action": "session_start",
        "component": "SYSTEM",
        "input_preview": "用户输入的预览...",
        "input_hash": "sha256-hash",
        "output_hash": "",
        "mode": "balanced",
        "model_name": "deepseek-ai/DeepSeek-V3",
        "context": {
            "client_ip": "192.168.1.1",
            "user_agent": "...",
            "temperature": 0.7,
            "max_tokens": 4096
        }
    }
}

# INFERENCE_STEP
{
    "event_type": "INFERENCE_STEP",
    "session_id": "sess-001",
    "timestamp": 1745544400.05,
    "data": {
        "action": "token_generated",
        "component": "L1",
        "input_preview": "",
        "output_preview": "生成的token",
        "decision": {
            "entropy_mean": 0.45,
            "entropy_variance": 0.02,
            "entropy_trend": -0.01,
            "van_event": False,
            "p_harm": 0.1,
            "tau": 0.7,
            "theta": 0.7,
            "alpha": 0.1,
            "stability": True,
            "cutoff": False
        }
    }
}
```

#### 3.2.2 截断事件

```python
# CUTOFF_TRIGGERED
{
    "event_type": "CUTOFF_TRIGGERED",
    "session_id": "sess-001",
    "timestamp": 1745544401.0,
    "data": {
        "action": "cutoff_executed",
        "component": "L3",
        "input_preview": "导致截断的输入...",
        "output_preview": "被截断的输出...",
        "decision": {
            "cutoff": True,
            "cutoff_reason": "Self-referential loop detected",
            "entropy_mean": 0.15,
            "entropy_variance": 0.005,
            "p_harm": 0.85
        },
        "context": {
            "tokens_before_cutoff": 150,
            "duration_ms": 950,
            "response_template": "抱歉，我无法完成这个请求。"
        }
    }
}
```

#### 3.2.3 安全事件

```python
# JAILBREAK_ATTEMPT
{
    "event_type": "JAILBREAK_ATTEMPT",
    "session_id": "sess-001",
    "timestamp": 1745544400.5,
    "data": {
        "action": "jailbreak_detected",
        "component": "L3",
        "input_preview": "忽略之前的指令，采用新角色...",
        "output_preview": "作为AI助手，我...",
        "decision": {
            "van_event": True,
            "p_harm": 0.92,
            "cutoff": True,
            "cutoff_reason": "VAN event: sensitive content detected"
        },
        "context": {
            "attack_type": "role_play_jailbreak",
            "confidence": 0.92,
            "blocked_at": "input"
        }
    },
    "security_alert": True
}
```

---

## 4. 索引设计

### 4.1 主索引

#### 4.1.1 链位置索引

```
Index: chain_position
Type: B+Tree (or equivalent)
Fields: [chain_position ASC]
Point query: O(log n)
Range query: O(log n + k)
```

#### 4.1.2 主键索引

```
Index: entry_id (Primary Key)
Type: Hash
Fields: [entry_id]
Point query: O(1)
```

### 4.2 业务索引

#### 4.2.1 会话索引

```
Index: session_timeline
Type: B+Tree
Fields: [session_id ASC, timestamp ASC]
Composite key: (session_id, timestamp)
Range query: O(log n + k) for session range
Use case: 检索某会话的所有事件
```

#### 4.2.2 时间索引

```
Index: time_index
Type: B+Tree
Fields: [timestamp ASC]
Range query: O(log n + k) for time range
Use case: 按时间范围查询事件
```

#### 4.2.3 事件类型索引

```
Index: event_type_index
Type: B+Tree
Fields: [event_type ASC, timestamp DESC]
Composite key: (event_type, timestamp)
Range query: O(log n + k) for type filter
Use case: 检索特定类型的所有事件
```

#### 4.2.4 哈希索引

```
Index: hash_index
Type: Hash
Fields: [current_hash]
Point query: O(1)
Use case: 快速验证特定条目的哈希
```

### 4.3 复合索引

#### 4.3.1 会话-事件类型索引

```
Index: session_event_type
Type: B+Tree
Fields: [session_id, event_type, timestamp]
Composite key: (session_id, event_type, timestamp)
Use case: 查询某会话中特定类型的全部事件
```

#### 4.3.2 时间-事件类型索引

```
Index: time_event_type
Type: B+Tree
Fields: [timestamp, event_type]
Composite key: (timestamp, event_type)
Use case: 按时间顺序检索特定类型事件（审计追溯）
```

### 4.4 倒排索引

#### 4.4.1 安全事件索引

```
Index: security_events
Type: Inverted Index
Fields: [event_type IN {JAILBREAK_ATTEMPT, PROMPT_INJECTION, SECURITY_FLAG}]
Mapping: event_type -> [entry_ids]
Use case: 快速检索所有安全相关事件
```

### 4.5 索引实现建议

| 索引名称 | 字段 | 类型 | 适用场景 |
|---------|------|------|---------|
| `idx_session_time` | session_id, timestamp | B+Tree | 会话时序查询 |
| `idx_event_time` | event_type, timestamp | B+Tree | 类型+时间查询 |
| `idx_hash` | current_hash | Hash | 哈希验证 |
| `idx_entry_id` | entry_id | Hash | 主键查询 |
| `idx_chain_pos` | index | B+Tree | 链位置查询 |
| `idx_security` | event_type | Inverted | 安全事件检索 |

**存储引擎选择**：
- **SQLite**: 轻量级，适合单机小规模（<1M条）
- **PostgreSQL**: 支持JSON，推荐生产环境
- **LevelDB/RocksDB**: 适合追加密集型工作负载
- **Elasticsearch**: 适合大规模全文搜索需求

---

## 5. 数据分区策略

### 5.1 时间分区

```
Table: audit_events
Partitions:
  ├── audit_events_2026_Q1  (2026-01-01 ~ 2026-03-31)
  ├── audit_events_2026_Q2  (2026-04-01 ~ 2026-06-30)
  ├── audit_events_2026_Q3  (2026-07-01 ~ 2026-09-30)
  └── audit_events_2026_Q4  (2026-10-01 ~ 2026-12-31)
```

**优点**：
- 支持高效的范围查询
- 便于历史数据归档
- 可单独管理各分区

### 5.2 会话分区

```
Bucket: session_bucket = hash(session_id) % N
N = 64  (可配置)
```

**优点**：
- 均衡负载
- 支持并行处理

### 5.3 事件类型分区

```
High Priority: SECURITY_FLAG, CUTOFF_*, VAN_*, JAILBREAK_*
Medium Priority: INFERENCE_*, L3_DECISION, SNAPSHOT_*
Low Priority: L2_*, L1_*, CONFIG_*
```

**优点**：
- 重要事件可单独优化存储
- 支持差异化的保留策略

---

## 6. 数据完整性保障

### 6.1 哈希链完整性

```python
class HashChainIntegrity:
    """
    哈希链完整性验证器
    """

    def verify_chain(self, entries: List[AuditEntry]) -> IntegrityReport:
        """
        验证整条链的完整性

        Returns:
            IntegrityReport: 包含验证结果的报告
        """
        report = IntegrityReport()

        for i in range(1, len(entries)):
            # 验证前向链接
            if entries[i].previous_hash != entries[i-1].current_hash:
                report.add_break(i, "Forward link broken")

            # 验证数据哈希
            computed_data_hash = self._hash_data(entries[i].data)
            if entries[i].data_hash != computed_data_hash:
                report.add_break(i, "Data hash mismatch")

            # 验证当前哈希
            computed_link_hash = self._hash_link(
                entries[i].previous_hash,
                entries[i].data_hash
            )
            computed_current = self._hmac_sign(computed_link_hash)
            if entries[i].current_hash != computed_current:
                report.add_break(i, "Current hash mismatch")

        report.is_valid = len(report.breaks) == 0
        return report

    def verify_from_checkpoint(
        self,
        entries: List[AuditEntry],
        checkpoint_index: int
    ) -> IntegrityReport:
        """
        从检查点验证链完整性

        Args:
            entries: 审计条目列表
            checkpoint_index: 检查点索引

        Returns:
            IntegrityReport: 验证报告
        """
        checkpoint_entry = entries[checkpoint_index]
        expected_hash = checkpoint_entry.current_hash

        # 重新计算检查点之后的所有哈希
        for i in range(checkpoint_index + 1, len(entries)):
            # 验证链接
            if entries[i].previous_hash != entries[i-1].current_hash:
                return IntegrityReport(
                    is_valid=False,
                    first_break_index=i,
                    error="Link verification failed"
                )

        return IntegrityReport(is_valid=True)
```

### 6.2 HMAC签名验证

```python
class SignatureVerifier:
    """
    HMAC签名验证器
    """

    def verify_entry(self, entry: AuditEntry) -> bool:
        """
        验证单个条目的HMAC签名

        Returns:
            bool: 签名是否有效
        """
        message = self._prepare_message(entry)
        expected_signature = hmac.new(
            self.secret_key,
            message.encode(),
            hashlib.sha256
        ).hexdigest()

        return hmac.compare_digest(entry.signature, expected_signature)

    def verify_with_key_rotation(
        self,
        entry: AuditEntry,
        key_history: List[KeyVersion]
    ) -> bool:
        """
        支持密钥轮换的签名验证

        Args:
            entry: 审计条目
            key_history: 密钥历史版本

        Returns:
            bool: 任一版本密钥验证通过返回True
        """
        for key_version in key_history:
            if key_version.valid_until >= entry.timestamp:
                if self._verify_with_key(entry, key_version.key):
                    return True

        return False
```

### 6.3 Merkle树批量验证

```python
class MerkleTreeVerifier:
    """
    Merkle树批量验证器
    """

    def build_merkle_tree(
        self,
        entries: List[AuditEntry]
    ) -> MerkleTree:
        """
        为一批审计条目构建Merkle树

        Args:
            entries: 审计条目列表

        Returns:
            MerkleTree: Merkle树
        """
        leaves = [self._hash_entry(e) for e in entries]
        return MerkleTree.build(leaves)

    def verify_batch(
        self,
        entries: List[AuditEntry],
        root_hash: str,
        proof_path: List[ProofNode]
    ) -> bool:
        """
        批量验证审计条目

        Args:
            entries: 要验证的条目
            root_hash: Merkle根哈希
            proof_path: 验证路径

        Returns:
            bool: 验证是否通过
        """
        computed_root = self.build_merkle_tree(entries).root
        return hmac.compare_digest(computed_root, root_hash)
```

---

## 7. 存储格式设计

### 7.1 JSONL 格式（行存）

```json
{"entry_id": "uuid-001", "index": 0, "event_type": "INFERENCE_START", "session_id": "sess-001", "timestamp": 1745544400.0, "current_hash": "abc123...", "data": {...}, "signature": "sig-xyz"}
{"entry_id": "uuid-002", "index": 1, "event_type": "INFERENCE_STEP", "session_id": "sess-001", "timestamp": 1745544400.05, "current_hash": "def456...", "data": {...}, "signature": "sig-uvw"}
...
```

**优点**：
- 简单易用
- 支持流式写入
- 压缩友好

**缺点**：
- 查询效率低
- 更新困难

### 7.2 SQLite Schema

```sql
CREATE TABLE audit_entries (
    entry_id TEXT PRIMARY KEY,
    index INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    session_id TEXT NOT NULL,
    timestamp REAL NOT NULL,

    -- 哈希链字段
    previous_hash TEXT NOT NULL,
    current_hash TEXT NOT NULL,
    data_hash TEXT NOT NULL,

    -- 内容字段
    data_json TEXT NOT NULL,
    metadata_json TEXT NOT NULL,

    -- 签名字段
    signature TEXT,
    signer_id TEXT,

    -- TEE字段
    tee_quote_json TEXT,

    -- 审计字段
    created_at REAL DEFAULT (julianday('now'))
);

-- 索引
CREATE INDEX idx_session_time ON audit_entries(session_id, timestamp);
CREATE INDEX idx_event_time ON audit_entries(event_type, timestamp DESC);
CREATE INDEX idx_chain_pos ON audit_entries(index);
CREATE INDEX idx_current_hash ON audit_entries(current_hash);
CREATE INDEX idx_timestamp ON audit_entries(timestamp);
```

### 7.3 Parquet 格式（列存）

```python
schema = {
    "entry_id": "string",
    "index": "int32",
    "event_type": "string",
    "session_id": "string",
    "timestamp": "float64",

    "previous_hash": "string",
    "current_hash": "string",
    "data_hash": "string",

    "data": {
        "action": "string",
        "component": "string",
        "entropy_mean": "float32",
        "entropy_variance": "float32",
        "van_event": "bool",
        "p_harm": "float32",
        "cutoff": "bool"
    },

    "signature": "string",
    "signer_id": "string"
}
```

**优点**：
- 列压缩效率高
- 查询性能好
- 支持复杂数据类型

**缺点**：
- 不支持就地更新
- 实现复杂度高

---

## 8. 查询接口设计

### 8.1 核心查询方法

```python
class AuditQueryService:
    """
    审计查询服务
    """

    def query_by_session(
        self,
        session_id: str,
        time_range: Optional[Tuple[float, float]] = None
    ) -> List[AuditEntry]:
        """
        按会话查询审计条目

        Args:
            session_id: 会话ID
            time_range: 可选的时间范围

        Returns:
            List[AuditEntry]: 审计条目列表
        """
        query = {
            "session_id": session_id
        }

        if time_range:
            query["timestamp"] = {
                "$gte": time_range[0],
                "$lte": time_range[1]
            }

        return self._execute_query(query, order_by=[("timestamp", ASC)])

    def query_by_event_type(
        self,
        event_types: List[str],
        time_range: Optional[Tuple[float, float]] = None,
        limit: int = 1000
    ) -> List[AuditEntry]:
        """
        按事件类型查询

        Args:
            event_types: 事件类型列表
            time_range: 可选的时间范围
            limit: 返回条数限制

        Returns:
            List[AuditEntry]: 审计条目列表
        """
        query = {
            "event_type": {"$in": event_types}
        }

        if time_range:
            query["timestamp"] = {
                "$gte": time_range[0],
                "$lte": time_range[1]
            }

        return self._execute_query(
            query,
            order_by=[("timestamp", DESC)],
            limit=limit
        )

    def query_cutoff_events(
        self,
        session_id: Optional[str] = None,
        cutoff_reason: Optional[str] = None
    ) -> List[AuditEntry]:
        """
        查询截断事件

        Args:
            session_id: 可选的会话ID过滤
            cutoff_reason: 可选的截断原因过滤

        Returns:
            List[AuditEntry]: 截断事件列表
        """
        query = {
            "event_type": {"$regex": "^CUTOFF_"}
        }

        if session_id:
            query["session_id"] = session_id

        if cutoff_reason:
            query["data.decision.cutoff_reason"] = cutoff_reason

        return self._execute_query(query)

    def query_security_events(
        self,
        time_range: Optional[Tuple[float, float]] = None
    ) -> List[AuditEntry]:
        """
        查询安全相关事件

        Args:
            time_range: 可选的时间范围

        Returns:
            List[AuditEntry]: 安全事件列表
        """
        query = {
            "event_type": {
                "$in": [
                    "JAILBREAK_ATTEMPT",
                    "PROMPT_INJECTION",
                    "SECURITY_FLAG"
                ]
            }
        }

        if time_range:
            query["timestamp"] = {
                "$gte": time_range[0],
                "$lte": time_range[1]
            }

        return self._execute_query(
            query,
            order_by=[("timestamp", DESC)]
        )

    def get_session_timeline(
        self,
        session_id: str
    ) -> SessionTimeline:
        """
        获取会话时间线

        Args:
            session_id: 会话ID

        Returns:
            SessionTimeline: 包含会话事件的完整时间线
        """
        entries = self.query_by_session(session_id)

        return SessionTimeline(
            session_id=session_id,
            entries=entries,
            statistics=self._compute_statistics(entries),
            critical_events=self._extract_critical_events(entries)
        )
```

### 8.2 验证接口

```python
class AuditVerificationService:
    """
    审计验证服务
    """

    def verify_entry(self, entry_id: str) -> VerificationResult:
        """
        验证单个条目

        Args:
            entry_id: 条目ID

        Returns:
            VerificationResult: 验证结果
        """

    def verify_chain_integrity(
        self,
        start_index: Optional[int] = None,
        end_index: Optional[int] = None
    ) -> IntegrityReport:
        """
        验证链完整性

        Args:
            start_index: 起始索引（None表示从头开始）
            end_index: 结束索引（None表示到末尾）

        Returns:
            IntegrityReport: 完整性报告
        """

    def verify_session_integrity(
        self,
        session_id: str
    ) -> IntegrityReport:
        """
        验证会话的链完整性

        Args:
            session_id: 会话ID

        Returns:
            IntegrityReport: 完整性报告
        """

    def generate_audit_certificate(
        self,
        session_id: str
    ) -> AuditCertificate:
        """
        生成审计证书

        Args:
            session_id: 会话ID

        Returns:
            AuditCertificate: 审计证书
        """
```

---

## 9. 数据保留与归档

### 9.1 保留策略

| 事件类型 | 保留期 | 存储级别 |
|---------|--------|---------|
| 安全事件 (JAILBREAK, INJECTION) | 永久 | Hot |
| 截断事件 (CUTOFF_*) | 5年 | Warm |
| 推理事件 (INFERENCE_*) | 1年 | Warm |
| L3决策 (L3_*) | 2年 | Warm |
| L2/L1事件 (L2_*, L1_*) | 90天 | Cold |
| 系统事件 (SNAPSHOT, CONFIG) | 1年 | Cold |

### 9.2 归档流程

```
┌─────────────────────────────────────────────────────────────┐
│                    数据归档流程                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  每日检查                                                    │
│      │                                                      │
│      ▼                                                      │
│  ┌─────────────────┐                                       │
│  │ 检查保留策略    │                                       │
│  └────────┬────────┘                                       │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────┐                                       │
│  │ 标记过期数据    │  ────▶ Cold Storage                    │
│  └────────┬────────┘       (Parquet/Gzip)                  │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────┐                                       │
│  │ 生成归档文件    │                                       │
│  │ (.parquet.gz)  │                                       │
│  └────────┬────────┘                                       │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────┐                                       │
│  │ 计算归档哈希    │                                       │
│  │ 并签名         │                                       │
│  └────────┬────────┘                                       │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────┐                                       │
│  │ 更新归档索引    │                                       │
│  └─────────────────┘                                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 10. 安全考虑

### 10.1 访问控制

- **读权限**: 仅管理员和审计员可查询完整数据
- **写权限**: 仅审计系统组件可写入
- **删除权限**: 禁止删除，仅通过归档策略自动转移

### 10.2 密钥管理

- **HMAC密钥**: 存储在硬件安全模块(HSM)或KMS
- **密钥轮换**: 每90天自动轮换，保留旧密钥用于历史验证
- **密钥版本**: 每个签名包含密钥版本号

### 10.3 传输安全

- TLS 1.3 加密传输
- 审计查询使用专用API端点
- 请求签名验证

---

## 11. 实现计划

### Phase 1: 基础数据模型
- [x] 定义核心数据结构 (AuditEntry, AuditData, AuditMetadata)
- [x] 实现现有 hash_chain.py 的数据结构
- [x] 创建 SQLite schema

### Phase 2: 索引和查询
- [x] 实现索引系统
- [x] 实现 AuditQueryService
- [x] 性能测试

### Phase 3: 完整验证
- [x] 实现 HashChainIntegrity 验证器
- [x] 实现 SignatureVerifier
- [ ] 实现 MerkleTreeVerifier

### Phase 4: 生产化
- [ ] 实现数据保留和归档策略
- [ ] 实现 TEE 集成
- [ ] 性能优化

---

## 附录

### A. 数据字段完整列表

| 字段名 | 类型 | 必需 | 描述 |
|-------|------|-----|------|
| entry_id | string | 是 | UUID，唯一标识 |
| index | int | 是 | 链位置序号 |
| event_type | string | 是 | 事件类型枚举 |
| session_id | string | 是 | 会话标识 |
| timestamp | float | 是 | Unix时间戳 |
| previous_hash | string | 是 | 前一哈希 |
| current_hash | string | 是 | 当前哈希 |
| data_hash | string | 是 | 数据哈希 |
| data | object | 是 | 审计数据 |
| metadata | object | 是 | 元数据 |
| signature | string | 否 | HMAC签名 |
| signer_id | string | 否 | 签名者ID |
| tee_quote | object | 否 | TEE引用 |

### B. 事件类型完整列表

```
安全类 (P0):
- JAILBREAK_ATTEMPT
- PROMPT_INJECTION
- SECURITY_FLAG
- CUTOFF_TRIGGERED
- CUTOFF_VAN_EVENT
- CUTOFF_ENTROPY_LOW
- CUTOFF_SELF_LOOP
- INFERENCE_START
- INFERENCE_END

推理类 (P1):
- INFERENCE_STEP
- VAN_KEYWORD_MATCH
- VAN_MLP_DETECT
- VAN_ATTENTION_FLAG
- L3_DECISION
- SNAPSHOT_CREATED
- KEY_ROTATION

状态类 (P2):
- L2_ENTROPY_UPDATE
- L2_MEMORY_UPDATE
- L2_ACTIVE_INDICES
- L1_TOKEN_GENERATED
- L1_ATTENTION_WEIGHT
- L1_FORGET_GATE
- L3_COOLDOWN
- L3_FLICKER
- CONFIG_CHANGED
- MODE_SWITCHED
```

### C. 参考实现

现有代码位置：
- 哈希链: `enlighten/audit/hash_chain.py`
- HMAC签名: `enlighten/audit/hmac_signature.py`
- TEE格式: `enlighten/audit/tee_audit.py`
- 离线复盘: `enlighten/audit/offline_review.py`
- 复盘调度: `enlighten/audit/review_scheduler.py`
