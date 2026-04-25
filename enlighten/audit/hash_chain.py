"""
HashChain - 哈希链持久化存储实现
提供不可篡改的审计日志存储和完整性验证
"""

import hashlib
import json
import sqlite3
import time
import uuid
import hmac
import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
import threading
import pickle


class StorageBackend(Enum):
    """存储后端类型"""
    SQLITE = "sqlite"
    JSONL = "jsonl"
    MEMORY = "memory"


@dataclass
class HashChainEntry:
    """哈希链条目"""
    entry_id: str
    index: int
    previous_hash: str
    current_hash: str
    data_hash: str
    event_type: str
    timestamp: float
    session_id: str
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    signature: Optional[str] = None
    signer_id: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "entry_id": self.entry_id,
            "index": self.index,
            "previous_hash": self.previous_hash,
            "current_hash": self.current_hash,
            "data_hash": self.data_hash,
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "data": self.data,
            "metadata": self.metadata,
            "signature": self.signature,
            "signer_id": self.signer_id
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'HashChainEntry':
        return cls(**d)


@dataclass
class IntegrityReport:
    """完整性报告"""
    is_valid: bool
    total_entries: int
    verified_entries: int
    first_break_index: Optional[int] = None
    break_reason: Optional[str] = None
    errors: List[Dict] = field(default_factory=list)

    def add_error(self, index: int, error_type: str, details: str):
        self.errors.append({
            "index": index,
            "error_type": error_type,
            "details": details
        })

    def to_dict(self) -> Dict:
        return {
            "is_valid": self.is_valid,
            "total_entries": self.total_entries,
            "verified_entries": self.verified_entries,
            "first_break_index": self.first_break_index,
            "break_reason": self.break_reason,
            "errors": self.errors
        }


@dataclass
class ChainMetadata:
    """链元数据"""
    chain_id: str
    created_at: float
    last_updated: float
    total_entries: int
    storage_backend: str
    algorithm: str
    initial_hash: str
    version: str = "1.0"

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> 'ChainMetadata':
        return cls(**d)


class HashChain:
    """
    哈希链实现类

    特性:
    - 持久化存储: 支持SQLite、JSONL和内存三种后端
    - 完整性验证: 支持整链验证和增量验证
    - 性能优化: 支持批量写入、检查点、快照
    - 线程安全: 支持多线程并发访问
    """

    INITIAL_HASH = "0000000000000000000000000000000000000000000000000000000000000000"

    def __init__(
        self,
        storage_path: str = ":memory:",
        backend: StorageBackend = StorageBackend.SQLITE,
        algorithm: str = "sha256",
        secret_key: Optional[str] = None,
        chain_id: Optional[str] = None
    ):
        self.storage_path = storage_path
        self.backend = backend
        self.algorithm = algorithm
        self.secret_key = secret_key or os.urandom(32).hex()
        self.chain_id = chain_id or str(uuid.uuid4())
        self.hash_func = getattr(hashlib, algorithm)

        self._lock = threading.RLock()
        self._metadata: Optional[ChainMetadata] = None

        self._init_storage()

    def _init_storage(self):
        """初始化存储后端"""
        if self.backend == StorageBackend.SQLITE:
            self._init_sqlite()
        elif self.backend == StorageBackend.JSONL:
            self._init_jsonl()
        elif self.backend == StorageBackend.MEMORY:
            self._entries: List[HashChainEntry] = []
            self._init_metadata()

    def _init_sqlite(self):
        """初始化SQLite数据库"""
        conn = sqlite3.connect(self.storage_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS chain_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS hash_chain (
                entry_id TEXT PRIMARY KEY,
                chain_index INTEGER NOT NULL,
                previous_hash TEXT NOT NULL,
                current_hash TEXT NOT NULL,
                data_hash TEXT NOT NULL,
                event_type TEXT NOT NULL,
                timestamp REAL NOT NULL,
                session_id TEXT NOT NULL,
                data_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                signature TEXT,
                signer_id TEXT
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_chain_index ON hash_chain(chain_index)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_chain_session ON hash_chain(session_id, timestamp)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_chain_event ON hash_chain(event_type, timestamp)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_chain_hash ON hash_chain(current_hash)
        """)

        conn.commit()
        conn.close()

        self._load_metadata()

    def _init_jsonl(self):
        """初始化JSONL文件"""
        if self.storage_path != ":memory:":
            os.makedirs(os.path.dirname(self.storage_path) or ".", exist_ok=True)
            if not os.path.exists(self.storage_path):
                with open(self.storage_path, "w") as f:
                    pass

            metadata_path = self.storage_path + ".meta"
            if not os.path.exists(metadata_path):
                self._init_metadata()
                self._save_metadata()
            else:
                self._load_metadata()

    def _init_metadata(self):
        """初始化元数据"""
        self._metadata = ChainMetadata(
            chain_id=self.chain_id,
            created_at=time.time(),
            last_updated=time.time(),
            total_entries=0,
            storage_backend=self.backend.value,
            algorithm=self.algorithm,
            initial_hash=self.INITIAL_HASH
        )

    def _save_metadata(self):
        """保存元数据"""
        if self._metadata is None:
            return

        self._metadata.last_updated = time.time()

        if self.backend == StorageBackend.SQLITE:
            conn = sqlite3.connect(self.storage_path, check_same_thread=False)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO chain_metadata (key, value) VALUES (?, ?)",
                ("metadata", json.dumps(self._metadata.to_dict()))
            )
            conn.commit()
            conn.close()
        elif self.backend == StorageBackend.JSONL:
            metadata_path = self.storage_path + ".meta"
            with open(metadata_path, "w") as f:
                json.dump(self._metadata.to_dict(), f)

    def _load_metadata(self) -> Optional[ChainMetadata]:
        """加载元数据"""
        if self.backend == StorageBackend.SQLITE:
            conn = sqlite3.connect(self.storage_path, check_same_thread=False)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT value FROM chain_metadata WHERE key = ?",
                ("metadata",)
            )
            row = cursor.fetchone()
            conn.close()
            if row:
                self._metadata = ChainMetadata.from_dict(json.loads(row[0]))
                return self._metadata
        elif self.backend == StorageBackend.JSONL:
            metadata_path = self.storage_path + ".meta"
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    self._metadata = ChainMetadata.from_dict(json.load(f))
                return self._metadata

        self._init_metadata()
        return self._metadata

    def _compute_data_hash(self, data: Dict[str, Any]) -> str:
        """计算数据哈希"""
        serialized = json.dumps(data, sort_keys=True, default=str).encode()
        return self.hash_func(serialized).hexdigest()

    def _compute_link_hash(self, prev_hash: str, data_hash: str) -> str:
        """计算链接哈希"""
        combined = f"{prev_hash}{data_hash}".encode()
        return self.hash_func(combined).hexdigest()

    def _compute_entry_hash(self, entry: HashChainEntry) -> str:
        """计算条目哈希"""
        message = f"{entry.previous_hash}{entry.data_hash}".encode()
        return self.hash_func(message).hexdigest()

    def _sign_entry(self, entry: HashChainEntry) -> str:
        """签名条目"""
        if not self.secret_key:
            return ""

        message = f"{entry.previous_hash}{entry.data_hash}{entry.timestamp}".encode()
        signature = hmac.new(
            self.secret_key.encode(),
            message,
            self.hash_func
        ).hexdigest()
        return signature

    def _verify_entry_signature(self, entry: HashChainEntry) -> bool:
        """验证条目签名"""
        if not entry.signature or not self.secret_key:
            return True

        message = f"{entry.previous_hash}{entry.data_hash}{entry.timestamp}".encode()
        expected = hmac.new(
            self.secret_key.encode(),
            message,
            self.hash_func
        ).hexdigest()
        return hmac.compare_digest(entry.signature, expected)

    def append(
        self,
        event_type: str,
        session_id: str,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        entry_id: Optional[str] = None,
        signer_id: Optional[str] = None
    ) -> HashChainEntry:
        """
        追加新条目到哈希链

        Args:
            event_type: 事件类型
            session_id: 会话ID
            data: 审计数据
            metadata: 元数据
            entry_id: 条目ID（可选，自动生成）
            signer_id: 签名者ID

        Returns:
            HashChainEntry: 新增的条目
        """
        with self._lock:
            previous_entry = self._get_last_entry()
            prev_hash = previous_entry.current_hash if previous_entry else self.INITIAL_HASH
            next_index = (previous_entry.index + 1) if previous_entry else 0

            data_hash = self._compute_data_hash(data)

            entry = HashChainEntry(
                entry_id=entry_id or str(uuid.uuid4()),
                index=next_index,
                previous_hash=prev_hash,
                current_hash="",  # 先计算data_hash再计算current_hash
                data_hash=data_hash,
                event_type=event_type,
                timestamp=time.time(),
                session_id=session_id,
                data=data,
                metadata=metadata or {},
                signer_id=signer_id
            )

            link_hash = self._compute_link_hash(prev_hash, data_hash)
            entry.current_hash = link_hash

            signature = self._sign_entry(entry)
            entry.signature = signature

            self._store_entry(entry)

            self._metadata.total_entries = next_index + 1
            self._save_metadata()

            return entry

    def _store_entry(self, entry: HashChainEntry):
        """存储条目"""
        if self.backend == StorageBackend.SQLITE:
            self._store_entry_sqlite(entry)
        elif self.backend == StorageBackend.JSONL:
            self._store_entry_jsonl(entry)
        elif self.backend == StorageBackend.MEMORY:
            self._entries.append(entry)

    def _store_entry_sqlite(self, entry: HashChainEntry):
        """SQLite存储"""
        conn = sqlite3.connect(self.storage_path, check_same_thread=False)
        conn.execute("""
            INSERT INTO hash_chain (
                entry_id, chain_index, previous_hash, current_hash, data_hash,
                event_type, timestamp, session_id, data_json, metadata_json,
                signature, signer_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            entry.entry_id,
            entry.index,
            entry.previous_hash,
            entry.current_hash,
            entry.data_hash,
            entry.event_type,
            entry.timestamp,
            entry.session_id,
            json.dumps(entry.data),
            json.dumps(entry.metadata),
            entry.signature,
            entry.signer_id
        ))
        conn.commit()
        conn.close()

    def _store_entry_jsonl(self, entry: HashChainEntry):
        """JSONL存储"""
        with open(self.storage_path, "a") as f:
            f.write(json.dumps(entry.to_dict()) + "\n")

    def _get_last_entry(self) -> Optional[HashChainEntry]:
        """获取最后一个条目"""
        if self.backend == StorageBackend.SQLITE:
            return self._get_last_entry_sqlite()
        elif self.backend == StorageBackend.JSONL:
            return self._get_last_entry_jsonl()
        elif self.backend == StorageBackend.MEMORY:
            return self._entries[-1] if self._entries else None
        return None

    def _get_last_entry_sqlite(self) -> Optional[HashChainEntry]:
        """SQLite获取最后一个条目"""
        conn = sqlite3.connect(self.storage_path, check_same_thread=False)
        cursor = conn.execute("""
            SELECT entry_id, chain_index, previous_hash, current_hash, data_hash,
                   event_type, timestamp, session_id, data_json, metadata_json,
                   signature, signer_id
            FROM hash_chain
            ORDER BY chain_index DESC
            LIMIT 1
        """)
        row = cursor.fetchone()
        conn.close()

        if row:
            return HashChainEntry(
                entry_id=row[0],
                index=row[1],
                previous_hash=row[2],
                current_hash=row[3],
                data_hash=row[4],
                event_type=row[5],
                timestamp=row[6],
                session_id=row[7],
                data=json.loads(row[8]),
                metadata=json.loads(row[9]),
                signature=row[10],
                signer_id=row[11]
            )
        return None

    def _get_last_entry_jsonl(self) -> Optional[HashChainEntry]:
        """JSONL获取最后一个条目"""
        if not os.path.exists(self.storage_path):
            return None

        with open(self.storage_path, "r") as f:
            last_line = None
            for line in f:
                last_line = line

        if last_line:
            return HashChainEntry.from_dict(json.loads(last_line))
        return None

    def get_entry(self, index: int) -> Optional[HashChainEntry]:
        """获取指定索引的条目"""
        with self._lock:
            if self.backend == StorageBackend.SQLITE:
                return self._get_entry_sqlite(index)
            elif self.backend == StorageBackend.JSONL:
                return self._get_entry_jsonl(index)
            elif self.backend == StorageBackend.MEMORY:
                if 0 <= index < len(self._entries):
                    return self._entries[index]
        return None

    def _get_entry_sqlite(self, index: int) -> Optional[HashChainEntry]:
        """SQLite获取条目"""
        conn = sqlite3.connect(self.storage_path, check_same_thread=False)
        cursor = conn.execute("""
            SELECT entry_id, chain_index, previous_hash, current_hash, data_hash,
                   event_type, timestamp, session_id, data_json, metadata_json,
                   signature, signer_id
            FROM hash_chain
            WHERE chain_index = ?
        """, (index,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return HashChainEntry(
                entry_id=row[0],
                index=row[1],
                previous_hash=row[2],
                current_hash=row[3],
                data_hash=row[4],
                event_type=row[5],
                timestamp=row[6],
                session_id=row[7],
                data=json.loads(row[8]),
                metadata=json.loads(row[9]),
                signature=row[10],
                signer_id=row[11]
            )
        return None

    def _get_entry_jsonl(self, index: int) -> Optional[HashChainEntry]:
        """JSONL获取条目"""
        if not os.path.exists(self.storage_path):
            return None

        with open(self.storage_path, "r") as f:
            current_index = 0
            for line in f:
                if current_index == index:
                    return HashChainEntry.from_dict(json.loads(line))
                current_index += 1
        return None

    def get_entries_in_range(
        self,
        start_index: int,
        end_index: int
    ) -> List[HashChainEntry]:
        """获取指定索引范围的条目"""
        with self._lock:
            entries = []
            for i in range(start_index, end_index + 1):
                entry = self.get_entry(i)
                if entry:
                    entries.append(entry)
            return entries

    def get_chain_length(self) -> int:
        """获取链长度"""
        if self._metadata:
            return self._metadata.total_entries
        return 0

    def verify_integrity(self) -> IntegrityReport:
        """
        验证整条哈希链的完整性

        Returns:
            IntegrityReport: 完整性报告
        """
        with self._lock:
            report = IntegrityReport(
                is_valid=True,
                total_entries=self.get_chain_length(),
                verified_entries=0
            )

            if report.total_entries == 0:
                return report

            entries = self.get_entries_in_range(0, report.total_entries - 1)

            for i in range(1, len(entries)):
                entry = entries[i]
                prev_entry = entries[i - 1]

                if entry.previous_hash != prev_entry.current_hash:
                    report.is_valid = False
                    report.first_break_index = i
                    report.break_reason = f"前向链接断裂: 期望 {prev_entry.current_hash}, 实际 {entry.previous_hash}"
                    report.add_error(i, "LINK_ERROR", "前向链接验证失败")
                    break

                computed_data_hash = self._compute_data_hash(entry.data)
                if entry.data_hash != computed_data_hash:
                    report.is_valid = False
                    report.first_break_index = i
                    report.break_reason = f"数据哈希不匹配"
                    report.add_error(i, "DATA_HASH_ERROR", "数据哈希验证失败")
                    break

                computed_link = self._compute_link_hash(entry.previous_hash, entry.data_hash)
                if entry.current_hash != computed_link:
                    report.is_valid = False
                    report.first_break_index = i
                    report.break_reason = f"链接哈希不匹配"
                    report.add_error(i, "LINK_HASH_ERROR", "链接哈希验证失败")
                    break

                if not self._verify_entry_signature(entry):
                    report.is_valid = False
                    report.first_break_index = i
                    report.break_reason = f"签名验证失败"
                    report.add_error(i, "SIGNATURE_ERROR", "签名验证失败")
                    break

                report.verified_entries += 1

            if report.is_valid:
                report.verified_entries = len(entries)

            return report

    def verify_from_checkpoint(self, checkpoint_index: int) -> IntegrityReport:
        """
        从检查点验证链完整性

        Args:
            checkpoint_index: 检查点索引

        Returns:
            IntegrityReport: 完整性报告
        """
        with self._lock:
            total = self.get_chain_length()

            if checkpoint_index >= total:
                return IntegrityReport(
                    is_valid=False,
                    total_entries=total,
                    verified_entries=0,
                    first_break_index=checkpoint_index,
                    break_reason="检查点索引超出范围"
                )

            report = IntegrityReport(
                is_valid=True,
                total_entries=total,
                verified_entries=0
            )

            checkpoint_entry = self.get_entry(checkpoint_index)
            if not checkpoint_entry:
                report.is_valid = False
                report.break_reason = "检查点条目不存在"
                return report

            expected_prev_hash = checkpoint_entry.current_hash

            for i in range(checkpoint_index + 1, total):
                entry = self.get_entry(i)
                if not entry:
                    report.is_valid = False
                    report.first_break_index = i
                    report.break_reason = f"索引 {i} 的条目不存在"
                    report.add_error(i, "MISSING_ENTRY", "条目不存在")
                    break

                if entry.previous_hash != expected_prev_hash:
                    report.is_valid = False
                    report.first_break_index = i
                    report.break_reason = f"链接断裂于索引 {i}"
                    report.add_error(i, "LINK_ERROR", "链接验证失败")
                    break

                computed_link = self._compute_link_hash(entry.previous_hash, entry.data_hash)
                if entry.current_hash != computed_link:
                    report.is_valid = False
                    report.first_break_index = i
                    report.break_reason = f"哈希不匹配于索引 {i}"
                    report.add_error(i, "HASH_ERROR", "哈希验证失败")
                    break

                expected_prev_hash = entry.current_hash
                report.verified_entries += 1

            return report

    def verify_entry(self, index: int) -> Tuple[bool, str]:
        """
        验证单个条目

        Args:
            index: 条目索引

        Returns:
            Tuple[bool, str]: (是否有效, 错误信息)
        """
        with self._lock:
            entry = self.get_entry(index)
            if not entry:
                return False, f"条目 {index} 不存在"

            if index > 0:
                prev_entry = self.get_entry(index - 1)
                if not prev_entry:
                    return False, f"前一个条目 {index - 1} 不存在"

                if entry.previous_hash != prev_entry.current_hash:
                    return False, "前向链接验证失败"

            computed_data_hash = self._compute_data_hash(entry.data)
            if entry.data_hash != computed_data_hash:
                return False, "数据哈希验证失败"

            computed_link = self._compute_link_hash(entry.previous_hash, entry.data_hash)
            if entry.current_hash != computed_link:
                return False, "链接哈希验证失败"

            if not self._verify_entry_signature(entry):
                return False, "签名验证失败"

            return True, ""

    def batch_append(
        self,
        entries: List[Dict[str, Any]],
        event_type: str,
        session_id: str
    ) -> List[HashChainEntry]:
        """
        批量追加条目

        Args:
            entries: 条目数据列表
            event_type: 事件类型
            session_id: 会话ID

        Returns:
            List[HashChainEntry]: 新增的条目列表
        """
        with self._lock:
            result = []
            for entry_data in entries:
                entry = self.append(
                    event_type=event_type,
                    session_id=session_id,
                    data=entry_data
                )
                result.append(entry)
            return result

    def create_checkpoint(self) -> Dict[str, Any]:
        """
        创建检查点

        Returns:
            Dict: 检查点信息
        """
        with self._lock:
            last_entry = self._get_last_entry()
            checkpoint = {
                "chain_id": self.chain_id,
                "index": last_entry.index if last_entry else -1,
                "hash": last_entry.current_hash if last_entry else self.INITIAL_HASH,
                "timestamp": time.time(),
                "total_entries": self.get_chain_length()
            }
            return checkpoint

    def get_summary(self) -> Dict[str, Any]:
        """获取链摘要"""
        with self._lock:
            last_entry = self._get_last_entry()
            return {
                "chain_id": self.chain_id,
                "length": self.get_chain_length(),
                "backend": self.backend.value,
                "algorithm": self.algorithm,
                "first_hash": self.INITIAL_HASH if self.get_chain_length() > 0 else None,
                "last_hash": last_entry.current_hash if last_entry else None,
                "created_at": self._metadata.created_at if self._metadata else None,
                "last_updated": self._metadata.last_updated if self._metadata else None
            }

    def export_chain(self, start_index: int = 0, end_index: Optional[int] = None) -> List[Dict]:
        """
        导出链数据

        Args:
            start_index: 起始索引
            end_index: 结束索引

        Returns:
            List[Dict]: 导出的数据
        """
        with self._lock:
            if end_index is None:
                end_index = self.get_chain_length() - 1

            entries = []
            for i in range(start_index, end_index + 1):
                entry = self.get_entry(i)
                if entry:
                    entries.append(entry.to_dict())
            return entries

    def clear(self):
        """清空链"""
        with self._lock:
            if self.backend == StorageBackend.SQLITE:
                conn = sqlite3.connect(self.storage_path, check_same_thread=False)
                conn.execute("DELETE FROM hash_chain")
                conn.commit()
                conn.close()
            elif self.backend == StorageBackend.JSONL:
                if os.path.exists(self.storage_path):
                    os.remove(self.storage_path)
            elif self.backend == StorageBackend.MEMORY:
                self._entries.clear()

            self._init_metadata()
            self._save_metadata()


class HashChainFactory:
    """哈希链工厂类"""

    @staticmethod
    def create_sqlite_chain(
        db_path: str,
        algorithm: str = "sha256",
        secret_key: Optional[str] = None
    ) -> HashChain:
        """创建SQLite后端的哈希链"""
        return HashChain(
            storage_path=db_path,
            backend=StorageBackend.SQLITE,
            algorithm=algorithm,
            secret_key=secret_key
        )

    @staticmethod
    def create_jsonl_chain(
        file_path: str,
        algorithm: str = "sha256",
        secret_key: Optional[str] = None
    ) -> HashChain:
        """创建JSONL后端的哈希链"""
        return HashChain(
            storage_path=file_path,
            backend=StorageBackend.JSONL,
            algorithm=algorithm,
            secret_key=secret_key
        )

    @staticmethod
    def create_memory_chain(
        algorithm: str = "sha256",
        secret_key: Optional[str] = None
    ) -> HashChain:
        """创建内存后端的哈希链"""
        return HashChain(
            storage_path=":memory:",
            backend=StorageBackend.MEMORY,
            algorithm=algorithm,
            secret_key=secret_key
        )
