"""
HMAC Signature - HMAC签名持久化存储模块
提供审计条目的HMAC签名生成、持久化存储和完整性验证
"""

import hmac
import hashlib
import json
import time
import os
import sqlite3
import secrets
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, asdict, field
from pathlib import Path
from contextlib import contextmanager


@dataclass
class SignatureRecord:
    """签名记录"""
    record_id: str
    entry_id: str
    key_id: str
    signature: str
    timestamp: float
    data_hash: str
    signed_fields: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> 'SignatureRecord':
        return cls(**d)


@dataclass
class KeyVersion:
    """密钥版本"""
    key_id: str
    secret_key: str
    algorithm: str
    created_at: float
    valid_until: Optional[float] = None
    is_active: bool = True

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> 'KeyVersion':
        return cls(**d)


class HMACSignature:
    """
    HMAC签名实现类

    功能:
    - 生成HMAC签名
    - 持久化存储签名记录
    - 验证签名完整性
    - 支持密钥轮换
    - 性能测试支持
    """

    def __init__(
        self,
        db_path: str = "audit_signatures.db",
        algorithm: str = "sha256",
        key_rotation_interval: int = 86400
    ):
        self.db_path = db_path
        self.algorithm = algorithm
        self.key_rotation_interval = key_rotation_interval
        self.hash_func = getattr(hashlib, algorithm)

        self._current_key: Optional[KeyVersion] = None
        self._init_database()

    def _init_database(self) -> None:
        """初始化数据库"""
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS signature_records (
                    record_id TEXT PRIMARY KEY,
                    entry_id TEXT NOT NULL,
                    key_id TEXT NOT NULL,
                    signature TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    data_hash TEXT NOT NULL,
                    signed_fields_json TEXT NOT NULL,
                    metadata_json TEXT DEFAULT '{}',
                    created_at REAL DEFAULT (julianday('now'))
                );

                CREATE TABLE IF NOT EXISTS key_versions (
                    key_id TEXT PRIMARY KEY,
                    secret_key TEXT NOT NULL,
                    algorithm TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    valid_until REAL,
                    is_active INTEGER DEFAULT 1
                );

                CREATE INDEX IF NOT EXISTS idx_entry_id ON signature_records(entry_id);
                CREATE INDEX IF NOT EXISTS idx_key_id ON signature_records(key_id);
                CREATE INDEX IF NOT EXISTS idx_timestamp ON signature_records(timestamp);
                CREATE INDEX IF NOT EXISTS idx_data_hash ON signature_records(data_hash);
            """)

    @contextmanager
    def _get_connection(self):
        """获取数据库连接"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def initialize_key(self, key_id: Optional[str] = None, secret_key: Optional[str] = None) -> KeyVersion:
        """
        初始化密钥

        Args:
            key_id: 密钥ID（可选，自动生成）
            secret_key: 密钥（可选，自动生成）

        Returns:
            KeyVersion: 密钥版本对象
        """
        if key_id is None:
            key_id = f"key_{secrets.token_hex(8)}"

        if secret_key is None:
            secret_key = secrets.token_hex(32)

        key_version = KeyVersion(
            key_id=key_id,
            secret_key=secret_key,
            algorithm=self.algorithm,
            created_at=time.time(),
            is_active=True
        )

        with self._get_connection() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO key_versions
                   (key_id, secret_key, algorithm, created_at, is_active)
                   VALUES (?, ?, ?, ?, ?)""",
                (key_version.key_id, key_version.secret_key, key_version.algorithm,
                 key_version.created_at, 1 if key_version.is_active else 0)
            )

        self._current_key = key_version
        return key_version

    def get_current_key(self) -> Optional[KeyVersion]:
        """获取当前密钥"""
        if self._current_key is not None:
            return self._current_key

        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM key_versions WHERE is_active = 1 ORDER BY created_at DESC LIMIT 1"
            ).fetchone()

            if row:
                self._current_key = KeyVersion(
                    key_id=row['key_id'],
                    secret_key=row['secret_key'],
                    algorithm=row['algorithm'],
                    created_at=row['created_at'],
                    valid_until=row['valid_until'],
                    is_active=bool(row['is_active'])
                )
                return self._current_key

        return None

    def sign(
        self,
        entry_id: str,
        data: Dict[str, Any],
        key_id: Optional[str] = None,
        timestamp: Optional[float] = None
    ) -> SignatureRecord:
        """
        生成HMAC签名并持久化

        Args:
            entry_id: 条目ID
            data: 要签名的数据
            key_id: 密钥ID（可选，使用当前密钥）
            timestamp: 时间戳（可选）

        Returns:
            SignatureRecord: 签名记录
        """
        if key_id is None:
            current_key = self.get_current_key()
            if current_key is None:
                current_key = self.initialize_key()
            key_id = current_key.key_id
        else:
            current_key = self._get_key(key_id)
            if current_key is None:
                raise ValueError(f"Unknown key_id: {key_id}")

        timestamp = timestamp or time.time()

        data_hash = self._compute_data_hash(data)

        message = self._prepare_sign_message(entry_id, data_hash, timestamp, key_id)

        signature = hmac.new(
            current_key.secret_key.encode(),
            message.encode(),
            self.hash_func
        ).hexdigest()

        record = SignatureRecord(
            record_id=f"sig_{secrets.token_hex(16)}",
            entry_id=entry_id,
            key_id=key_id,
            signature=signature,
            timestamp=timestamp,
            data_hash=data_hash,
            signed_fields=data.copy()
        )

        self._save_signature_record(record)

        return record

    def _compute_data_hash(self, data: Dict[str, Any]) -> str:
        """计算数据哈希"""
        serialized = json.dumps(data, sort_keys=True, default=str).encode()
        return self.hash_func(serialized).hexdigest()

    def _prepare_sign_message(
        self,
        entry_id: str,
        data_hash: str,
        timestamp: float,
        key_id: str
    ) -> str:
        """准备签名消息"""
        return f"{entry_id}:{data_hash}:{timestamp}:{key_id}"

    def _get_key(self, key_id: str) -> Optional[KeyVersion]:
        """获取指定密钥"""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM key_versions WHERE key_id = ?", (key_id,)
            ).fetchone()

            if row:
                return KeyVersion(
                    key_id=row['key_id'],
                    secret_key=row['secret_key'],
                    algorithm=row['algorithm'],
                    created_at=row['created_at'],
                    valid_until=row['valid_until'],
                    is_active=bool(row['is_active'])
                )
        return None

    def _save_signature_record(self, record: SignatureRecord) -> None:
        """保存签名记录到数据库"""
        with self._get_connection() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO signature_records
                   (record_id, entry_id, key_id, signature, timestamp, data_hash,
                    signed_fields_json, metadata_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record.record_id,
                    record.entry_id,
                    record.key_id,
                    record.signature,
                    record.timestamp,
                    record.data_hash,
                    json.dumps(record.signed_fields, default=str),
                    json.dumps(record.metadata, default=str)
                )
            )

    def verify(
        self,
        entry_id: str,
        data: Dict[str, Any],
        signature: str,
        key_id: str,
        timestamp: Optional[float] = None,
        tolerance: float = 300
    ) -> bool:
        """
        验证HMAC签名

        Args:
            entry_id: 条目ID
            data: 原始数据
            signature: 签名
            key_id: 密钥ID
            timestamp: 时间戳
            tolerance: 时间容差（秒）

        Returns:
            bool: 签名是否有效
        """
        key = self._get_key(key_id)
        if key is None:
            return False

        if timestamp is not None:
            current_time = time.time()
            if abs(current_time - timestamp) > tolerance:
                return False

        data_hash = self._compute_data_hash(data)

        message = self._prepare_sign_message(entry_id, data_hash, timestamp or time.time(), key_id)

        expected_signature = hmac.new(
            key.secret_key.encode(),
            message.encode(),
            self.hash_func
        ).hexdigest()

        return hmac.compare_digest(signature, expected_signature)

    def verify_from_db(
        self,
        entry_id: str,
        data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        从数据库验证签名

        Args:
            entry_id: 条目ID
            data: 原始数据（可选，如果不提供则从数据库获取）

        Returns:
            bool: 签名是否有效
        """
        record = self.get_signature_record(entry_id)
        if record is None:
            return False

        if data is None:
            data = record.signed_fields

        return self.verify(
            entry_id=entry_id,
            data=data,
            signature=record.signature,
            key_id=record.key_id,
            timestamp=record.timestamp
        )

    def get_signature_record(self, entry_id: str) -> Optional[SignatureRecord]:
        """
        获取签名记录

        Args:
            entry_id: 条目ID

        Returns:
            SignatureRecord: 签名记录，不存在则返回None
        """
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM signature_records WHERE entry_id = ? ORDER BY timestamp DESC LIMIT 1",
                (entry_id,)
            ).fetchone()

            if row:
                return SignatureRecord(
                    record_id=row['record_id'],
                    entry_id=row['entry_id'],
                    key_id=row['key_id'],
                    signature=row['signature'],
                    timestamp=row['timestamp'],
                    data_hash=row['data_hash'],
                    signed_fields=json.loads(row['signed_fields_json']),
                    metadata=json.loads(row['metadata_json'])
                )
        return None

    def get_all_signature_records(
        self,
        key_id: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        limit: int = 1000
    ) -> List[SignatureRecord]:
        """
        获取所有签名记录

        Args:
            key_id: 密钥ID过滤
            start_time: 起始时间
            end_time: 结束时间
            limit: 返回数量限制

        Returns:
            List[SignatureRecord]: 签名记录列表
        """
        query = "SELECT * FROM signature_records WHERE 1=1"
        params = []

        if key_id:
            query += " AND key_id = ?"
            params.append(key_id)

        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)

        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        records = []
        with self._get_connection() as conn:
            for row in conn.execute(query, params):
                records.append(SignatureRecord(
                    record_id=row['record_id'],
                    entry_id=row['entry_id'],
                    key_id=row['key_id'],
                    signature=row['signature'],
                    timestamp=row['timestamp'],
                    data_hash=row['data_hash'],
                    signed_fields=json.loads(row['signed_fields_json']),
                    metadata=json.loads(row['metadata_json'])
                ))

        return records

    def rotate_key(
        self,
        new_key_id: Optional[str] = None,
        retire_old_after: Optional[float] = None
    ) -> KeyVersion:
        """
        轮换密钥

        Args:
            new_key_id: 新密钥ID
            retire_old_after: 旧密钥失效时间

        Returns:
            KeyVersion: 新密钥
        """
        current_key = self.get_current_key()
        if current_key:
            with self._get_connection() as conn:
                retire_time = retire_old_after or (time.time() + self.key_rotation_interval)
                conn.execute(
                    "UPDATE key_versions SET is_active = 0, valid_until = ? WHERE key_id = ?",
                    (retire_time, current_key.key_id)
                )

        new_key = self.initialize_key(key_id=new_key_id)
        self._current_key = new_key
        return new_key

    def verify_with_key_history(
        self,
        entry_id: str,
        data: Dict[str, Any],
        signature: str,
        timestamp: Optional[float] = None,
        tolerance: float = 300
    ) -> Tuple[bool, Optional[str]]:
        """
        使用密钥历史验证签名

        Args:
            entry_id: 条目ID
            data: 原始数据
            signature: 签名
            timestamp: 时间戳
            tolerance: 时间容差

        Returns:
            Tuple[bool, Optional[str]]: (验证是否成功, 成功的密钥ID)
        """
        with self._get_connection() as conn:
            rows = conn.execute(
                """SELECT * FROM key_versions
                   WHERE created_at <= ?
                   ORDER BY created_at DESC""",
                (timestamp or time.time(),)
            ).fetchall()

            for row in rows:
                key = KeyVersion(
                    key_id=row['key_id'],
                    secret_key=row['secret_key'],
                    algorithm=row['algorithm'],
                    created_at=row['created_at'],
                    valid_until=row['valid_until'],
                    is_active=bool(row['is_active'])
                )

                if self.verify(
                    entry_id=entry_id,
                    data=data,
                    signature=signature,
                    key_id=key.key_id,
                    timestamp=timestamp,
                    tolerance=tolerance
                ):
                    return True, key.key_id

        return False, None

    def verify_chain_integrity(
        self,
        entries: List[Dict[str, Any]],
        entry_ids: List[str],
        signatures: List[str],
        key_ids: List[str],
        timestamps: List[float]
    ) -> Tuple[bool, List[int]]:
        """
        批量验证链完整性

        Args:
            entries: 条目数据列表
            entry_ids: 条目ID列表
            signatures: 签名列表
            key_ids: 密钥ID列表
            timestamps: 时间戳列表

        Returns:
            Tuple[bool, List[int]]: (是否全部验证通过, 失败索引列表)
        """
        if not (len(entries) == len(entry_ids) == len(signatures) == len(key_ids) == len(timestamps)):
            raise ValueError("All lists must have the same length")

        failed_indices = []

        for i in range(len(entries)):
            valid = self.verify(
                entry_id=entry_ids[i],
                data=entries[i],
                signature=signatures[i],
                key_id=key_ids[i],
                timestamp=timestamps[i]
            )

            if not valid:
                failed_indices.append(i)

        return len(failed_indices) == 0, failed_indices

    def export_signatures(
        self,
        output_path: str,
        format: str = "jsonl",
        key_id: Optional[str] = None
    ) -> int:
        """
        导出签名记录

        Args:
            output_path: 输出路径
            format: 导出格式（jsonl/json）
            key_id: 密钥ID过滤

        Returns:
            int: 导出记录数
        """
        records = self.get_all_signature_records(key_id=key_id, limit=100000)

        if format == "jsonl":
            with open(output_path, 'w', encoding='utf-8') as f:
                for record in records:
                    f.write(json.dumps(record.to_dict(), default=str) + '\n')
        else:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump([r.to_dict() for r in records], f, indent=2, default=str)

        return len(records)

    def import_signatures(self, input_path: str) -> int:
        """
        导入签名记录

        Args:
            input_path: 输入路径

        Returns:
            int: 导入记录数
        """
        count = 0
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                data = json.loads(line)
                record = SignatureRecord.from_dict(data)
                self._save_signature_record(record)
                count += 1

        return count

    def delete_signature_record(self, entry_id: str) -> bool:
        """
        删除签名记录

        Args:
            entry_id: 条目ID

        Returns:
            bool: 是否删除成功
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM signature_records WHERE entry_id = ?",
                (entry_id,)
            )
            return cursor.rowcount > 0

    def get_signature_stats(self) -> Dict[str, Any]:
        """
        获取签名统计信息

        Returns:
            Dict: 统计信息
        """
        with self._get_connection() as conn:
            total_records = conn.execute(
                "SELECT COUNT(*) as count FROM signature_records"
            ).fetchone()['count']

            total_keys = conn.execute(
                "SELECT COUNT(*) as count FROM key_versions"
            ).fetchone()['count']

            active_keys = conn.execute(
                "SELECT COUNT(*) as count FROM key_versions WHERE is_active = 1"
            ).fetchone()['count']

            oldest_record = conn.execute(
                "SELECT MIN(timestamp) as min_ts FROM signature_records"
            ).fetchone()['min_ts']

            newest_record = conn.execute(
                "SELECT MAX(timestamp) as max_ts FROM signature_records"
            ).fetchone()['max_ts']

            return {
                "total_records": total_records,
                "total_keys": total_keys,
                "active_keys": active_keys,
                "oldest_record_timestamp": oldest_record,
                "newest_record_timestamp": newest_record,
                "db_path": self.db_path
            }


class SignatureVerifier:
    """
    签名验证器（无状态版本）

    用于不依赖持久化存储的验证场景
    """

    def __init__(self, secret_key: str, algorithm: str = "sha256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.hash_func = getattr(hashlib, algorithm)

    def _prepare_sign_message(
        self,
        entry_id: str,
        data_hash: str,
        timestamp: float,
        key_id: str
    ) -> str:
        """准备签名消息"""
        return f"{entry_id}:{data_hash}:{timestamp}:{key_id}"

    def _compute_data_hash(self, data: Dict[str, Any]) -> str:
        """计算数据哈希"""
        serialized = json.dumps(data, sort_keys=True, default=str).encode()
        return self.hash_func(serialized).hexdigest()

    def verify(
        self,
        entry_id: str,
        data: Dict[str, Any],
        signature: str,
        key_id: str,
        timestamp: Optional[float] = None
    ) -> bool:
        """
        验证签名

        Args:
            entry_id: 条目ID
            data: 原始数据
            signature: 签名
            key_id: 密钥ID
            timestamp: 时间戳

        Returns:
            bool: 签名是否有效
        """
        data_hash = self._compute_data_hash(data)
        message = self._prepare_sign_message(entry_id, data_hash, timestamp or time.time(), key_id)

        expected_signature = hmac.new(
            self.secret_key.encode(),
            message.encode(),
            self.hash_func
        ).hexdigest()

        return hmac.compare_digest(signature, expected_signature)


def benchmark_signing(num_operations: int = 1000) -> Dict[str, float]:
    """
    签名生成性能测试

    Args:
        num_operations: 操作数量

    Returns:
        Dict: 性能统计
    """
    import time
    import tempfile
    import os

    db_fd, db_path = tempfile.mkstemp(suffix='.db')
    hmac_sig = HMACSignature(db_path=db_path)
    hmac_sig.initialize_key()

    entry_id = "test_entry_001"
    data = {
        "action": "token_generated",
        "component": "L1",
        "session_id": "test_session",
        "input_hash": "a" * 64,
        "output_hash": "b" * 64
    }

    start_time = time.perf_counter()
    for _ in range(num_operations):
        hmac_sig.sign(entry_id, data)
    end_time = time.perf_counter()

    total_time = end_time - start_time
    ops_per_second = num_operations / total_time

    os.close(db_fd)
    os.unlink(db_path)

    return {
        "num_operations": num_operations,
        "total_time_seconds": total_time,
        "ops_per_second": ops_per_second,
        "avg_time_ms": (total_time * 1000) / num_operations
    }


def benchmark_verification(num_operations: int = 1000) -> Dict[str, float]:
    """
    签名验证性能测试

    Args:
        num_operations: 操作数量

    Returns:
        Dict: 性能统计
    """
    import time
    import tempfile
    import os

    db_fd, db_path = tempfile.mkstemp(suffix='.db')
    hmac_sig = HMACSignature(db_path=db_path)
    hmac_sig.initialize_key()

    entry_id = "test_entry_001"
    data = {
        "action": "token_generated",
        "component": "L1",
        "session_id": "test_session",
        "input_hash": "a" * 64,
        "output_hash": "b" * 64
    }

    record = hmac_sig.sign(entry_id, data)

    start_time = time.perf_counter()
    for _ in range(num_operations):
        hmac_sig.verify(
            entry_id=record.entry_id,
            data=data,
            signature=record.signature,
            key_id=record.key_id,
            timestamp=record.timestamp
        )
    end_time = time.perf_counter()

    total_time = end_time - start_time
    ops_per_second = num_operations / total_time

    os.close(db_fd)
    os.unlink(db_path)

    return {
        "num_operations": num_operations,
        "total_time_seconds": total_time,
        "ops_per_second": ops_per_second,
        "avg_time_ms": (total_time * 1000) / num_operations
    }