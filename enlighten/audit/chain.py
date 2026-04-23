"""
Audit Hash Chain - 审计哈希链
提供不可篡改的审计日志
"""

import hashlib
import json
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import pickle


@dataclass
class AuditEntry:
    """审计条目"""
    index: int
    hash: str
    data: Dict[str, Any]
    timestamp: float
    signature: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> 'AuditEntry':
        return cls(**d)


class AuditHashChain:
    """
    审计哈希链

    特性:
    - 链接: H_i = SHA256(H_{i-1} || data_i)
    - 不可篡改: 任何data_i的修改都会导致后续哈希不匹配
    - 可验证: 从任意点可验证链的完整性
    """

    def __init__(
        self,
        algorithm: str = "sha256",
        initial_hash: str = "0000000000000000000000000000000000000000000000000000000000000000"
    ):
        self.algorithm = algorithm
        self.initial_hash = initial_hash
        self.chain: List[AuditEntry] = []

        self.hash_func = getattr(hashlib, algorithm)

    def _hash_data(self, data: Dict) -> str:
        """
        计算数据的哈希
        """
        serialized = json.dumps(data, sort_keys=True, default=str).encode()
        return self.hash_func(serialized).hexdigest()

    def _hash_link(self, prev_hash: str, current_hash: str) -> str:
        """
        计算链接哈希
        """
        combined = f"{prev_hash}{current_hash}".encode()
        return self.hash_func(combined).hexdigest()

    def append(self, data: Dict, signature: Optional[str] = None) -> str:
        """
        追加新条目

        Args:
            data: 要记录的数据
            signature: 可选的HMAC签名

        Returns:
            link_hash: 新条目的链接哈希
        """
        # 添加前一个哈希到数据中，增强完整性
        if self.chain:
            data["_prev_hash"] = self.chain[-1].hash
        else:
            data["_prev_hash"] = self.initial_hash

        current_hash = self._hash_data(data)

        if self.chain:
            prev_hash = self.chain[-1].hash
            link_hash = self._hash_link(prev_hash, current_hash)
        else:
            link_hash = self._hash_link(self.initial_hash, current_hash)

        entry = AuditEntry(
            index=len(self.chain),
            hash=link_hash,
            data=data,
            timestamp=time.time(),
            signature=signature
        )

        self.chain.append(entry)

        return link_hash

    def verify(self) -> bool:
        """
        验证链的完整性

        Returns:
            valid: 链是否完整
        """
        if not self.chain:
            return True

        for i in range(1, len(self.chain)):
            expected_prev = self.chain[i].data.get("_prev_hash")
            actual_prev = self.chain[i-1].hash

            if expected_prev and expected_prev != actual_prev:
                return False

        for i in range(len(self.chain)):
            computed_hash = self._hash_data(self.chain[i].data)

            prev_hash = self.initial_hash if i == 0 else self.chain[i-1].hash
            expected_link = self._hash_link(prev_hash, computed_hash)

            if self.chain[i].hash != expected_link:
                return False

        return True

    def verify_from(self, index: int) -> bool:
        """
        从指定索引验证链

        Args:
            index: 起始索引

        Returns:
            valid: 从index开始的链是否完整
        """
        if index >= len(self.chain):
            return False

        return self.verify()

    def get_entry(self, index: int) -> Optional[AuditEntry]:
        """
        获取指定索引的条目
        """
        if 0 <= index < len(self.chain):
            return self.chain[index]
        return None

    def get_chain_length(self) -> int:
        """
        获取链长度
        """
        return len(self.chain)

    def save(self, path: str) -> None:
        """
        保存链到文件
        """
        with open(path, 'wb') as f:
            pickle.dump(self.chain, f)

    def load(self, path: str) -> None:
        """
        从文件加载链
        """
        with open(path, 'rb') as f:
            self.chain = pickle.load(f)

    def clear(self) -> None:
        """
        清空链
        """
        self.chain = []

    def get_summary(self) -> Dict:
        """
        获取链摘要
        """
        if not self.chain:
            return {"length": 0, "valid": True, "first_hash": None, "last_hash": None}

        return {
            "length": len(self.chain),
            "valid": self.verify(),
            "first_hash": self.chain[0].hash,
            "last_hash": self.chain[-1].hash,
            "first_timestamp": self.chain[0].timestamp,
            "last_timestamp": self.chain[-1].timestamp
        }


class MerkleTree:
    """
    Merkle树
    用于批量验证和高效证明
    """

    def __init__(self):
        self.leaves = []
        self.tree = []

    def build(self, data_list: List[Dict]) -> None:
        """
        构建Merkle树

        Args:
            data_list: 数据列表
        """
        self.leaves = [hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()
                       for d in data_list]

        current_level = self.leaves[:]
        self.tree = [current_level]

        while len(current_level) > 1:
            next_level = []
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = current_level[i + 1] if i + 1 < len(current_level) else left
                combined = hashlib.sha256((left + right).encode()).hexdigest()
                next_level.append(combined)
            self.tree.append(next_level)
            current_level = next_level

    def get_root(self) -> str:
        """
        获取根哈希
        """
        if not self.tree:
            return ""
        return self.tree[-1][0] if self.tree[-1] else ""

    def verify(self, data: Dict, proof: List[str], root: str) -> bool:
        """
        验证数据在树中

        Args:
            data: 要验证的数据
            proof: 验证路径
            root: Merkle根
        """
        leaf = hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()

        current = leaf
        for p in proof:
            current = hashlib.sha256((current + p).encode()).hexdigest()

        return current == root
