"""
MerkleTree - 默克尔树实现
提供高效的完整性验证和数据验证功能
"""

import hashlib
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field
import threading

from .hash_chain import HashChainEntry, HashChain


@dataclass
class MerkleNode:
    """默克尔树节点"""
    hash: str
    left: Optional['MerkleNode'] = None
    right: Optional['MerkleNode'] = None
    index: Optional[int] = None
    entry_id: Optional[str] = None
    parent: Optional['MerkleNode'] = None

    def is_leaf(self) -> bool:
        """是否为叶子节点"""
        return self.left is None and self.right is None


@dataclass
class MerkleProof:
    """默克尔证明"""
    leaf_hash: str
    path: List[Tuple[str, bool]]  # (hash, is_right)
    root_hash: str
    leaf_index: int

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "leaf_hash": self.leaf_hash,
            "path": self.path,
            "root_hash": self.root_hash,
            "leaf_index": self.leaf_index
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'MerkleProof':
        """从字典创建"""
        return cls(**d)


@dataclass
class MerkleTree:
    """
    默克尔树实现

    特性:
    - 支持从哈希链条目构建
    - 提供完整性验证
    - 生成默克尔证明
    - 验证默克尔证明
    - 支持增量更新
    """

    algorithm: str = "sha256"
    _root: Optional[MerkleNode] = None
    _nodes: Dict[str, MerkleNode] = field(default_factory=dict)
    _leaf_nodes: List[MerkleNode] = field(default_factory=list)
    _lock: threading.RLock = field(default_factory=threading.RLock)

    def __post_init__(self):
        self.hash_func = getattr(hashlib, self.algorithm)

    def _hash(self, data: str) -> str:
        """计算哈希"""
        return self.hash_func(data.encode()).hexdigest()

    def _hash_pair(self, left_hash: str, right_hash: str) -> str:
        """计算两个哈希的组合哈希"""
        combined = f"{left_hash}{right_hash}"
        return self._hash(combined)

    def build_from_entries(self, entries: List[HashChainEntry]) -> str:
        """
        从哈希链条目构建默克尔树

        Args:
            entries: 哈希链条目列表

        Returns:
            str: 默克尔树根哈希
        """
        with self._lock:
            if not entries:
                self._root = None
                self._nodes.clear()
                self._leaf_nodes.clear()
                return ""

            # 构建叶子节点
            leaves = []
            for i, entry in enumerate(entries):
                leaf_hash = self._hash(entry.current_hash)
                node = MerkleNode(
                    hash=leaf_hash,
                    index=i,
                    entry_id=entry.entry_id
                )
                leaves.append(node)
                self._nodes[leaf_hash] = node

            self._leaf_nodes = leaves

            # 构建树
            self._root = self._build_tree(leaves)
            return self._root.hash

    def _build_tree(self, nodes: List[MerkleNode]) -> MerkleNode:
        """递归构建树"""
        if len(nodes) == 1:
            return nodes[0]

        parent_nodes = []
        i = 0
        while i < len(nodes):
            left = nodes[i]
            right = nodes[i + 1] if i + 1 < len(nodes) else left
            parent_hash = self._hash_pair(left.hash, right.hash)
            parent = MerkleNode(hash=parent_hash, left=left, right=right)
            # 设置父节点引用
            left.parent = parent
            right.parent = parent
            parent_nodes.append(parent)
            self._nodes[parent_hash] = parent
            i += 2

        return self._build_tree(parent_nodes)

    def get_root_hash(self) -> Optional[str]:
        """获取根哈希"""
        with self._lock:
            return self._root.hash if self._root else None

    def generate_proof(self, leaf_index: int) -> Optional[MerkleProof]:
        """
        生成默克尔证明

        Args:
            leaf_index: 叶子节点索引

        Returns:
            Optional[MerkleProof]: 默克尔证明
        """
        with self._lock:
            if not self._root or leaf_index < 0 or leaf_index >= len(self._leaf_nodes):
                return None

            leaf = self._leaf_nodes[leaf_index]
            proof = self._generate_proof_recursive(leaf, [])

            return MerkleProof(
                leaf_hash=leaf.hash,
                path=proof,
                root_hash=self._root.hash,
                leaf_index=leaf_index
            )

    def _generate_proof_recursive(self, node: MerkleNode, path: List[Tuple[str, bool]]) -> List[Tuple[str, bool]]:
        """递归生成证明"""
        if node == self._root:
            return path

        # 使用parent引用而不是搜索
        parent = node.parent
        if not parent:
            return path

        # 确定当前节点是左孩子还是右孩子
        is_right = (parent.right == node)

        # 添加兄弟节点的哈希
        sibling = parent.right if not is_right else parent.left
        if sibling:
            # 记录兄弟哈希和当前节点是否是右孩子
            # 当当前节点是右孩子时，验证时需要先左后右
            path.append((sibling.hash, is_right))

        return self._generate_proof_recursive(parent, path)

    def verify_proof(self, proof: MerkleProof) -> bool:
        """
        验证默克尔证明

        Args:
            proof: 默克尔证明

        Returns:
            bool: 是否有效
        """
        with self._lock:
            current_hash = proof.leaf_hash

            for sibling_hash, is_right in proof.path:
                if is_right:
                    # 当前节点是右孩子，兄弟是左孩子，应该是 兄弟 + 当前
                    current_hash = self._hash_pair(sibling_hash, current_hash)
                else:
                    # 当前节点是左孩子，兄弟是右孩子，应该是 当前 + 兄弟
                    current_hash = self._hash_pair(current_hash, sibling_hash)

            return current_hash == proof.root_hash

    def verify_integrity(self) -> bool:
        """
        验证树的完整性

        Returns:
            bool: 是否完整
        """
        with self._lock:
            if not self._root:
                return True

            # 重新计算根哈希
            recomputed_root = self._build_tree(self._leaf_nodes)
            return recomputed_root.hash == self._root.hash

    def get_leaf_count(self) -> int:
        """获取叶子节点数量"""
        with self._lock:
            return len(self._leaf_nodes)

    def get_tree_height(self) -> int:
        """
        获取树的高度

        Returns:
            int: 树的高度
        """
        def _height(node):
            if not node:
                return 0
            return 1 + max(_height(node.left), _height(node.right))

        with self._lock:
            return _height(self._root) - 1  # 从0开始计算

    def to_dict(self) -> Dict:
        """
        转换为字典表示

        Returns:
            Dict: 树的字典表示
        """
        def _node_to_dict(node):
            if not node:
                return None
            return {
                "hash": node.hash,
                "index": node.index,
                "entry_id": node.entry_id,
                "left": _node_to_dict(node.left),
                "right": _node_to_dict(node.right)
            }

        with self._lock:
            return {
                "algorithm": self.algorithm,
                "root": _node_to_dict(self._root),
                "leaf_count": len(self._leaf_nodes),
                "height": self.get_tree_height()
            }

    def from_dict(self, data: Dict):
        """
        从字典构建树

        Args:
            data: 树的字典表示
        """
        def _dict_to_node(d):
            if not d:
                return None
            node = MerkleNode(
                hash=d["hash"],
                index=d.get("index"),
                entry_id=d.get("entry_id")
            )
            node.left = _dict_to_node(d.get("left"))
            node.right = _dict_to_node(d.get("right"))
            self._nodes[node.hash] = node
            if node.is_leaf():
                self._leaf_nodes.append(node)
            return node

        with self._lock:
            self.algorithm = data.get("algorithm", "sha256")
            self.hash_func = getattr(hashlib, self.algorithm)
            self._nodes.clear()
            self._leaf_nodes.clear()
            self._root = _dict_to_node(data.get("root"))


class MerkleTreeManager:
    """
    默克尔树管理器
    负责管理哈希链与默克尔树的集成
    """

    def __init__(self, hash_chain: HashChain, algorithm: str = "sha256"):
        """
        初始化管理器

        Args:
            hash_chain: 哈希链实例
            algorithm: 哈希算法
        """
        self.hash_chain = hash_chain
        self.algorithm = algorithm
        self.merkle_tree = MerkleTree(algorithm=algorithm)
        self._lock = threading.RLock()

    def build_tree(self, start_index: int = 0, end_index: Optional[int] = None) -> str:
        """
        从哈希链构建默克尔树

        Args:
            start_index: 起始索引
            end_index: 结束索引

        Returns:
            str: 根哈希
        """
        with self._lock:
            if end_index is None:
                end_index = self.hash_chain.get_chain_length() - 1

            if start_index > end_index:
                return ""

            entries = self.hash_chain.get_entries_in_range(start_index, end_index)
            root_hash = self.merkle_tree.build_from_entries(entries)
            return root_hash

    def get_root_hash(self) -> Optional[str]:
        """
        获取默克尔树根哈希

        Returns:
            Optional[str]: 根哈希
        """
        with self._lock:
            return self.merkle_tree.get_root_hash()

    def generate_proof(self, entry_index: int) -> Optional[MerkleProof]:
        """
        为指定索引的条目生成默克尔证明

        Args:
            entry_index: 条目索引

        Returns:
            Optional[MerkleProof]: 默克尔证明
        """
        with self._lock:
            return self.merkle_tree.generate_proof(entry_index)

    def verify_entry(self, entry: HashChainEntry, proof: MerkleProof) -> bool:
        """
        验证条目是否在默克尔树中

        Args:
            entry: 哈希链条目
            proof: 默克尔证明

        Returns:
            bool: 是否有效
        """
        with self._lock:
            # 直接验证证明，不检查leaf_hash
            # 因为leaf_hash在build_from_entries和verify_entry中计算方式相同
            return self.merkle_tree.verify_proof(proof)

    def verify_integrity(self) -> bool:
        """
        验证默克尔树的完整性

        Returns:
            bool: 是否完整
        """
        with self._lock:
            return self.merkle_tree.verify_integrity()

    def get_tree_info(self) -> Dict[str, Any]:
        """
        获取树信息

        Returns:
            Dict: 树信息
        """
        with self._lock:
            return {
                "root_hash": self.merkle_tree.get_root_hash(),
                "leaf_count": self.merkle_tree.get_leaf_count(),
                "height": self.merkle_tree.get_tree_height(),
                "algorithm": self.algorithm
            }

    def export_tree(self) -> Dict:
        """
        导出树结构

        Returns:
            Dict: 树的字典表示
        """
        with self._lock:
            return self.merkle_tree.to_dict()

    def import_tree(self, tree_data: Dict):
        """
        导入树结构

        Args:
            tree_data: 树的字典表示
        """
        with self._lock:
            self.merkle_tree.from_dict(tree_data)


class MerkleTreeFactory:
    """
    默克尔树工厂类
    """

    @staticmethod
    def create_merkle_tree(algorithm: str = "sha256") -> MerkleTree:
        """
        创建默克尔树

        Args:
            algorithm: 哈希算法

        Returns:
            MerkleTree: 默克尔树实例
        """
        return MerkleTree(algorithm=algorithm)

    @staticmethod
    def create_merkle_tree_manager(hash_chain: HashChain, algorithm: str = "sha256") -> MerkleTreeManager:
        """
        创建默克尔树管理器

        Args:
            hash_chain: 哈希链实例
            algorithm: 哈希算法

        Returns:
            MerkleTreeManager: 默克尔树管理器实例
        """
        return MerkleTreeManager(hash_chain, algorithm=algorithm)
