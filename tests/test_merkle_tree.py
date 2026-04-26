"""
测试默克尔树功能
"""

import pytest
import tempfile
import os
from enlighten.audit import HashChain, HashChainFactory, MerkleTree, MerkleTreeFactory, MerkleTreeManager


class TestMerkleTree:
    """测试默克尔树功能"""

    def setup_method(self):
        """设置测试环境"""
        # 创建内存哈希链
        self.hash_chain = HashChainFactory.create_memory_chain()
        # 构建测试数据
        for i in range(10):
            self.hash_chain.append(
                event_type=f"test_event_{i}",
                session_id="test_session",
                data={"value": i, "message": f"Test message {i}"}
            )

    def test_build_from_entries(self):
        """测试从哈希链条目构建默克尔树"""
        manager = MerkleTreeFactory.create_merkle_tree_manager(self.hash_chain)
        root_hash = manager.build_tree()
        
        assert root_hash is not None
        assert len(root_hash) > 0
        
        tree_info = manager.get_tree_info()
        assert tree_info["leaf_count"] == 10
        assert tree_info["height"] >= 4  # 2^4 = 16 > 10

    def test_get_root_hash(self):
        """测试获取根哈希"""
        manager = MerkleTreeFactory.create_merkle_tree_manager(self.hash_chain)
        root_hash1 = manager.build_tree()
        root_hash2 = manager.get_root_hash()
        
        assert root_hash1 == root_hash2

    def test_generate_and_verify_proof(self):
        """测试生成和验证默克尔证明"""
        manager = MerkleTreeFactory.create_merkle_tree_manager(self.hash_chain)
        manager.build_tree()
        
        # 测试所有叶子节点
        for i in range(10):
            proof = manager.generate_proof(i)
            assert proof is not None
            
            # 验证证明
            entry = self.hash_chain.get_entry(i)
            assert entry is not None
            
            is_valid = manager.verify_entry(entry, proof)
            assert is_valid, f"Proof verification failed for index {i}"

    def test_verify_integrity(self):
        """测试验证树的完整性"""
        manager = MerkleTreeFactory.create_merkle_tree_manager(self.hash_chain)
        manager.build_tree()
        
        is_valid = manager.verify_integrity()
        assert is_valid

    def test_empty_tree(self):
        """测试空树"""
        empty_chain = HashChainFactory.create_memory_chain()
        manager = MerkleTreeFactory.create_merkle_tree_manager(empty_chain)
        
        root_hash = manager.build_tree()
        assert root_hash == ""
        
        proof = manager.generate_proof(0)
        assert proof is None

    def test_single_entry(self):
        """测试单个条目"""
        single_chain = HashChainFactory.create_memory_chain()
        single_chain.append(
            event_type="test_event",
            session_id="test_session",
            data={"value": 1, "message": "Test message"}
        )
        
        manager = MerkleTreeFactory.create_merkle_tree_manager(single_chain)
        root_hash = manager.build_tree()
        
        assert root_hash is not None
        
        proof = manager.generate_proof(0)
        assert proof is not None
        
        entry = single_chain.get_entry(0)
        is_valid = manager.verify_entry(entry, proof)
        assert is_valid

    def test_tree_export_import(self):
        """测试树的导出和导入"""
        manager = MerkleTreeFactory.create_merkle_tree_manager(self.hash_chain)
        manager.build_tree()
        
        # 导出树
        tree_data = manager.export_tree()
        assert tree_data is not None
        
        # 创建新的管理器
        new_manager = MerkleTreeFactory.create_merkle_tree_manager(self.hash_chain)
        new_manager.import_tree(tree_data)
        
        # 验证根哈希相同
        assert manager.get_root_hash() == new_manager.get_root_hash()

    def test_tree_height(self):
        """测试树的高度计算"""
        manager = MerkleTreeFactory.create_merkle_tree_manager(self.hash_chain)
        manager.build_tree()
        
        tree_info = manager.get_tree_info()
        height = tree_info["height"]
        
        # 10个叶子节点，高度应该是4 (2^3=8 < 10 ≤ 2^4=16)
        assert height == 4

    def test_merkle_tree_direct(self):
        """直接测试MerkleTree类"""
        # 构建测试数据
        entries = []
        for i in range(5):
            entry = self.hash_chain.get_entry(i)
            if entry:
                entries.append(entry)
        
        tree = MerkleTree()
        root_hash = tree.build_from_entries(entries)
        
        assert root_hash is not None
        assert tree.get_leaf_count() == 5
        
        # 测试生成证明
        proof = tree.generate_proof(2)
        assert proof is not None
        
        # 测试验证证明
        is_valid = tree.verify_proof(proof)
        assert is_valid


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
