"""
测试Engram锚点和rewind机制
"""

import torch
import pytest
from enlighten.memory.working_memory import WorkingMemory


class TestEngram:
    """测试Engram锚点和rewind机制"""

    def test_create_engram(self):
        """测试创建Engram锚点"""
        memory = WorkingMemory(memory_size=10, embedding_dim=16)
        memory.initialize(1, torch.device('cpu'))

        # 初始化一些内存内容
        key = torch.randn(1, 5, 16)
        value = torch.randn(1, 5, 16)
        memory.update(key, value)

        # 创建Engram锚点
        engram_index = memory.create_engram("test_engram")
        assert engram_index == 0

        # 验证Engram列表
        engrams = memory.list_engrams()
        assert len(engrams) == 1
        assert engrams[0]['name'] == "test_engram"
        assert engrams[0]['index'] == 0

    def test_rewind_to_engram(self):
        """测试回退到Engram锚点"""
        memory = WorkingMemory(memory_size=10, embedding_dim=16)
        memory.initialize(1, torch.device('cpu'))

        # 第一次更新
        key1 = torch.randn(1, 5, 16)
        value1 = torch.randn(1, 5, 16)
        memory.update(key1, value1)
        memory.create_engram("before_change")

        # 第二次更新
        key2 = torch.randn(1, 5, 16)
        value2 = torch.randn(1, 5, 16)
        memory.update(key2, value2)

        # 回退到锚点
        success = memory.rewind_to_engram(0)
        assert success

        # 验证Engram列表
        engrams = memory.list_engrams()
        assert len(engrams) == 1

    def test_max_engrams(self):
        """测试最大Engram数量限制"""
        memory = WorkingMemory(memory_size=10, embedding_dim=16, max_engrams=3)
        memory.initialize(1, torch.device('cpu'))

        # 创建超过限制的Engram
        for i in range(5):
            key = torch.randn(1, 3, 16)
            value = torch.randn(1, 3, 16)
            memory.update(key, value)
            memory.create_engram(f"engram_{i}")

        # 验证只保留最新的3个
        engrams = memory.list_engrams()
        assert len(engrams) == 3
        assert engrams[0]['name'] == "engram_2"
        assert engrams[1]['name'] == "engram_3"
        assert engrams[2]['name'] == "engram_4"

    def test_invalid_engram_index(self):
        """测试无效的Engram索引"""
        memory = WorkingMemory(memory_size=10, embedding_dim=16)
        memory.initialize(1, torch.device('cpu'))

        # 尝试回退到不存在的Engram
        success = memory.rewind_to_engram(999)
        assert not success

    def test_reset_clears_engrams(self):
        """测试重置会清除Engram"""
        memory = WorkingMemory(memory_size=10, embedding_dim=16)
        memory.initialize(1, torch.device('cpu'))

        # 创建Engram
        key = torch.randn(1, 5, 16)
        value = torch.randn(1, 5, 16)
        memory.update(key, value)
        memory.create_engram("test_engram")
        assert len(memory.list_engrams()) == 1

        # 重置内存
        memory.reset()
        assert len(memory.list_engrams()) == 0

    def test_snapshot_with_engrams(self):
        """测试包含Engram的记忆快照"""
        memory = WorkingMemory(memory_size=10, embedding_dim=16)
        memory.initialize(1, torch.device('cpu'))

        # 创建Engram
        key = torch.randn(1, 5, 16)
        value = torch.randn(1, 5, 16)
        memory.update(key, value)
        memory.create_engram("test_engram")

        # 获取快照
        snapshot = memory.get_memory_snapshot()
        assert 'engrams' in snapshot
        assert len(snapshot['engrams']) == 1

        # 加载快照
        new_memory = WorkingMemory(memory_size=10, embedding_dim=16)
        new_memory.initialize(1, torch.device('cpu'))
        new_memory.load_snapshot(snapshot)
        assert len(new_memory.list_engrams()) == 1
        assert new_memory.list_engrams()[0]['name'] == "test_engram"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])