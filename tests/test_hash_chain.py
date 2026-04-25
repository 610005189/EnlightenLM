"""
HashChain 测试模块
测试哈希链的持久化存储、完整性和性能
"""

import unittest
import tempfile
import os
import time
import shutil
from typing import List, Dict, Any

from enlighten.audit.hash_chain import (
    HashChain,
    HashChainFactory,
    StorageBackend,
    HashChainEntry,
    IntegrityReport
)


class TestHashChainBasics(unittest.TestCase):
    """基础功能测试"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_chain.db")

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_create_memory_chain(self):
        """测试创建内存后端哈希链"""
        chain = HashChainFactory.create_memory_chain()
        self.assertEqual(chain.get_chain_length(), 0)
        self.assertIsNotNone(chain.chain_id)

    def test_create_sqlite_chain(self):
        """测试创建SQLite后端哈希链"""
        chain = HashChainFactory.create_sqlite_chain(self.db_path)
        self.assertEqual(chain.get_chain_length(), 0)
        self.assertTrue(os.path.exists(self.db_path))

    def test_append_single_entry(self):
        """测试追加单个条目"""
        chain = HashChainFactory.create_memory_chain()

        entry = chain.append(
            event_type="TEST_EVENT",
            session_id="session-001",
            data={"action": "test", "value": 123}
        )

        self.assertIsNotNone(entry.entry_id)
        self.assertEqual(entry.index, 0)
        self.assertEqual(entry.event_type, "TEST_EVENT")
        self.assertEqual(chain.get_chain_length(), 1)

    def test_append_multiple_entries(self):
        """测试追加多个条目"""
        chain = HashChainFactory.create_memory_chain()

        for i in range(10):
            entry = chain.append(
                event_type="TEST_EVENT",
                session_id="session-001",
                data={"index": i, "value": i * 2}
            )
            self.assertEqual(entry.index, i)

        self.assertEqual(chain.get_chain_length(), 10)

    def test_hash_chain_link(self):
        """测试哈希链链接"""
        chain = HashChainFactory.create_memory_chain()

        entries = []
        for i in range(5):
            entry = chain.append(
                event_type="TEST_EVENT",
                session_id="session-001",
                data={"index": i}
            )
            entries.append(entry)

        for i in range(1, len(entries)):
            self.assertEqual(
                entries[i].previous_hash,
                entries[i - 1].current_hash,
                f"链接验证失败于索引 {i}"
            )


class TestHashChainPersistence(unittest.TestCase):
    """持久化存储测试"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_sqlite_persistence(self):
        """测试SQLite持久化"""
        db_path = os.path.join(self.temp_dir, "persist_test.db")
        chain1 = HashChainFactory.create_sqlite_chain(db_path)

        for i in range(20):
            chain1.append(
                event_type="PERSIST_TEST",
                session_id="session-001",
                data={"index": i}
            )

        chain2 = HashChainFactory.create_sqlite_chain(db_path)
        self.assertEqual(chain2.get_chain_length(), 20)

        summary = chain2.get_summary()
        self.assertEqual(summary["length"], 20)

    def test_jsonl_persistence(self):
        """测试JSONL持久化"""
        jsonl_path = os.path.join(self.temp_dir, "test_chain.jsonl")
        chain1 = HashChainFactory.create_jsonl_chain(jsonl_path)

        for i in range(15):
            chain1.append(
                event_type="JSONL_TEST",
                session_id="session-002",
                data={"index": i}
            )

        chain2 = HashChainFactory.create_jsonl_chain(jsonl_path)
        self.assertEqual(chain2.get_chain_length(), 15)

    def test_sqlite_batch_append(self):
        """测试SQLite批量追加"""
        db_path = os.path.join(self.temp_dir, "batch_test.db")
        chain = HashChainFactory.create_sqlite_chain(db_path)

        entries_data = [{"value": i} for i in range(100)]
        result = chain.batch_append(
            entries=entries_data,
            event_type="BATCH_TEST",
            session_id="session-003"
        )

        self.assertEqual(len(result), 100)
        self.assertEqual(chain.get_chain_length(), 100)


class TestHashChainIntegrity(unittest.TestCase):
    """完整性验证测试"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_verify_empty_chain(self):
        """测试空链验证"""
        chain = HashChainFactory.create_memory_chain()
        report = chain.verify_integrity()

        self.assertTrue(report.is_valid)
        self.assertEqual(report.total_entries, 0)

    def test_verify_valid_chain(self):
        """测试有效链验证"""
        chain = HashChainFactory.create_memory_chain()

        for i in range(50):
            chain.append(
                event_type="VALID_TEST",
                session_id="session-001",
                data={"index": i, "timestamp": time.time()}
            )

        report = chain.verify_integrity()

        self.assertTrue(report.is_valid)
        self.assertEqual(report.total_entries, 50)
        self.assertEqual(report.verified_entries, 50)

    def test_verify_detect_tampering(self):
        """测试检测数据篡改"""
        db_path = os.path.join(self.temp_dir, "tamper_test.db")
        chain = HashChainFactory.create_sqlite_chain(db_path)

        for i in range(30):
            chain.append(
                event_type="TAMPER_TEST",
                session_id="session-001",
                data={"index": i, "original": True}
            )

        import sqlite3
        conn = sqlite3.connect(db_path)
        conn.execute("""
            UPDATE hash_chain
            SET data_json = ?, data_hash = ?
            WHERE chain_index = ?
        """, ('{"index": 999, "original": true, "tampered": true}', 'fake_hash_after_tampering', 15))
        conn.commit()
        conn.close()

        chain2 = HashChainFactory.create_sqlite_chain(db_path)
        report = chain2.verify_integrity()

        self.assertFalse(report.is_valid)
        self.assertIsNotNone(report.first_break_index)
        self.assertIsNotNone(report.break_reason)

    def test_verify_from_checkpoint(self):
        """测试从检查点验证"""
        db_path = os.path.join(self.temp_dir, "checkpoint_test.db")
        chain = HashChainFactory.create_sqlite_chain(db_path)

        for i in range(100):
            chain.append(
                event_type="CHECKPOINT_TEST",
                session_id="session-001",
                data={"index": i}
            )

        checkpoint = chain.create_checkpoint()
        self.assertEqual(checkpoint["index"], 99)

        report = chain.verify_from_checkpoint(50)
        self.assertTrue(report.is_valid)
        self.assertEqual(report.verified_entries, 49)

    def test_verify_entry(self):
        """测试单个条目验证"""
        chain = HashChainFactory.create_memory_chain()

        chain.append(
            event_type="ENTRY_TEST",
            session_id="session-001",
            data={"value": 42}
        )

        valid, error = chain.verify_entry(0)
        self.assertTrue(valid)
        self.assertEqual(error, "")


class TestHashChainPerformance(unittest.TestCase):
    """性能测试"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_memory_write_performance(self):
        """测试内存写入性能"""
        chain = HashChainFactory.create_memory_chain()

        start_time = time.time()
        for i in range(1000):
            chain.append(
                event_type="PERF_TEST",
                session_id="session-001",
                data={"index": i, "data": "x" * 100}
            )
        elapsed = time.time() - start_time

        print(f"\n内存写入 1000 条目耗时: {elapsed:.4f}秒")
        print(f"平均每条目: {elapsed / 1000 * 1000:.2f}毫秒")

        self.assertLess(elapsed, 10.0)

    def test_sqlite_write_performance(self):
        """测试SQLite写入性能"""
        db_path = os.path.join(self.temp_dir, "perf_test.db")
        chain = HashChainFactory.create_sqlite_chain(db_path)

        start_time = time.time()
        for i in range(500):
            chain.append(
                event_type="PERF_TEST",
                session_id="session-001",
                data={"index": i, "data": "x" * 100}
            )
        elapsed = time.time() - start_time

        print(f"\nSQLite写入 500 条目耗时: {elapsed:.4f}秒")
        print(f"平均每条目: {elapsed / 500 * 1000:.2f}毫秒")

        self.assertLess(elapsed, 30.0)

    def test_batch_write_performance(self):
        """测试批量写入性能"""
        db_path = os.path.join(self.temp_dir, "batch_perf.db")
        chain = HashChainFactory.create_sqlite_chain(db_path)

        entries_data = [{"index": i, "data": "x" * 100} for i in range(500)]

        start_time = time.time()
        chain.batch_append(
            entries=entries_data,
            event_type="BATCH_PERF",
            session_id="session-001"
        )
        elapsed = time.time() - start_time

        print(f"\n批量写入 500 条目耗时: {elapsed:.4f}秒")
        print(f"平均每条目: {elapsed / 500 * 1000:.2f}毫秒")

        self.assertEqual(chain.get_chain_length(), 500)

    def test_verify_performance(self):
        """测试验证性能"""
        db_path = os.path.join(self.temp_dir, "verify_perf.db")
        chain = HashChainFactory.create_sqlite_chain(db_path)

        for i in range(200):
            chain.append(
                event_type="VERIFY_PERF",
                session_id="session-001",
                data={"index": i}
            )

        start_time = time.time()
        report = chain.verify_integrity()
        elapsed = time.time() - start_time

        print(f"\n验证 200 条目耗时: {elapsed:.4f}秒")
        print(f"平均每条目: {elapsed / 200 * 1000:.2f}毫秒")

        self.assertTrue(report.is_valid)

    def test_checkpoint_verify_performance(self):
        """测试检查点验证性能"""
        db_path = os.path.join(self.temp_dir, "checkpoint_perf.db")
        chain = HashChainFactory.create_sqlite_chain(db_path)

        for i in range(1000):
            chain.append(
                event_type="CP_PERF",
                session_id="session-001",
                data={"index": i}
            )

        start_time = time.time()
        report = chain.verify_from_checkpoint(500)
        elapsed = time.time() - start_time

        print(f"\n检查点验证 (从500开始) 499 条目耗时: {elapsed:.4f}秒")
        print(f"平均每条目: {elapsed / 499 * 1000:.2f}毫秒")

        self.assertTrue(report.is_valid)


class TestHashChainExport(unittest.TestCase):
    """导出功能测试"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_export_chain(self):
        """测试导出链数据"""
        chain = HashChainFactory.create_memory_chain()

        for i in range(20):
            chain.append(
                event_type="EXPORT_TEST",
                session_id="session-001",
                data={"index": i}
            )

        exported = chain.export_chain(5, 15)

        self.assertEqual(len(exported), 11)
        self.assertEqual(exported[0]["index"], 5)
        self.assertEqual(exported[-1]["index"], 15)

    def test_export_full_chain(self):
        """测试导出完整链"""
        db_path = os.path.join(self.temp_dir, "export_test.db")
        chain = HashChainFactory.create_sqlite_chain(db_path)

        for i in range(30):
            chain.append(
                event_type="FULL_EXPORT",
                session_id="session-001",
                data={"index": i}
            )

        exported = chain.export_chain()

        self.assertEqual(len(exported), 30)


class TestHashChainSummary(unittest.TestCase):
    """摘要功能测试"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_empty_chain_summary(self):
        """测试空链摘要"""
        chain = HashChainFactory.create_memory_chain()
        summary = chain.get_summary()

        self.assertEqual(summary["length"], 0)
        self.assertIsNone(summary["first_hash"])
        self.assertIsNone(summary["last_hash"])

    def test_chain_summary(self):
        """测试链摘要"""
        db_path = os.path.join(self.temp_dir, "summary_test.db")
        chain = HashChainFactory.create_sqlite_chain(db_path)

        for i in range(10):
            chain.append(
                event_type="SUMMARY_TEST",
                session_id="session-001",
                data={"index": i}
            )

        summary = chain.get_summary()

        self.assertEqual(summary["length"], 10)
        self.assertIsNotNone(summary["first_hash"])
        self.assertIsNotNone(summary["last_hash"])
        self.assertEqual(summary["backend"], "sqlite")


if __name__ == "__main__":
    unittest.main(verbosity=2)
