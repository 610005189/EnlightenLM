"""
HMAC Signature Tests - HMAC签名功能测试
测试签名生成、持久化存储、完整性验证和性能
"""

import unittest
import time
import os
import tempfile
import json
from typing import Dict, Any

from enlighten.audit.hmac_signature import (
    HMACSignature,
    SignatureVerifier,
    SignatureRecord,
    KeyVersion,
    benchmark_signing,
    benchmark_verification
)


class TestHMACSignature(unittest.TestCase):
    """HMACSignature功能测试"""

    def setUp(self):
        """测试前准备"""
        self.db_fd, self.db_path = tempfile.mkstemp(suffix='.db')
        self.hmac_sig = HMACSignature(db_path=self.db_path)
        self.hmac_sig.initialize_key(key_id="test_key_001")

        self.test_data = {
            "action": "token_generated",
            "component": "L1",
            "session_id": "test_session_001",
            "input_hash": "a" * 64,
            "output_hash": "b" * 64,
            "entropy_mean": 0.45,
            "entropy_variance": 0.02,
            "tau": 0.7,
            "cutoff": False
        }

    def tearDown(self):
        """测试后清理"""
        os.close(self.db_fd)
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)

    def test_sign_and_verify(self):
        """测试签名生成和验证"""
        entry_id = "entry_001"

        record = self.hmac_sig.sign(entry_id, self.test_data)

        self.assertIsNotNone(record.signature)
        self.assertEqual(record.entry_id, entry_id)
        self.assertEqual(record.key_id, "test_key_001")

        valid = self.hmac_sig.verify(
            entry_id=entry_id,
            data=self.test_data,
            signature=record.signature,
            key_id=record.key_id,
            timestamp=record.timestamp
        )

        self.assertTrue(valid)

    def test_verify_invalid_signature(self):
        """测试无效签名验证"""
        entry_id = "entry_002"

        record = self.hmac_sig.sign(entry_id, self.test_data)

        invalid_signature = "invalid_signature_here"

        valid = self.hmac_sig.verify(
            entry_id=entry_id,
            data=self.test_data,
            signature=invalid_signature,
            key_id=record.key_id,
            timestamp=record.timestamp
        )

        self.assertFalse(valid)

    def test_verify_tampered_data(self):
        """测试数据篡改检测"""
        entry_id = "entry_003"

        record = self.hmac_sig.sign(entry_id, self.test_data)

        tampered_data = self.test_data.copy()
        tampered_data["entropy_mean"] = 0.99

        valid = self.hmac_sig.verify(
            entry_id=entry_id,
            data=tampered_data,
            signature=record.signature,
            key_id=record.key_id,
            timestamp=record.timestamp
        )

        self.assertFalse(valid)

    def test_verify_from_db(self):
        """测试从数据库验证签名"""
        entry_id = "entry_004"

        record = self.hmac_sig.sign(entry_id, self.test_data)

        valid = self.hmac_sig.verify_from_db(entry_id)

        self.assertTrue(valid)

    def test_get_signature_record(self):
        """测试获取签名记录"""
        entry_id = "entry_005"

        record1 = self.hmac_sig.sign(entry_id, self.test_data)

        record2 = self.hmac_sig.get_signature_record(entry_id)

        self.assertIsNotNone(record2)
        self.assertEqual(record1.signature, record2.signature)
        self.assertEqual(record1.entry_id, record2.entry_id)

    def test_multiple_entries(self):
        """测试多个条目签名"""
        entry_ids = [f"entry_{i:03d}" for i in range(10)]

        for entry_id in entry_ids:
            record = self.hmac_sig.sign(entry_id, self.test_data)
            self.assertIsNotNone(record.signature)

        all_records = self.hmac_sig.get_all_signature_records()
        self.assertEqual(len(all_records), 10)

    def test_key_rotation(self):
        """测试密钥轮换"""
        entry_id = "entry_rotation_test"

        record1 = self.hmac_sig.sign(entry_id, self.test_data)
        old_key_id = record1.key_id

        new_key = self.hmac_sig.rotate_key(new_key_id="new_key_001")
        self.assertEqual(new_key.key_id, "new_key_001")

        record2 = self.hmac_sig.sign(entry_id, self.test_data)
        self.assertEqual(record2.key_id, "new_key_001")
        self.assertNotEqual(record1.signature, record2.signature)

    def test_verify_with_key_history(self):
        """测试使用密钥历史验证"""
        entry_id = "entry_history_test"

        record1 = self.hmac_sig.sign(entry_id, self.test_data)

        self.hmac_sig.rotate_key(new_key_id="new_key_002")

        record2 = self.hmac_sig.sign(entry_id, self.test_data)

        valid_old, key_id = self.hmac_sig.verify_with_key_history(
            entry_id=entry_id,
            data=self.test_data,
            signature=record1.signature,
            timestamp=record1.timestamp
        )

        self.assertTrue(valid_old)
        self.assertEqual(key_id, record1.key_id)

    def test_chain_integrity_verification(self):
        """测试链完整性验证"""
        num_entries = 5
        entries = []
        entry_ids = []
        signatures = []
        key_ids = []
        timestamps = []

        for i in range(num_entries):
            entry_id = f"chain_entry_{i:03d}"
            data = self.test_data.copy()
            data["index"] = i

            record = self.hmac_sig.sign(entry_id, data)

            entries.append(data)
            entry_ids.append(entry_id)
            signatures.append(record.signature)
            key_ids.append(record.key_id)
            timestamps.append(record.timestamp)

        valid, failed = self.hmac_sig.verify_chain_integrity(
            entries, entry_ids, signatures, key_ids, timestamps
        )

        self.assertTrue(valid)
        self.assertEqual(len(failed), 0)

    def test_chain_integrity_with_tampered_entry(self):
        """测试链完整性验证（篡改数据）"""
        num_entries = 5
        entries = []
        entry_ids = []
        signatures = []
        key_ids = []
        timestamps = []

        for i in range(num_entries):
            entry_id = f"chain_entry_{i:03d}"
            data = self.test_data.copy()
            data["index"] = i

            record = self.hmac_sig.sign(entry_id, data)

            entries.append(data)
            entry_ids.append(entry_id)
            signatures.append(record.signature)
            key_ids.append(record.key_id)
            timestamps.append(record.timestamp)

        entries[2]["entropy_mean"] = 0.99

        valid, failed = self.hmac_sig.verify_chain_integrity(
            entries, entry_ids, signatures, key_ids, timestamps
        )

        self.assertFalse(valid)
        self.assertIn(2, failed)

    def test_export_import_signatures(self):
        """测试导出导入签名"""
        for i in range(5):
            entry_id = f"export_entry_{i:03d}"
            self.hmac_sig.sign(entry_id, self.test_data)

        export_path = tempfile.mktemp(suffix='.jsonl')
        import_db_fd, import_db_path = tempfile.mkstemp(suffix='.db')

        try:
            count = self.hmac_sig.export_signatures(export_path, format="jsonl")
            self.assertEqual(count, 5)

            new_hmac_sig = HMACSignature(db_path=import_db_path)
            new_hmac_sig.initialize_key()

            import_count = new_hmac_sig.import_signatures(export_path)
            self.assertEqual(import_count, 5)

        finally:
            if os.path.exists(export_path):
                os.unlink(export_path)
            os.close(import_db_fd)
            if os.path.exists(import_db_path):
                os.unlink(import_db_path)

    def test_delete_signature_record(self):
        """测试删除签名记录"""
        entry_id = "entry_delete_test"

        self.hmac_sig.sign(entry_id, self.test_data)

        deleted = self.hmac_sig.delete_signature_record(entry_id)
        self.assertTrue(deleted)

        record = self.hmac_sig.get_signature_record(entry_id)
        self.assertIsNone(record)

    def test_signature_stats(self):
        """测试签名统计"""
        for i in range(3):
            self.hmac_sig.sign(f"stats_entry_{i:03d}", self.test_data)

        stats = self.hmac_sig.get_signature_stats()

        self.assertEqual(stats["total_records"], 3)
        self.assertEqual(stats["total_keys"], 1)
        self.assertEqual(stats["active_keys"], 1)

    def test_timestamp_tolerance(self):
        """测试时间戳容差"""
        entry_id = "entry_tolerance_test"

        current_time = time.time()

        record = self.hmac_sig.sign(entry_id, self.test_data, timestamp=current_time)

        valid_current = self.hmac_sig.verify(
            entry_id=entry_id,
            data=self.test_data,
            signature=record.signature,
            key_id=record.key_id,
            timestamp=current_time,
            tolerance=300
        )
        self.assertTrue(valid_current)

        old_time = current_time - 600

        valid_old = self.hmac_sig.verify(
            entry_id=entry_id,
            data=self.test_data,
            signature=record.signature,
            key_id=record.key_id,
            timestamp=old_time,
            tolerance=300
        )
        self.assertFalse(valid_old)


class TestSignatureVerifier(unittest.TestCase):
    """SignatureVerifier无状态验证器测试"""

    def test_verify_with_secret_key(self):
        """测试使用密钥验证"""
        secret_key = "test_secret_key_12345"

        verifier = SignatureVerifier(secret_key=secret_key)

        data = {
            "action": "test_action",
            "value": 123
        }

        entry_id = "entry_001"
        timestamp = time.time()
        key_id = "test_key"

        data_hash = verifier._compute_data_hash(data)
        message = verifier._prepare_sign_message(entry_id, data_hash, timestamp, key_id)

        import hmac
        import hashlib
        expected_signature = hmac.new(
            secret_key.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()

        valid = verifier.verify(
            entry_id=entry_id,
            data=data,
            signature=expected_signature,
            key_id=key_id,
            timestamp=timestamp
        )

        self.assertTrue(valid)


class TestPerformance(unittest.TestCase):
    """性能测试"""

    def test_signing_performance(self):
        """测试签名生成性能"""
        results = benchmark_signing(num_operations=100)

        print(f"\n签名生成性能:")
        print(f"  操作次数: {results['num_operations']}")
        print(f"  总耗时: {results['total_time_seconds']:.4f}秒")
        print(f"  每秒操作数: {results['ops_per_second']:.2f}")
        print(f"  平均耗时: {results['avg_time_ms']:.4f}毫秒")

        self.assertGreater(results['ops_per_second'], 100)

    def test_verification_performance(self):
        """测试签名验证性能"""
        results = benchmark_verification(num_operations=100)

        print(f"\n签名验证性能:")
        print(f"  操作次数: {results['num_operations']}")
        print(f"  总耗时: {results['total_time_seconds']:.4f}秒")
        print(f"  每秒操作数: {results['ops_per_second']:.2f}")
        print(f"  平均耗时: {results['avg_time_ms']:.4f}毫秒")

        self.assertGreater(results['ops_per_second'], 100)

    def test_persistence_performance(self):
        """测试持久化存储性能"""
        import time

        db_fd, db_path = tempfile.mkstemp(suffix='.db')
        hmac_sig = HMACSignature(db_path=db_path)
        hmac_sig.initialize_key()

        test_data = {
            "action": "test_action",
            "data": "x" * 100
        }

        num_operations = 100
        start_time = time.perf_counter()

        for i in range(num_operations):
            hmac_sig.sign(f"entry_{i:03d}", test_data)

        end_time = time.perf_counter()
        total_time = end_time - start_time
        ops_per_second = num_operations / total_time

        print(f"\n持久化存储性能:")
        print(f"  操作次数: {num_operations}")
        print(f"  总耗时: {total_time:.4f}秒")
        print(f"  每秒操作数: {ops_per_second:.2f}")

        os.close(db_fd)
        os.unlink(db_path)

        self.assertGreater(ops_per_second, 50)


class TestKeyVersion(unittest.TestCase):
    """KeyVersion密钥版本测试"""

    def test_key_version_creation(self):
        """测试密钥版本创建"""
        key = KeyVersion(
            key_id="key_001",
            secret_key="secret_123",
            algorithm="sha256",
            created_at=time.time()
        )

        self.assertEqual(key.key_id, "key_001")
        self.assertTrue(key.is_active)

    def test_key_version_serialization(self):
        """测试密钥版本序列化"""
        key = KeyVersion(
            key_id="key_002",
            secret_key="secret_456",
            algorithm="sha256",
            created_at=time.time()
        )

        key_dict = key.to_dict()
        restored_key = KeyVersion.from_dict(key_dict)

        self.assertEqual(key.key_id, restored_key.key_id)
        self.assertEqual(key.secret_key, restored_key.secret_key)
        self.assertEqual(key.algorithm, restored_key.algorithm)


class TestSignatureRecord(unittest.TestCase):
    """SignatureRecord签名记录测试"""

    def test_signature_record_creation(self):
        """测试签名记录创建"""
        record = SignatureRecord(
            record_id="sig_001",
            entry_id="entry_001",
            key_id="key_001",
            signature="abc123",
            timestamp=time.time(),
            data_hash="hash123",
            signed_fields={"action": "test"}
        )

        self.assertEqual(record.record_id, "sig_001")
        self.assertEqual(record.entry_id, "entry_001")

    def test_signature_record_serialization(self):
        """测试签名记录序列化"""
        record = SignatureRecord(
            record_id="sig_002",
            entry_id="entry_002",
            key_id="key_001",
            signature="def456",
            timestamp=time.time(),
            data_hash="hash456",
            signed_fields={"action": "test2"}
        )

        record_dict = record.to_dict()
        restored_record = SignatureRecord.from_dict(record_dict)

        self.assertEqual(record.record_id, restored_record.record_id)
        self.assertEqual(record.signature, restored_record.signature)


if __name__ == '__main__':
    unittest.main(verbosity=2)