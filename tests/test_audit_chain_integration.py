"""
测试审计链完整性与哈希链接入推理管线

测试审计链对推理过程的记录和验证能力：
1. 哈希链记录
2. 完整性验证
3. 审计日志获取
"""

import pytest
from enlighten.hybrid_architecture import HybridEnlightenLM
from enlighten.audit.tee_audit import TEEAuditWriter


class TestAuditChainIntegration:
    """测试审计链与推理管线的集成"""

    def test_model_initialization_with_audit(self):
        """测试模型初始化时包含审计写入器"""
        model = HybridEnlightenLM(
            use_bayesian_l3=True,
            use_l3_controller=True
        )

        assert hasattr(model, "audit_writer")
        assert isinstance(model.audit_writer, TEEAuditWriter)

    def test_generate_with_audit_log(self):
        """测试生成时创建审计日志"""
        model = HybridEnlightenLM(
            use_bayesian_l3=True,
            use_l3_controller=True
        )

        initial_entries = len(model.audit_writer.get_entries())

        result = model.generate(
            prompt="你好，请介绍一下自己",
            max_length=100,
            temperature=0.7
        )

        assert result is not None
        assert hasattr(result, "audit_signature")
        assert hasattr(result, "audit_hash_chain")

        final_entries = len(model.audit_writer.get_entries())
        assert final_entries > initial_entries

    def test_audit_signature_format(self):
        """测试审计签名格式"""
        model = HybridEnlightenLM(
            use_bayesian_l3=True,
            use_l3_controller=True
        )

        result = model.generate(
            prompt="你好",
            max_length=50,
            temperature=0.7
        )

        if result.audit_signature:
            # SHA256哈希应该是64字符的十六进制字符串
            assert isinstance(result.audit_signature, str)
            assert len(result.audit_signature) == 64
            assert all(c in '0123456789abcdef' for c in result.audit_signature)

    def test_audit_hash_chain_format(self):
        """测试审计哈希链格式"""
        model = HybridEnlightenLM(
            use_bayesian_l3=True,
            use_l3_controller=True
        )

        result = model.generate(
            prompt="你好",
            max_length=50,
            temperature=0.7
        )

        if result.audit_hash_chain:
            assert isinstance(result.audit_hash_chain, list)
            for hash_value in result.audit_hash_chain:
                assert isinstance(hash_value, str)
                assert len(hash_value) == 64

    def test_multiple_generations_chain_growth(self):
        """测试多次生成时哈希链增长"""
        model = HybridEnlightenLM(
            use_bayesian_l3=True,
            use_l3_controller=True
        )

        initial_length = len(model.audit_writer.get_entries())

        for i in range(3):
            model.generate(
                prompt=f"测试{i}",
                max_length=50,
                temperature=0.7
            )

        final_length = len(model.audit_writer.get_entries())
        assert final_length == initial_length + 3

    def test_get_audit_log_method(self):
        """测试获取审计日志方法"""
        model = HybridEnlightenLM(
            use_bayesian_l3=True,
            use_l3_controller=True
        )

        # 执行几次生成
        for i in range(2):
            model.generate(
                prompt=f"测试{i}",
                max_length=50,
                temperature=0.7
            )

        audit_log = model.get_audit_log(limit=10)

        assert isinstance(audit_log, dict)
        assert "total_entries" in audit_log
        assert "recent_entries" in audit_log
        assert "chain_verified" in audit_log
        assert audit_log["total_entries"] >= 2

    def test_chain_verification(self):
        """测试链完整性验证"""
        model = HybridEnlightenLM(
            use_bayesian_l3=True,
            use_l3_controller=True
        )

        # 执行几次生成
        for i in range(3):
            model.generate(
                prompt=f"测试{i}",
                max_length=50,
                temperature=0.7
            )

        # 验证链完整性
        is_valid = model.audit_writer.verify_chain()
        assert is_valid is True

    def test_audit_entry_structure(self):
        """测试审计条目结构"""
        model = HybridEnlightenLM(
            use_bayesian_l3=True,
            use_l3_controller=True
        )

        result = model.generate(
            prompt="你好",
            max_length=50,
            temperature=0.7
        )

        entries = model.audit_writer.get_entries()
        if entries:
            latest_entry = entries[-1]

            assert hasattr(latest_entry, "current_hash")
            assert hasattr(latest_entry, "previous_hash")
            assert hasattr(latest_entry, "header")

    def test_cutoff_generates_audit_entry(self):
        """测试截断时也生成审计条目"""
        model = HybridEnlightenLM(
            use_bayesian_l3=True,
            use_l3_controller=True
        )

        initial_entries = len(model.audit_writer.get_entries())

        # 尝试触发截断 - 使用超长输入
        long_prompt = "武器" * 100

        result = model.generate(
            prompt=long_prompt,
            max_length=50,
            temperature=0.7
        )

        final_entries = len(model.audit_writer.get_entries())
        # 即使被截断，也应该生成审计条目
        assert final_entries > initial_entries


class TestTEEAuditWriter:
    """测试TEE审计写入器"""

    def test_audit_writer_initialization(self):
        """测试审计写入器初始化"""
        writer = TEEAuditWriter(output_path="logs/test_audit")

        assert writer is not None
        assert hasattr(writer, "formatter")
        assert hasattr(writer, "entries")

    def test_write_entry(self):
        """测试写入审计条目"""
        writer = TEEAuditWriter(output_path="logs/test_audit")

        initial_count = len(writer.get_entries())

        audit_data = {
            "session_id": "test-session",
            "input_text": "测试输入",
            "output_text": "测试输出",
            "tokens": 10,
            "entropy_stats": {"mean": 0.5, "variance": 0.1}
        }

        entry = writer.write_entry(audit_data)

        assert entry is not None
        assert hasattr(entry, "current_hash")
        assert hasattr(entry, "previous_hash")

        final_count = len(writer.get_entries())
        assert final_count == initial_count + 1

    def test_verify_chain(self):
        """测试链验证"""
        writer = TEEAuditWriter(output_path="logs/test_audit")

        # 写入多个条目
        for i in range(3):
            writer.write_entry({
                "session_id": f"test-{i}",
                "input_text": f"输入{i}",
                "output_text": f"输出{i}"
            })

        # 验证链完整性
        is_valid = writer.verify_chain()
        assert is_valid is True

    def test_chain_link_integrity(self):
        """测试链链接完整性"""
        writer = TEEAuditWriter(output_path="logs/test_audit")

        entries = []
        for i in range(5):
            entry = writer.write_entry({
                "session_id": f"test-{i}",
                "input_text": f"输入{i}",
                "output_text": f"输出{i}"
            })
            entries.append(entry)

        # 验证每个条目的previous_hash都指向前一个条目
        for i in range(1, len(entries)):
            assert entries[i].previous_hash == entries[i-1].current_hash


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
