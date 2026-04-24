"""
L3 Trace Collector 测试用例

测试 TraceCollector 的核心功能：
1. 噪声注入功能
2. 记录格式
3. 数据摘要统计
"""

import os
import pytest
import numpy as np
import random
from pathlib import Path
from unittest.mock import MagicMock, patch

from tests.l3_trace_collector import TraceCollector, TraceRecord


class TestNoiseInjection:
    """噪声注入功能测试"""

    def test_inject_noise_basic(self):
        """测试基本噪声注入"""
        collector = TraceCollector()

        original = "Hello World"
        noisy = collector._inject_noise(original, noise_rate=0.2)

        assert len(noisy) == len(original)
        assert noisy != original

    def test_inject_noise_empty(self):
        """测试空字符串"""
        collector = TraceCollector()
        noisy = collector._inject_noise("", noise_rate=0.2)
        assert noisy == ""

    def test_inject_noise_rate_zero(self):
        """测试零噪声率"""
        collector = TraceCollector()
        original = "Hello World"
        noisy = collector._inject_noise(original, noise_rate=0.0)
        assert noisy == original

    def test_inject_noise_rate_one(self):
        """测试全噪声率"""
        collector = TraceCollector()
        original = "Hello World"
        noisy = collector._inject_noise(original, noise_rate=1.0)
        assert len(noisy) == len(original)


class TestTraceRecord:
    """TraceRecord 数据类测试"""

    def test_trace_record_creation(self):
        """测试记录创建"""
        record = TraceRecord(
            timestamp="2024-01-01T00:00:00",
            condition="normal",
            session_id="abc123",
            turn=1,
            mu_H=0.5,
            sigma_H2=0.1,
            k_H=0.01,
            p_harm_raw=0.2,
            input_text="Hello",
            output_text="Hi there"
        )

        assert record.condition == "normal"
        assert record.mu_H == 0.5
        assert record.sigma_H2 == 0.1
        assert record.k_H == 0.01
        assert record.p_harm_raw == 0.2


class TestTraceCollectorDataSummary:
    """数据摘要统计测试"""

    def test_data_summary_empty(self):
        """测试空数据的摘要"""
        collector = TraceCollector()
        summary = collector.get_data_summary()
        assert summary == {}

    def test_data_summary_single_condition(self):
        """测试单条件数据的摘要"""
        collector = TraceCollector()

        for i in range(5):
            collector.records.append(TraceRecord(
                timestamp=f"2024-01-01T00:00:{i:02d}",
                condition="normal",
                session_id=f"session_{i}",
                turn=i,
                mu_H=float(i) * 0.1,
                sigma_H2=0.1,
                k_H=0.01,
                p_harm_raw=0.1,
                input_text="test",
                output_text="result"
            ))

        summary = collector.get_data_summary()

        assert "normal" in summary
        assert summary["normal"]["count"] == 5
        assert summary["normal"]["mu_H"]["mean"] == pytest.approx(0.2, rel=0.01)

    def test_data_summary_multiple_conditions(self):
        """测试多条件数据的摘要"""
        collector = TraceCollector()

        for condition in ["normal", "noise_injection", "bias_injection"]:
            for i in range(3):
                collector.records.append(TraceRecord(
                    timestamp=f"2024-01-01T00:00:{i:02d}",
                    condition=condition,
                    session_id=f"session_{condition}_{i}",
                    turn=i,
                    mu_H=0.5 if condition == "normal" else 0.3,
                    sigma_H2=0.1,
                    k_H=0.01,
                    p_harm_raw=0.1,
                    input_text="test",
                    output_text="result"
                ))

        summary = collector.get_data_summary()

        assert len(summary) == 3
        assert "normal" in summary
        assert "noise_injection" in summary
        assert "bias_injection" in summary


class TestTraceCollectorCSV:
    """CSV保存功能测试"""

    def test_save_csv_empty(self, tmp_path):
        """测试空数据保存"""
        collector = TraceCollector(output_dir=str(tmp_path))
        filepath = collector.save_csv("test_empty.csv")

        assert Path(filepath).exists()

        with open(filepath, 'r') as f:
            lines = f.readlines()
            assert len(lines) == 1
            assert "timestamp" in lines[0]

    def test_save_csv_with_records(self, tmp_path):
        """测试有记录的数据保存"""
        collector = TraceCollector(output_dir=str(tmp_path))

        collector.records.append(TraceRecord(
            timestamp="2024-01-01T00:00:00",
            condition="normal",
            session_id="test123",
            turn=1,
            mu_H=0.5,
            sigma_H2=0.1,
            k_H=0.01,
            p_harm_raw=0.2,
            input_text="Hello",
            output_text="Hi"
        ))

        filepath = collector.save_csv("test_records.csv")

        assert Path(filepath).exists()

        with open(filepath, 'r') as f:
            lines = f.readlines()
            assert len(lines) == 2
            assert "normal" in lines[1]
            assert "test123" in lines[1]


class TestHybridEnlightenLMIntegration:
    """与 HybridEnlightenLM 集成测试"""

    def test_get_l3_trace_signals(self):
        """测试 L3 trace signals 获取"""
        from enlighten.hybrid_architecture import HybridEnlightenLM

        model = HybridEnlightenLM(use_local_model=False)

        model.working_memory.add_turn("user", "Hello")
        model.working_memory.add_turn("assistant", "Hi there")

        signals = model.get_l3_trace_signals()

        assert "mu_H" in signals
        assert "sigma_H2" in signals
        assert "k_H" in signals
        assert "p_harm_raw" in signals

        assert isinstance(signals["mu_H"], float)
        assert isinstance(signals["sigma_H2"], float)
        assert isinstance(signals["k_H"], float)
        assert isinstance(signals["p_harm_raw"], float)

    @pytest.mark.skipif(
        not os.environ.get("DEEPSEEK_API_KEY"),
        reason="DEEPSEEK_API_KEY not set"
    )
    def test_generate_with_trace_callback(self):
        """测试 generate 方法的 trace_callback"""
        from enlighten.hybrid_architecture import HybridEnlightenLM
        from enlighten.api.deepseek_client import DeepSeekAPIClient, DeepSeekConfig

        config = DeepSeekConfig(api_key=os.environ["DEEPSEEK_API_KEY"], model="deepseek-chat")
        client = DeepSeekAPIClient(config)

        model = HybridEnlightenLM(use_local_model=False, api_client=client)

        callback_data = {}

        def trace_callback(mu_H, sigma_H2, k_H, p_harm_raw):
            callback_data["mu_H"] = mu_H
            callback_data["sigma_H2"] = sigma_H2
            callback_data["k_H"] = k_H
            callback_data["p_harm_raw"] = p_harm_raw

        model.working_memory.add_turn("user", "Hello")
        model.working_memory.add_turn("assistant", "Hi there")

        result = model.generate(
            prompt="Test trace",
            max_length=100,
            enable_trace=True,
            trace_callback=trace_callback
        )

        assert "mu_H" in callback_data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
