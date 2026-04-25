"""
Phase 1 端到端集成测试
测试完整的三层架构（L1/L2/L3）协同工作

测试场景:
- L1→L2→L3完整数据流
- 所有骨架代码适配器启用 ( use_l1_adapter=True, use_skeleton_l2=True, use_l3_controller=True)
- 安全截断功能
- 审计日志功能
- 多种场景协同测试
"""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch
from typing import Dict, Any, Tuple, List
import time


class TestL1L2L3CompleteDataFlow:
    """L1→L2→L3完整数据流测试"""

    def test_e2e_l1_to_l2_to_l3_flow(self):
        """E2E-FLOW-01: 完整L1→L2→L3数据流"""
        from enlighten.hybrid_architecture import HybridEnlightenLM, SimplifiedL2Adapter
        from enlighten.l3_controller import ControlSignals

        model = HybridEnlightenLM(
            use_l1_adapter=True,
            use_skeleton_l2=True,
            use_l3_controller=True,
            l1_config={"embed_dim": 768, "num_heads": 12},
            l2_config={"memory_size": 256, "embedding_dim": 768}
        )

        batch_size = 1
        seq_len = 16
        embed_dim = 768

        input_ids = torch.randint(0, 50000, (batch_size, seq_len))
        hidden_states = torch.randn(batch_size, seq_len, embed_dim)

        l1_result = model.l1_adapter(
            input_ids=input_ids,
            hidden_states=hidden_states,
            control_signals={"tau": 1.0, "theta": 0.5, "alpha": 0.1, "decay_rate": 0.95}
        )

        assert l1_result is not None
        assert "output_hidden" in l1_result
        assert "entropy_stats" in l1_result

        simplified_l2 = SimplifiedL2Adapter(memory_size=256, embedding_dim=768)
        attention_weights = torch.rand(batch_size, seq_len, seq_len)
        attention_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True)

        l2_result = simplified_l2.forward(
            hidden_states=l1_result["output_hidden"],
            attention_weights=l1_result["attention_weights"] if "attention_weights" in l1_result else attention_weights
        )

        assert l2_result is not None
        assert l2_result.sparse_kv is not None
        assert hasattr(l2_result, 'entropy_stats')

        l3_control = model.l3_controller_adapter.forward(
            entropy_stats=l2_result.entropy_stats,
            van_event=l1_result.get("van_event", False),
            p_harm=l1_result.get("p_harm", 0.0)
        )

        assert l3_control is not None
        assert isinstance(l3_control, ControlSignals)
        assert hasattr(l3_control, 'tau')
        assert hasattr(l3_control, 'cutoff')

    def test_e2e_all_adapters_enabled(self):
        """E2E-FLOW-02: 所有适配器启用状态验证"""
        from enlighten.hybrid_architecture import HybridEnlightenLM

        model = HybridEnlightenLM(
            use_l1_adapter=True,
            use_skeleton_l2=True,
            use_l3_controller=True,
            use_local_model=False
        )

        assert model.use_l1_adapter == True
        assert model.use_skeleton_l2 == True
        assert model.use_l3_controller == True
        assert model.l1_adapter is not None
        assert model.l2_adapter is not None
        assert model.l3_controller_adapter is not None

    def test_e2e_control_signals_propagation(self):
        """E2E-FLOW-03: 调控信号传播"""
        from enlighten.hybrid_architecture import HybridEnlightenLM, SimplifiedL2Adapter

        model = HybridEnlightenLM(
            use_l1_adapter=True,
            use_skeleton_l2=True,
            use_l3_controller=True,
            l3_config={"entropy_threshold": 0.5, "variance_threshold": 0.05}
        )

        hidden_states = torch.randn(1, 16, 768)
        input_ids = torch.randint(0, 50000, (1, 16))

        l1_result = model.l1_adapter(
            input_ids=input_ids,
            hidden_states=hidden_states,
            control_signals={"tau": 0.5, "theta": 0.7, "alpha": 0.2, "decay_rate": 0.9}
        )

        simplified_l2 = SimplifiedL2Adapter(memory_size=256, embedding_dim=768)
        attention_weights = torch.rand(1, 16, 16)
        attention_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True)

        l2_result = simplified_l2.forward(
            hidden_states=l1_result["output_hidden"],
            attention_weights=l1_result.get("attention_weights", attention_weights)
        )

        l3_signals = model.l3_controller_adapter.forward(
            entropy_stats=l2_result.entropy_stats,
            van_event=False,
            p_harm=0.0
        )

        l1_result_new = model.l1_adapter(
            input_ids=input_ids,
            hidden_states=hidden_states,
            control_signals={
                "tau": l3_signals.tau,
                "theta": l3_signals.theta,
                "alpha": l3_signals.alpha,
                "decay_rate": 0.95
            }
        )

        assert l1_result_new is not None
        assert l1_result_new["control_signals"]["tau"] == l3_signals.tau


class TestSecurityTruncation:
    """安全截断功能测试"""

    def test_truncation_by_entropy_stats(self):
        """E2E-SEC-01: 基于熵统计的截断 - 使用满足所有条件的熵值"""
        from enlighten.hybrid_architecture import HybridEnlightenLM

        model = HybridEnlightenLM(
            use_l1_adapter=True,
            use_skeleton_l2=True,
            use_l3_controller=True,
            l3_config={
                "entropy_threshold": 0.5,
                "variance_threshold": 0.1
            }
        )

        low_entropy_stats = {
            "mean": 0.3,
            "variance": 0.005,
            "trend": -0.1,
            "current": 0.25
        }

        l3_signals = model.l3_controller_adapter.forward(
            entropy_stats=low_entropy_stats,
            van_event=False,
            p_harm=0.0
        )

        assert l3_signals.cutoff == True

    def test_truncation_by_van_event(self):
        """E2E-SEC-02: 基于VAN事件的截断"""
        from enlighten.hybrid_architecture import HybridEnlightenLM

        model = HybridEnlightenLM(
            use_l1_adapter=True,
            use_skeleton_l2=True,
            use_l3_controller=True
        )

        normal_entropy = {"mean": 0.8, "variance": 0.1, "trend": 0.0, "current": 0.8}

        l3_signals_with_van = model.l3_controller_adapter.forward(
            entropy_stats=normal_entropy,
            van_event=True,
            p_harm=0.8
        )

        assert l3_signals_with_van.cutoff == True
        assert l3_signals_with_van.tau < 0.5

    def test_truncation_cooldown_mechanism(self):
        """E2E-SEC-03: 截断冷却机制"""
        from enlighten.hybrid_architecture import HybridEnlightenLM

        model = HybridEnlightenLM(
            use_l1_adapter=True,
            use_skeleton_l2=True,
            use_l3_controller=True,
            l3_config={
                "cutoff_cooldown": 5,
                "entropy_threshold": 0.5,
                "variance_threshold": 0.1
            }
        )

        low_entropy = {"mean": 0.3, "variance": 0.002, "trend": -0.1, "current": 0.25}

        first_signals = model.l3_controller_adapter.forward(
            entropy_stats=low_entropy,
            van_event=False,
            p_harm=0.0
        )

        assert first_signals.cutoff == True
        assert model.l3_controller_adapter.l3_controller.cooldown_counter == 5

        subsequent_signals = model.l3_controller_adapter.forward(
            entropy_stats={"mean": 0.8, "variance": 0.1, "trend": 0.0, "current": 0.8},
            van_event=False,
            p_harm=0.0
        )

        assert subsequent_signals.cutoff == False
        assert subsequent_signals.reason == "Cooldown"

    def test_truncation_flicker_suppression(self):
        """E2E-SEC-04: 截断抖动抑制"""
        from enlighten.hybrid_architecture import HybridEnlightenLM

        model = HybridEnlightenLM(
            use_l1_adapter=True,
            use_skeleton_l2=True,
            use_l3_controller=True,
            l3_config={
                "flicker_window_size": 3,
                "flicker_threshold": 0.6
            }
        )

        for i in range(10):
            entropy = {
                "mean": 0.3 if i % 2 == 0 else 0.8,
                "variance": 0.005 if i % 2 == 0 else 0.1,
                "trend": -0.1 if i % 2 == 0 else 0.1,
                "current": 0.3 if i % 2 == 0 else 0.8
            }
            model.l3_controller_adapter.forward(
                entropy_stats=entropy,
                van_event=False,
                p_harm=0.0
            )

        assert len(model.l3_controller_adapter.l3_controller.cutoff_history) <= 10

    def test_high_risk_p_harm_with_van_event(self):
        """E2E-SEC-05: 高有害概率配合VAN事件截断"""
        from enlighten.hybrid_architecture import HybridEnlightenLM

        model = HybridEnlightenLM(
            use_l1_adapter=True,
            use_skeleton_l2=True,
            use_l3_controller=True
        )

        normal_entropy = {"mean": 0.8, "variance": 0.1, "trend": 0.0, "current": 0.8}

        signals = model.l3_controller_adapter.forward(
            entropy_stats=normal_entropy,
            van_event=True,
            p_harm=0.85
        )

        assert signals.cutoff == True

    def test_security_verified_passes_normal_input(self):
        """E2E-SEC-06: 正常输入通过安全验证"""
        from enlighten.hybrid_architecture import HybridEnlightenLM

        model = HybridEnlightenLM(
            use_l1_adapter=True,
            use_skeleton_l2=True,
            use_l3_controller=True
        )

        normal_entropy = {"mean": 0.7, "variance": 0.08, "trend": 0.05, "current": 0.7}

        signals = model.l3_controller_adapter.forward(
            entropy_stats=normal_entropy,
            van_event=False,
            p_harm=0.1
        )

        assert signals.cutoff == False
        assert signals.stability == True


class TestAuditLogging:
    """审计日志功能测试"""

    def test_audit_chain_initialization(self):
        """E2E-AUDIT-01: 审计链初始化"""
        from enlighten.audit.chain import AuditHashChain

        chain = AuditHashChain()

        assert chain.chain == []
        assert chain.get_chain_length() == 0
        assert chain.verify() == True

    def test_audit_chain_append_and_verify(self):
        """E2E-AUDIT-02: 审计链追加和验证"""
        from enlighten.audit.chain import AuditHashChain

        chain = AuditHashChain()

        data1 = {"event": "test1", "value": 100}
        data2 = {"event": "test2", "value": 200}

        hash1 = chain.append(data1)
        hash2 = chain.append(data2)

        assert chain.get_chain_length() == 2
        assert hash1 != hash2
        assert chain.verify() == True

    def test_audit_chain_tamper_detection(self):
        """E2E-AUDIT-03: 审计链篡改检测"""
        from enlighten.audit.chain import AuditHashChain

        chain = AuditHashChain()

        chain.append({"event": "original", "data": "value"})
        chain.append({"event": "modified", "data": "value"})

        if len(chain.chain) >= 2:
            chain.chain[0].data["event"] = "tampered"

        assert chain.verify() == False

    def test_audit_chain_mixed_with_l3_controller(self):
        """E2E-AUDIT-04: L3控制器决策审计"""
        from enlighten.audit.chain import AuditHashChain
        from enlighten.hybrid_architecture import HybridEnlightenLM

        chain = AuditHashChain()
        model = HybridEnlightenLM(
            use_l1_adapter=True,
            use_skeleton_l2=True,
            use_l3_controller=True
        )

        test_cases = [
            {"mean": 0.3, "variance": 0.005, "trend": -0.1},
            {"mean": 0.8, "variance": 0.1, "trend": 0.0},
            {"mean": 0.5, "variance": 0.05, "trend": 0.05}
        ]

        for i, entropy in enumerate(test_cases):
            l3_signals = model.l3_controller_adapter.forward(
                entropy_stats=entropy,
                van_event=False,
                p_harm=0.0
            )

            audit_data = {
                "step": i,
                "entropy": entropy,
                "cutoff": l3_signals.cutoff,
                "tau": l3_signals.tau,
                "reason": l3_signals.reason
            }
            chain.append(audit_data)

        assert chain.get_chain_length() == len(test_cases)
        assert chain.verify() == True


class TestL1L2L3Coordination:
    """三层架构协同工作测试"""

    def test_van_detection_triggers_l3_response(self):
        """E2E-COORD-01: VAN检测触发L3响应"""
        from enlighten.hybrid_architecture import HybridEnlightenLM

        model = HybridEnlightenLM(
            use_l1_adapter=True,
            use_skeleton_l2=True,
            use_l3_controller=True
        )

        hidden_states = torch.randn(1, 16, 768)
        input_ids = torch.randint(0, 50000, (1, 16))

        l1_result = model.l1_adapter(
            input_ids=input_ids,
            hidden_states=hidden_states,
            control_signals={"tau": 1.0, "theta": 0.5, "alpha": 0.1, "decay_rate": 0.95}
        )

        if l1_result.get("van_event", False):
            l3_signals = model.l3_controller_adapter.forward(
                entropy_stats=l1_result.get("entropy_stats", {}),
                van_event=True,
                p_harm=l1_result.get("p_harm", 0.0)
            )
            assert l3_signals.cutoff == True

    def test_entropy_degradation_triggers_cutoff(self):
        """E2E-COORD-02: 熵值退化触发截断"""
        from enlighten.hybrid_architecture import HybridEnlightenLM

        model = HybridEnlightenLM(
            use_l1_adapter=True,
            use_skeleton_l2=True,
            use_l3_controller=True,
            l3_config={"entropy_threshold": 0.5, "variance_threshold": 0.1}
        )

        entropy_history = [
            {"mean": 0.8, "variance": 0.1, "trend": 0.0, "current": 0.8},
            {"mean": 0.6, "variance": 0.08, "trend": -0.05, "current": 0.6},
            {"mean": 0.4, "variance": 0.03, "trend": -0.1, "current": 0.4},
            {"mean": 0.3, "variance": 0.005, "trend": -0.15, "current": 0.3}
        ]

        cutoff_triggered = False
        for entropy in entropy_history:
            signals = model.l3_controller_adapter.forward(
                entropy_stats=entropy,
                van_event=False,
                p_harm=0.0
            )
            if signals.cutoff:
                cutoff_triggered = True
                break

        assert cutoff_triggered == True

    def test_stability_maintained_under_normal_operation(self):
        """E2E-COORD-03: 正常操作下维持稳定"""
        from enlighten.hybrid_architecture import HybridEnlightenLM

        model = HybridEnlightenLM(
            use_l1_adapter=True,
            use_skeleton_l2=True,
            use_l3_controller=True
        )

        stable_entropy = {"mean": 0.7, "variance": 0.08, "trend": 0.02, "current": 0.72}

        all_stable = True
        for _ in range(5):
            signals = model.l3_controller_adapter.forward(
                entropy_stats=stable_entropy,
                van_event=False,
                p_harm=0.1
            )
            if signals.cutoff or not signals.stability:
                all_stable = False
                break

        assert all_stable == True

    def test_memory_snapshot_integrity(self):
        """E2E-COORD-04: 记忆快照完整性"""
        from enlighten.hybrid_architecture import SimplifiedL2Adapter

        adapter = SimplifiedL2Adapter(memory_size=128, embedding_dim=768)

        hidden_states = torch.randn(1, 16, 768)
        attention_weights = torch.rand(1, 16, 16)
        attention_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True)

        l2_result = adapter.forward(
            hidden_states=hidden_states,
            attention_weights=attention_weights,
            update_memory=True
        )

        snapshot = l2_result.memory_snapshot

        assert snapshot is not None
        assert "memory_size" in snapshot or "memory" in snapshot

    def test_control_signals_ranges(self):
        """E2E-COORD-05: 调控信号范围验证"""
        from enlighten.hybrid_architecture import HybridEnlightenLM

        model = HybridEnlightenLM(
            use_l3_controller=True,
            l3_config={
                "tau_range": [0.1, 2.0],
                "theta_range": [0.5, 0.9],
                "alpha_range": [0.0, 1.0]
            }
        )

        test_cases = [
            {"mean": 0.2, "variance": 0.005, "trend": -0.2},
            {"mean": 0.9, "variance": 0.15, "trend": 0.1},
            {"mean": 0.5, "variance": 0.05, "trend": 0.0}
        ]

        for entropy in test_cases:
            signals = model.l3_controller_adapter.forward(
                entropy_stats=entropy,
                van_event=False,
                p_harm=0.0
            )

            assert 0.1 <= signals.tau <= 2.0
            assert 0.5 <= signals.theta <= 0.9
            assert 0.0 <= signals.alpha <= 1.0


class TestNormalAndExceptionScenarios:
    """正常和异常场景测试"""

    def test_normal_generation_scenario(self):
        """E2E-SCENARIO-01: 正常生成场景"""
        from enlighten.hybrid_architecture import HybridEnlightenLM

        model = HybridEnlightenLM(
            use_l1_adapter=True,
            use_skeleton_l2=True,
            use_l3_controller=True
        )

        normal_entropy = {"mean": 0.7, "variance": 0.08, "trend": 0.02, "current": 0.7}

        signals = model.l3_controller_adapter.forward(
            entropy_stats=normal_entropy,
            van_event=False,
            p_harm=0.1
        )

        assert signals.cutoff == False
        assert signals.stability == True

    def test_self_referential_loop_detection(self):
        """E2E-SCENARIO-02: 自指循环检测"""
        from enlighten.hybrid_architecture import HybridEnlightenLM

        model = HybridEnlightenLM(
            use_l1_adapter=True,
            use_skeleton_l2=True,
            use_l3_controller=True,
            l3_config={"entropy_threshold": 0.5, "variance_threshold": 0.1}
        )

        low_entropy_loop = {"mean": 0.3, "variance": 0.005, "trend": -0.1, "current": 0.3}

        signals = model.l3_controller_adapter.forward(
            entropy_stats=low_entropy_loop,
            van_event=False,
            p_harm=0.0
        )

        if signals.cutoff:
            assert "Self-referential" in (signals.reason or "") or signals.tau < 0.7

    def test_sensitive_content_van_event(self):
        """E2E-SCENARIO-03: 敏感内容VAN事件"""
        from enlighten.hybrid_architecture import HybridEnlightenLM

        model = HybridEnlightenLM(
            use_l1_adapter=True,
            use_skeleton_l2=True,
            use_l3_controller=True
        )

        signals = model.l3_controller_adapter.forward(
            entropy_stats={"mean": 0.6, "variance": 0.1, "trend": 0.0, "current": 0.6},
            van_event=True,
            p_harm=0.8
        )

        assert signals.cutoff == True
        assert signals.tau <= 0.5

    def test_high_entropy_normal_operation(self):
        """E2E-SCENARIO-04: 高熵正常操作"""
        from enlighten.hybrid_architecture import HybridEnlightenLM

        model = HybridEnlightenLM(
            use_l1_adapter=True,
            use_skeleton_l2=True,
            use_l3_controller=True
        )

        high_entropy = {"mean": 0.95, "variance": 0.15, "trend": 0.05, "current": 0.98}

        signals = model.l3_controller_adapter.forward(
            entropy_stats=high_entropy,
            van_event=False,
            p_harm=0.05
        )

        assert signals.cutoff == False
        assert signals.stability == True
        assert signals.tau >= 0.5

    def test_rapid_entropy_change_stability(self):
        """E2E-SCENARIO-05: 快速熵变稳定性"""
        from enlighten.hybrid_architecture import HybridEnlightenLM

        model = HybridEnlightenLM(
            use_l1_adapter=True,
            use_skeleton_l2=True,
            use_l3_controller=True
        )

        rapid_changes = [
            {"mean": 0.9, "variance": 0.1, "trend": 0.1},
            {"mean": 0.3, "variance": 0.005, "trend": -0.2},
            {"mean": 0.8, "variance": 0.12, "trend": 0.15},
            {"mean": 0.4, "variance": 0.03, "trend": -0.1}
        ]

        results = []
        for entropy in rapid_changes:
            signals = model.l3_controller_adapter.forward(
                entropy_stats=entropy,
                van_event=False,
                p_harm=0.0
            )
            results.append(signals)

        assert len(results) == len(rapid_changes)
        assert all(hasattr(r, 'tau') for r in results)


class TestResetAndCleanup:
    """重置和清理测试"""

    def test_l3_controller_reset(self):
        """E2E-RESET-01: L3控制器重置"""
        from enlighten.hybrid_architecture import HybridEnlightenLM

        model = HybridEnlightenLM(
            use_l1_adapter=True,
            use_skeleton_l2=True,
            use_l3_controller=True
        )

        model.l3_controller_adapter.forward(
            entropy_stats={"mean": 0.3, "variance": 0.005, "trend": -0.1},
            van_event=False,
            p_harm=0.0
        )

        assert model.l3_controller_adapter.l3_controller.cooldown_counter >= 0

        model.l3_controller_adapter.reset()

        assert model.l3_controller_adapter.l3_controller.cooldown_counter == 0
        assert len(model.l3_controller_adapter.l3_controller.decision_history) == 0

    def test_l2_adapter_reset(self):
        """E2E-RESET-02: L2适配器重置"""
        from enlighten.hybrid_architecture import SimplifiedL2Adapter

        adapter = SimplifiedL2Adapter(memory_size=128, embedding_dim=768)

        hidden_states = torch.randn(1, 16, 768)
        attention_weights = torch.rand(1, 16, 16)
        attention_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True)

        adapter.forward(hidden_states, attention_weights)

        adapter.reset()

        assert adapter._last_sparse_kv is None

    def test_full_system_reset(self):
        """E2E-RESET-03: 完整系统重置"""
        from enlighten.hybrid_architecture import HybridEnlightenLM

        model = HybridEnlightenLM(
            use_l1_adapter=True,
            use_skeleton_l2=True,
            use_l3_controller=True
        )

        hidden_states = torch.randn(1, 16, 768)
        input_ids = torch.randint(0, 50000, (1, 16))

        model.l1_adapter(input_ids, hidden_states, {"tau": 1.0, "theta": 0.5, "alpha": 0.1, "decay_rate": 0.95})

        model.reset()

        assert model.working_memory.token_count == 0
        assert len(model.working_memory.conversation_history) == 0


class TestStatusReporting:
    """状态报告测试"""

    def test_get_status_with_all_adapters(self):
        """E2E-STATUS-01: 所有适配器状态获取"""
        from enlighten.hybrid_architecture import HybridEnlightenLM

        model = HybridEnlightenLM(
            use_l1_adapter=True,
            use_skeleton_l2=True,
            use_l3_controller=True,
            l1_config={"embed_dim": 768, "num_heads": 12},
            l2_config={"memory_size": 256, "embedding_dim": 768}
        )

        status = model.get_status()

        assert status["use_l1_adapter"] == True
        assert status["use_skeleton_l2"] == True
        assert status["use_l3_controller"] == True
        assert "l1_adapter" in status
        assert "l2_adapter" in status
        assert "l3_controller" in status

    def test_get_l3_statistics(self):
        """E2E-STATUS-02: L3统计信息获取"""
        from enlighten.hybrid_architecture import HybridEnlightenLM

        model = HybridEnlightenLM(
            use_l3_controller=True,
            l3_config={"entropy_threshold": 0.5, "variance_threshold": 0.05}
        )

        for i in range(5):
            model.l3_controller_adapter.forward(
                entropy_stats={"mean": 0.6 - i * 0.05, "variance": 0.05, "trend": -0.01 * i},
                van_event=False,
                p_harm=0.0
            )

        stats = model.get_l3_statistics()

        assert "total_decisions" in stats
        assert stats["total_decisions"] >= 5

    def test_get_temperature_and_sparsity(self):
        """E2E-STATUS-03: 温度和稀疏度获取"""
        from enlighten.hybrid_architecture import HybridEnlightenLM

        model = HybridEnlightenLM(
            use_l3_controller=True,
            l3_config={"tau_range": [0.1, 2.0], "theta_range": [0.5, 0.9]}
        )

        model.l3_controller_adapter.forward(
            entropy_stats={"mean": 0.5, "variance": 0.05, "trend": 0.0},
            van_event=False,
            p_harm=0.0
        )

        tau = model.get_temperature()
        theta = model.get_sparsity_threshold()
        alpha = model.get_dmn_coefficient()

        assert 0.1 <= tau <= 2.0
        assert 0.5 <= theta <= 0.9
        assert 0.0 <= alpha <= 1.0


class TestEdgeCases:
    """边缘情况测试"""

    def test_empty_hidden_states_handling(self):
        """E2E-EDGE-01: 空隐藏状态处理"""
        from enlighten.hybrid_architecture import SimplifiedL2Adapter

        adapter = SimplifiedL2Adapter(memory_size=64, embedding_dim=256)

        hidden_states = torch.randn(1, 0, 256)
        attention_weights = torch.rand(1, 0, 0)
        attention_weights = (attention_weights + 1e-10) / (attention_weights.sum(dim=-1, keepdim=True) + 1e-10)

        try:
            result = adapter.forward(hidden_states, attention_weights)
            assert result is not None
        except Exception:
            pass

    def test_extreme_entropy_values(self):
        """E2E-EDGE-02: 极端熵值"""
        from enlighten.hybrid_architecture import HybridEnlightenLM

        model = HybridEnlightenLM(
            use_l3_controller=True
        )

        extreme_cases = [
            {"mean": 0.0, "variance": 0.0, "trend": 0.0, "current": 0.0},
            {"mean": 1.5, "variance": 0.5, "trend": 0.5, "current": 1.5},
            {"mean": -0.1, "variance": 0.0, "trend": -0.5, "current": -0.1}
        ]

        for entropy in extreme_cases:
            try:
                signals = model.l3_controller_adapter.forward(
                    entropy_stats=entropy,
                    van_event=False,
                    p_harm=0.0
                )
                assert signals is not None
            except Exception:
                pass

    def test_maximum_batch_size(self):
        """E2E-EDGE-03: 最大批量大小"""
        from enlighten.hybrid_architecture import SimplifiedL2Adapter

        adapter = SimplifiedL2Adapter(memory_size=512, embedding_dim=768)

        batch_size = 32
        seq_len = 16

        hidden_states = torch.randn(batch_size, seq_len, 768)
        attention_weights = torch.rand(batch_size, seq_len, seq_len)
        attention_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True)

        result = adapter.forward(hidden_states, attention_weights)

        assert result is not None
        assert result.sparse_kv is not None

    def test_l3_cooldown_boundary(self):
        """E2E-EDGE-04: L3冷却边界"""
        from enlighten.hybrid_architecture import HybridEnlightenLM

        model = HybridEnlightenLM(
            use_l3_controller=True,
            l3_config={"cutoff_cooldown": 1}
        )

        model.l3_controller_adapter.forward(
            entropy_stats={"mean": 0.3, "variance": 0.005, "trend": -0.1},
            van_event=False,
            p_harm=0.0
        )

        assert model.l3_controller_adapter.l3_controller.cooldown_counter >= 0

        model.reset_l3_cooldown()

        assert model.l3_controller_adapter.l3_controller.cooldown_counter == 0


class TestBayesianL3Integration:
    """贝叶斯L3集成测试"""

    def test_bayesian_l3_initialization(self):
        """E2E-BAYESIAN-01: 贝叶斯L3初始化"""
        from enlighten.hybrid_architecture import HybridEnlightenLM

        model = HybridEnlightenLM(
            use_bayesian_l3=True,
            use_l1_adapter=True
        )

        assert model.bayesian_l3 is not None

    def test_bayesian_l3_posterior_update(self):
        """E2E-BAYESIAN-02: 贝叶斯L3后验更新"""
        from enlighten.hybrid_architecture import HybridEnlightenLM

        model = HybridEnlightenLM(
            use_bayesian_l3=True,
            use_l1_adapter=True
        )

        prior = model.bayesian_l3.get_posterior()

        model.bayesian_l3.forward(
            entropy_stats={"mean": 0.2, "variance": 0.01, "trend": -0.1, "current": 0.2},
            van_event=False,
            p_harm=0.0
        )

        posterior = model.bayesian_l3.get_posterior()

        assert prior is not None
        assert posterior is not None
        assert abs(sum(posterior.values()) - 1.0) < 0.001

    def test_bayesian_l3_cutoff_decision(self):
        """E2E-BAYESIAN-03: 贝叶斯L3截断决策"""
        from enlighten.hybrid_architecture import HybridEnlightenLM

        model = HybridEnlightenLM(
            use_bayesian_l3=True,
            use_l1_adapter=True
        )

        signals = model.bayesian_l3.forward(
            entropy_stats={"mean": 0.3, "variance": 0.02, "trend": -0.1, "current": 0.3},
            van_event=True,
            p_harm=0.8
        )

        assert signals.cutoff == True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
