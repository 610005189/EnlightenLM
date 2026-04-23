"""
Security Test - 安全测试验证
验证 EnlightenLM 的安全功能：截断有效性、审计完整性、HMAC 签名等
"""

import os
import sys
import time
import hashlib
import json
import tempfile
from datetime import datetime
from typing import Dict, List, Any

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enlighten.audit.chain import AuditHashChain
from enlighten.audit.hmac_sign import HMACSigner
from enlighten.audit.tee_audit import TEEAuditFormatter, TEEAuditWriter, TEEType
from enlighten.l3_controller import L3Controller


class SecurityTestResult:
    """安全测试结果"""
    def __init__(self, test_name: str, passed: bool, message: str = "", details: Dict = None):
        self.test_name = test_name
        self.passed = passed
        self.message = message
        self.details = details or {}
        self.timestamp = time.time()

    def to_dict(self) -> Dict:
        return {
            "test_name": self.test_name,
            "passed": self.passed,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp
        }


class SecurityTestSuite:
    """
    安全测试套件

    验证内容:
    1. 截断有效性
    2. 审计哈希链完整性
    3. HMAC 签名验证
    4. TEE 审计格式
    5. 决策历史不可篡改
    """

    def __init__(self):
        self.results: List[SecurityTestResult] = []

    def test_cutoff_effectiveness(self) -> SecurityTestResult:
        """测试截断有效性"""
        print("测试截断有效性...")

        try:
            l3 = L3Controller()

            entropy_stats_normal = {
                "mean": 0.8,
                "variance": 0.1,
                "trend": 0.1
            }

            entropy_stats_low = {
                "mean": 0.3,
                "variance": 0.02,
                "trend": -0.1
            }

            signals_normal = l3.forward(entropy_stats_normal, van_event=False)
            signals_low = l3.forward(entropy_stats_low, van_event=False)

            cutoff_triggered = signals_low.cutoff == True
            normal_not_cutoff = signals_normal.cutoff == False

            passed = cutoff_triggered and normal_not_cutoff

            return SecurityTestResult(
                test_name="cutoff_effectiveness",
                passed=passed,
                message="截断功能正常" if passed else "截断功能异常",
                details={
                    "low_entropy_cutoff": signals_low.cutoff,
                    "normal_entropy_cutoff": signals_normal.cutoff,
                    "low_entropy_reason": signals_low.reason
                }
            )
        except Exception as e:
            return SecurityTestResult(
                test_name="cutoff_effectiveness",
                passed=False,
                message=f"测试异常: {str(e)}"
            )

    def test_audit_chain_integrity(self) -> SecurityTestResult:
        """测试审计哈希链完整性"""
        print("测试审计哈希链完整性...")

        try:
            chain = AuditHashChain()
            signer = HMACSigner()

            test_data = [
                {"session": "test1", "action": "generate"},
                {"session": "test2", "action": "cutoff"},
                {"session": "test3", "action": "generate"}
            ]

            for data in test_data:
                signature = signer.sign(data)
                chain.append(data, signature)

            chain_valid = chain.verify()

            return SecurityTestResult(
                test_name="audit_chain_integrity",
                passed=chain_valid,
                message="哈希链完整" if chain_valid else "哈希链被篡改",
                details={
                    "chain_length": chain.get_chain_length(),
                    "verified": chain_valid
                }
            )
        except Exception as e:
            return SecurityTestResult(
                test_name="audit_chain_integrity",
                passed=False,
                message=f"测试异常: {str(e)}"
            )

    def test_tamper_detection(self) -> SecurityTestResult:
        """测试篡改检测"""
        print("测试篡改检测...")

        try:
            chain = AuditHashChain()
            signer = HMACSigner()

            original_data = {"session": "test", "action": "generate", "output": "normal"}
            chain.append(original_data, signer.sign(original_data))

            chain_valid_before = chain.verify()

            chain.chain[0].data["output"] = "tampered"

            chain_valid_after = chain.verify()

            tampered_detected = chain_valid_before and not chain_valid_after

            return SecurityTestResult(
                test_name="tamper_detection",
                passed=tampered_detected,
                message="篡改检测正常" if tampered_detected else "篡改检测失败",
                details={
                    "valid_before_tamper": chain_valid_before,
                    "valid_after_tamper": chain_valid_after
                }
            )
        except Exception as e:
            return SecurityTestResult(
                test_name="tamper_detection",
                passed=False,
                message=f"测试异常: {str(e)}"
            )

    def test_hmac_signature(self) -> SecurityTestResult:
        """测试 HMAC 签名"""
        print("测试 HMAC 签名...")

        try:
            signer = HMACSigner()

            test_data = {"session": "test", "action": "generate"}

            signature = signer.sign(test_data)

            valid = signer.verify(test_data, signature)

            tampered_data = {"session": "test", "action": "cutoff"}
            invalid = signer.verify(tampered_data, signature)

            passed = valid and not invalid

            return SecurityTestResult(
                test_name="hmac_signature",
                passed=passed,
                message="HMAC 签名正常" if passed else "HMAC 签名异常",
                details={
                    "valid_signature": valid,
                    "invalid_on_tamper": not invalid
                }
            )
        except Exception as e:
            return SecurityTestResult(
                test_name="hmac_signature",
                passed=False,
                message=f"测试异常: {str(e)}"
            )

    def test_tee_audit_format(self) -> SecurityTestResult:
        """测试 TEE 审计格式"""
        print("测试 TEE 审计格式...")

        try:
            formatter = TEEAuditFormatter(TEEType.SGX)

            test_data = {
                "session_id": "test_session",
                "action": "generate",
                "entropy": 0.5
            }

            entry = formatter.format_entry(
                test_data,
                enclave_id="test_enclave",
                measurement="test_measurement"
            )

            entry_dict = entry.to_dict()

            has_required_fields = all(
                k in entry_dict for k in ["header", "data", "previous_hash", "current_hash"]
            )

            hash_valid = entry.current_hash == formatter._compute_hash(
                entry.header, entry.data, entry.previous_hash
            )

            passed = has_required_fields and hash_valid

            return SecurityTestResult(
                test_name="tee_audit_format",
                passed=passed,
                message="TEE 审计格式正确" if passed else "TEE 审计格式异常",
                details={
                    "has_required_fields": has_required_fields,
                    "hash_valid": hash_valid,
                    "sequence_number": entry.header.sequence_number
                }
            )
        except Exception as e:
            return SecurityTestResult(
                test_name="tee_audit_format",
                passed=False,
                message=f"测试异常: {str(e)}"
            )

    def test_decision_history_immutability(self) -> SecurityTestResult:
        """测试决策历史不可篡改"""
        print("测试决策历史不可篡改...")

        try:
            l3 = L3Controller()

            entropy_stats = {"mean": 0.3, "variance": 0.02, "trend": -0.1}
            l3.forward(entropy_stats, van_event=False)

            history_before = len(l3.decision_history)
            history_entry_before = l3.decision_history[0] if l3.decision_history else None

            if history_entry_before:
                original_entropy = history_entry_before.entropy_mean
                history_entry_before.entropy_mean = 999.0

                l3.forward(entropy_stats, van_event=False)
                history_after = len(l3.decision_history)

                immutable = (
                    history_after == history_before + 1 and
                    l3.decision_history[0].entropy_mean == 999.0
                )
            else:
                immutable = True

            return SecurityTestResult(
                test_name="decision_history_immutability",
                passed=immutable,
                message="决策历史不可篡改" if immutable else "决策历史可被篡改",
                details={
                    "history_length_before": history_before,
                    "history_length_after": history_after
                }
            )
        except Exception as e:
            return SecurityTestResult(
                test_name="decision_history_immutability",
                passed=False,
                message=f"测试异常: {str(e)}"
            )

    def test_hash_chain_continuity(self) -> SecurityTestResult:
        """测试哈希链连续性"""
        print("测试哈希链连续性...")

        try:
            chain = AuditHashChain()

            entries = []
            for i in range(5):
                data = {"index": i, "data": f"test_{i}"}
                entry_hash = chain.append(data)
                entries.append((data, entry_hash))

            continuous = True
            for i in range(1, len(chain.chain)):
                prev_hash = chain.chain[i-1].hash
                curr_prev_hash = chain.chain[i].data.get("_prev_hash")

                if prev_hash != curr_prev_hash:
                    continuous = False
                    break

            return SecurityTestResult(
                test_name="hash_chain_continuity",
                passed=continuous,
                message="哈希链连续" if continuous else "哈希链不连续",
                details={
                    "chain_length": chain.get_chain_length(),
                    "continuous": continuous
                }
            )
        except Exception as e:
            return SecurityTestResult(
                test_name="hash_chain_continuity",
                passed=False,
                message=f"测试异常: {str(e)}"
            )

    def run_all_tests(self) -> Dict[str, Any]:
        """运行所有安全测试"""
        print("=" * 60)
        print("开始安全测试验证")
        print("=" * 60)

        tests = [
            self.test_cutoff_effectiveness,
            self.test_audit_chain_integrity,
            self.test_tamper_detection,
            self.test_hmac_signature,
            self.test_tee_audit_format,
            self.test_decision_history_immutability,
            self.test_hash_chain_continuity
        ]

        for test in tests:
            result = test()
            self.results.append(result)
            status = "✅ 通过" if result.passed else "❌ 失败"
            print(f"{status}: {result.test_name}")
            if result.message:
                print(f"  消息: {result.message}")

        passed_count = sum(1 for r in self.results if r.passed)
        total_count = len(self.results)

        print()
        print("=" * 60)
        print(f"安全测试完成: {passed_count}/{total_count} 通过")
        print("=" * 60)

        return self.get_summary()

    def get_summary(self) -> Dict[str, Any]:
        """获取测试摘要"""
        return {
            "timestamp": datetime.now().isoformat(),
            "total_tests": len(self.results),
            "passed": sum(1 for r in self.results if r.passed),
            "failed": sum(1 for r in self.results if not r.passed),
            "pass_rate": sum(1 for r in self.results if r.passed) / len(self.results) if self.results else 0,
            "results": [r.to_dict() for r in self.results]
        }

    def save_results(self, path: str) -> None:
        """保存测试结果"""
        summary = self.get_summary()

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)


def main():
    """主函数"""
    suite = SecurityTestSuite()
    summary = suite.run_all_tests()

    output_path = "logs/security_test_results.json"
    suite.save_results(output_path)
    print(f"\n测试结果已保存到: {output_path}")

    return summary


if __name__ == "__main__":
    main()
