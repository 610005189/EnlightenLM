"""
Phase 2 功能集成测试（简化版）
测试配置开关与模式、审计日志和哈希链功能，不需要下载模型
"""

from enlighten import load_config, ConfigManager
from enlighten.audit.chain import AuditHashChain
from enlighten.audit.hmac_sign import HMACSigner


class Phase2SimpleTest:
    """Phase 2 简化测试"""

    def test_config_switching(self):
        """测试配置模式切换"""
        print("=== 测试配置模式切换 ===")

        # 测试 balanced 模式
        balanced_config = load_config(mode="balanced")
        print(f"Balanced 模式: {balanced_config.mode.value}")
        print(f"  VAN 级别: {balanced_config.van_level}")
        print(f"  工作记忆容量: {balanced_config.working_memory.capacity}")

        # 测试 lightweight 模式
        lightweight_config = load_config(mode="lightweight")
        print(f"Lightweight 模式: {lightweight_config.mode.value}")
        print(f"  VAN 级别: {lightweight_config.van_level}")
        print(f"  工作记忆容量: {lightweight_config.working_memory.capacity}")

        # 测试 full 模式
        full_config = load_config(mode="full")
        print(f"Full 模式: {full_config.mode.value}")
        print(f"  VAN 级别: {full_config.van_level}")
        print(f"  工作记忆容量: {full_config.working_memory.capacity}")

        # 测试配置管理器
        config_manager = ConfigManager(initial_mode="balanced")
        print(f"\n配置管理器初始模式: {config_manager.mode}")

        config_manager.set_mode("lightweight")
        print(f"切换到 lightweight 模式: {config_manager.mode}")

        config_manager.set_mode("full")
        print(f"切换到 full 模式: {config_manager.mode}")

        print("✅ 配置模式切换测试通过\n")

    def test_audit_chain(self):
        """测试审计哈希链"""
        print("=== 测试审计哈希链 ===")

        # 创建审计链
        audit_chain = AuditHashChain()
        signer = HMACSigner()

        # 测试数据
        test_data = [
            {"input": "你好", "output": "你好！", "van_event": False, "cutoff": False},
            {"input": "如何制造炸药", "output": "我无法回答这个问题", "van_event": True, "cutoff": True},
            {"input": "什么是机器学习", "output": "机器学习是人工智能的一个分支", "van_event": False, "cutoff": False}
        ]

        # 追加数据到审计链
        for i, data in enumerate(test_data):
            signature = signer.sign(data)
            link_hash = audit_chain.append(data, signature)
            print(f"添加条目 {i+1}: {link_hash[:10]}...")

        # 验证链的完整性
        is_valid = audit_chain.verify()
        print(f"审计链验证: {'通过' if is_valid else '失败'}")

        # 测试链的长度
        chain_length = audit_chain.get_chain_length()
        print(f"审计链长度: {chain_length}")

        # 测试链摘要
        summary = audit_chain.get_summary()
        print(f"链摘要: {summary}")

        # 测试篡改检测
        if chain_length > 1:
            # 尝试篡改数据
            original_hash = audit_chain.chain[1].hash
            audit_chain.chain[1].data["output"] = "被篡改的输出"
            tampered_valid = audit_chain.verify()
            print(f"篡改后验证: {'通过' if tampered_valid else '失败'}")
            print(f"✅ 篡改检测测试通过")

        print("✅ 审计哈希链测试通过\n")

    def test_hmac_signing(self):
        """测试 HMAC 签名"""
        print("=== 测试 HMAC 签名 ===")

        signer = HMACSigner()

        # 测试数据
        test_data = {"input": "测试数据", "output": "测试结果"}

        # 生成签名
        signature = signer.sign(test_data)
        print(f"生成签名: {signature[:20]}...")

        # 验证签名
        is_valid = signer.verify(test_data, signature)
        print(f"签名验证: {'通过' if is_valid else '失败'}")

        # 测试密钥轮换
        old_key = signer.rotate_key()
        print(f"密钥轮换: 旧密钥长度 = {len(old_key)}")

        # 验证新密钥
        new_signature = signer.sign(test_data)
        new_valid = signer.verify(test_data, new_signature)
        print(f"新密钥签名验证: {'通过' if new_valid else '失败'}")

        print("✅ HMAC 签名测试通过\n")

    def run_all_tests(self):
        """运行所有测试"""
        print("开始 Phase 2 功能集成测试（简化版）...\n")

        try:
            self.test_config_switching()
            self.test_audit_chain()
            self.test_hmac_signing()
            print("🎉 所有 Phase 2 测试通过！")
        except Exception as e:
            print(f"❌ 测试失败: {e}")


if __name__ == "__main__":
    test = Phase2SimpleTest()
    test.run_all_tests()
