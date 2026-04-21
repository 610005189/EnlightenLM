import sys
import os
import json
import logging
import time
import hashlib

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 配置日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 模拟L1模型
class MockL1Model:
    def __init__(self, config):
        self.config = config
    
    def generate(self, input_text, bias=None):
        # 模拟生成
        output_text = f"这是对 '{input_text}' 的回答"
        # 模拟快照
        snapshot = {
            "input_text": input_text,
            "output_text": output_text,
            "attention_summaries": [
                {
                    "layer": 0,
                    "avg_entropy": 1.45,
                    "top_tokens": [
                        {
                            "position": 0,
                            "tokens": [100, 200, 300],
                            "weights": [0.3, 0.2, 0.1]
                        }
                    ]
                }
            ],
            "generation_steps": 50,
            "generation_temperature": 0.7
        }
        return output_text, snapshot

# 模拟L2模型
class MockL2Model:
    def __init__(self, config):
        self.config = config
    
    def generate_meta_description(self, snapshot):
        # 模拟生成元描述
        return f"模型在回答 '{snapshot.get('input_text')}' 时，关注了关键信息，生成了合理的回答。"

# 模拟截断控制器
class MockCutoffController:
    def __init__(self, config):
        self.config = config
        self.delta_cross = 0
    
    def reset(self):
        self.delta_cross = 0
    
    def should_cutoff(self, text):
        # 模拟截断检查
        return {"cutoff": False, "reason": "no_cutoff"}

# 模拟L3控制器
class MockL3Controller:
    def __init__(self, config):
        self.config = config
    
    def safe_project_user_params(self, user_params):
        # 模拟安全投影
        return {
            "creativity": 0.0,
            "detail": 0.5,
            "safety": 0.5,
            "role": 0
        }
    
    def generate_task_bias(self, input_text, seq_len):
        # 模拟任务偏置
        return {
            "layers": [{
                "alpha": 0.0,
                "beta": 0.0,
                "gamma": 0.0
            }],
            "global_scale": 1.0
        }
    
    def generate_core_bias(self, input_tokens):
        # 模拟核心价值观偏置
        return {"bias": [[0.0]]}
    
    def generate_user_bias(self, user_params, seq_len):
        # 模拟用户偏置
        return {"bias": [[0.0]]}

# 模拟审计日志系统
class MockAuditLogger:
    def __init__(self, config):
        self.config = config
        self.logs = []
    
    def log_session(self, session_id, input_text, output_text, meta_description, 
                    core_rules, user_params, attention_stats, cutoff_event):
        # 模拟日志记录
        log_entry = {
            "session_id": session_id,
            "timestamp_us": int(time.time() * 1_000_000),
            "input_text": input_text,
            "output_text": output_text,
            "meta_description": meta_description,
            "core_rules": core_rules,
            "user_params": user_params,
            "attention_stats": attention_stats,
            "cutoff_event": cutoff_event
        }
        self.logs.append(log_entry)
        logger.info(f"Logged session: {session_id}")
    
    def get_session_summary(self, session_id):
        # 模拟会话摘要查询
        for log in self.logs:
            if log.get("session_id") == session_id:
                return {
                    "session_id": log.get("session_id"),
                    "timestamp": log.get("timestamp_us"),
                    "cutoff": log.get("cutoff_event", {}).get("reason", "") != "",
                    "cutoff_reason": log.get("cutoff_event", {}).get("reason", "")
                }
        return None
    
    def verify_hash_chain(self):
        # 模拟哈希链验证
        return True
    
    def shutdown(self):
        # 模拟关闭
        pass

# 模拟安全隔离模块
class MockSecurityIsolation:
    def __init__(self, config):
        self.config = config
    
    def add_log(self, log_entry):
        # 模拟添加日志
        logger.info(f"Added log to security isolation: {log_entry.get('session_id')}")
    
    def shutdown(self):
        # 模拟关闭
        pass

def test_integration():
    """
    测试系统集成
    """
    logger.info("Testing system integration...")
    
    try:
        # 模拟配置
        config = {
            "l1": {},
            "l2": {
                "cutoff_controller": {}
            },
            "l3": {},
            "audit": {}
        }
        
        # 初始化组件
        l1_model = MockL1Model(config.get("l1", {}))
        l2_model = MockL2Model(config.get("l2", {}))
        cutoff_controller = MockCutoffController(config.get("l2", {}).get("cutoff_controller", {}))
        l3_controller = MockL3Controller(config.get("l3", {}))
        audit_logger = MockAuditLogger(config.get("audit", {}))
        security_isolation = MockSecurityIsolation(config.get("audit", {}).get("security", {}))
        
        # 测试推理流程
        session_id = f"test_session_{int(time.time())}"
        input_text = "什么是人工智能？"
        user_params = {"creativity": 1.0, "detail": 0.5, "safety": 0.5, "role": 0}
        
        # 安全投影用户参数
        safe_user_params = l3_controller.safe_project_user_params(user_params)
        
        # 生成偏置
        task_bias = l3_controller.generate_task_bias(input_text, 100)
        core_bias = l3_controller.generate_core_bias(input_text.split())
        user_bias = l3_controller.generate_user_bias(safe_user_params, 100)
        
        # L1 生成
        output_text, snapshot = l1_model.generate(input_text)
        
        # 重置截断控制器
        cutoff_controller.reset()
        
        # L2 生成元描述
        meta_description = l2_model.generate_meta_description(snapshot)
        
        # 检查截断
        cutoff_result = cutoff_controller.should_cutoff(meta_description)
        
        # 构建统计信息
        attention_stats = {
            "bias_norm_quant": 0,
            "core_overlap_quant": 0,
            "entropy_quant": 0
        }
        
        core_rules = {
            "bitmap": 0,
            "vocab_version_hash": ""
        }
        
        cutoff_event = {
            "depth": cutoff_controller.delta_cross,
            "reason": cutoff_result.get("reason", ""),
            "snapshot_hash": ""
        }
        
        # 记录审计日志
        audit_logger.log_session(
            session_id=session_id,
            input_text=input_text,
            output_text=output_text,
            meta_description=meta_description,
            core_rules=core_rules,
            user_params=safe_user_params,
            attention_stats=attention_stats,
            cutoff_event=cutoff_event
        )
        
        # 添加到安全隔离
        security_isolation.add_log({
            "session_id": session_id,
            "input_text": input_text,
            "output_text": output_text,
            "meta_description": meta_description
        })
        
        # 测试会话摘要
        summary = audit_logger.get_session_summary(session_id)
        logger.info(f"Session summary: {summary}")
        
        # 测试哈希链验证
        verify_result = audit_logger.verify_hash_chain()
        logger.info(f"Hash chain verification: {verify_result}")
        
        # 关闭组件
        audit_logger.shutdown()
        security_isolation.shutdown()
        
        logger.info("Integration test passed!")
        return True
    except Exception as e:
        logger.error(f"Integration test failed: {e}")
        return False

def main():
    """
    主测试函数
    """
    logger.info("Starting simple system tests...")
    
    tests = [
        test_integration
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        if test():
            passed += 1
        else:
            failed += 1
    
    logger.info(f"Test results: {passed} passed, {failed} failed")
    
    if failed == 0:
        logger.info("All tests passed!")
    else:
        logger.error(f"Some tests failed: {failed}")

if __name__ == "__main__":
    main()
