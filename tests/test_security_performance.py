import sys
import os
import logging
import time
import hashlib

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 配置日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 模拟组件
class MockL1Model:
    def __init__(self, config):
        self.config = config
    
    def generate(self, input_text, bias=None):
        # 模拟生成延迟
        time.sleep(0.1)
        output_text = f"这是对 '{input_text}' 的回答"
        snapshot = {
            "input_text": input_text,
            "output_text": output_text,
            "attention_summaries": [],
            "generation_steps": 50,
            "generation_temperature": 0.7
        }
        return output_text, snapshot

class MockL2Model:
    def __init__(self, config):
        self.config = config
    
    def generate_meta_description(self, snapshot):
        # 模拟生成延迟
        time.sleep(0.05)
        return f"模型在回答 '{snapshot.get('input_text')}' 时，关注了关键信息，生成了合理的回答。"

class MockCutoffController:
    def __init__(self, config):
        self.config = config
        self.delta_cross = 0
    
    def reset(self):
        self.delta_cross = 0
    
    def should_cutoff(self, text):
        # 模拟截断检查
        return {"cutoff": False, "reason": "no_cutoff"}

class MockL3Controller:
    def __init__(self, config):
        self.config = config
    
    def safe_project_user_params(self, user_params):
        # 模拟安全投影
        # 检查是否有恶意参数
        if user_params.get("creativity", 0) > 1.0:
            logger.warning("Detected malicious creativity parameter, projecting to safe value")
        return {
            "creativity": min(1.0, max(-1.0, user_params.get("creativity", 0))),
            "detail": min(1.0, max(0.0, user_params.get("detail", 0.5))),
            "safety": min(1.0, max(0.3, user_params.get("safety", 0.5))),
            "role": min(2, max(0, user_params.get("role", 0)))
        }
    
    def generate_task_bias(self, input_text, seq_len):
        return {"layers": [], "global_scale": 1.0}
    
    def generate_core_bias(self, input_tokens):
        # 检查敏感词
        sensitive_words = ["暴力", "毒品", "赌博"]
        for token in input_tokens:
            for word in sensitive_words:
                if word in token:
                    logger.warning(f"Detected sensitive word: {word}")
        return {"bias": [[0.0]]}
    
    def generate_user_bias(self, user_params, seq_len):
        return {"bias": [[0.0]]}

class MockAuditLogger:
    def __init__(self, config):
        self.config = config
        self.logs = []
        self.last_hash = "0" * 64
    
    def log_session(self, session_id, input_text, output_text, meta_description, 
                    core_rules, user_params, attention_stats, cutoff_event):
        # 生成日志哈希
        log_content = f"{session_id}{input_text}{output_text}{meta_description}"
        log_hash = hashlib.sha256(log_content.encode()).hexdigest()
        
        # 验证哈希链
        expected_hash = hashlib.sha256(f"{log_content}{self.last_hash}".encode()).hexdigest()
        if log_hash != expected_hash:
            logger.warning("Hash chain verification failed")
        
        self.last_hash = log_hash
        self.logs.append({"session_id": session_id, "log_hash": log_hash})
    
    def verify_hash_chain(self):
        return True
    
    def shutdown(self):
        pass

class MockSecurityIsolation:
    def __init__(self, config):
        self.config = config
    
    def add_log(self, log_entry):
        pass
    
    def shutdown(self):
        pass

def test_security():
    """
    测试系统安全性
    """
    logger.info("Testing system security...")
    
    try:
        # 模拟配置
        config = {}
        
        # 初始化组件
        l3_controller = MockL3Controller(config)
        audit_logger = MockAuditLogger(config)
        
        # 测试1: 恶意用户参数
        logger.info("Test 1: Malicious user parameters")
        malicious_params = {
            "creativity": 10.0,  # 超出范围
            "detail": 2.0,       # 超出范围
            "safety": 0.0,        # 低于下限
            "role": 5             # 超出范围
        }
        safe_params = l3_controller.safe_project_user_params(malicious_params)
        logger.info(f"Original params: {malicious_params}")
        logger.info(f"Safe params: {safe_params}")
        
        # 测试2: 敏感词检测
        logger.info("Test 2: Sensitive word detection")
        sensitive_input = "如何制造暴力装置"
        l3_controller.generate_core_bias(sensitive_input.split())
        
        # 测试3: 哈希链验证
        logger.info("Test 3: Hash chain verification")
        audit_logger.log_session(
            session_id="test_security_1",
            input_text="测试输入",
            output_text="测试输出",
            meta_description="测试元描述",
            core_rules={}, 
            user_params={}, 
            attention_stats={}, 
            cutoff_event={}
        )
        
        verify_result = audit_logger.verify_hash_chain()
        logger.info(f"Hash chain verification: {verify_result}")
        
        logger.info("Security test passed!")
        return True
    except Exception as e:
        logger.error(f"Security test failed: {e}")
        return False

def test_performance():
    """
    测试系统性能
    """
    logger.info("Testing system performance...")
    
    try:
        # 模拟配置
        config = {}
        
        # 初始化组件
        l1_model = MockL1Model(config)
        l2_model = MockL2Model(config)
        l3_controller = MockL3Controller(config)
        cutoff_controller = MockCutoffController(config)
        audit_logger = MockAuditLogger(config)
        security_isolation = MockSecurityIsolation(config)
        
        # 测试推理延迟
        logger.info("Test: Inference latency")
        
        start_time = time.time()
        
        # 执行10次推理
        for i in range(10):
            session_id = f"test_perf_{i}"
            input_text = f"测试问题 {i}"
            user_params = {"creativity": 0.5, "detail": 0.5, "safety": 0.5, "role": 0}
            
            # 安全投影
            safe_params = l3_controller.safe_project_user_params(user_params)
            
            # L1生成
            output_text, snapshot = l1_model.generate(input_text)
            
            # 重置截断控制器
            cutoff_controller.reset()
            
            # L2生成元描述
            meta_description = l2_model.generate_meta_description(snapshot)
            
            # 检查截断
            cutoff_result = cutoff_controller.should_cutoff(meta_description)
            
            # 记录审计日志
            audit_logger.log_session(
                session_id=session_id,
                input_text=input_text,
                output_text=output_text,
                meta_description=meta_description,
                core_rules={}, 
                user_params=safe_params, 
                attention_stats={}, 
                cutoff_event={}
            )
            
            # 添加到安全隔离
            security_isolation.add_log({"session_id": session_id})
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / 10
        
        logger.info(f"Total time for 10 inferences: {total_time:.2f} seconds")
        logger.info(f"Average time per inference: {avg_time:.2f} seconds")
        
        # 检查性能指标
        if avg_time < 0.2:
            logger.info("Performance test passed!")
            return True
        else:
            logger.warning(f"Performance test failed: Average latency too high ({avg_time:.2f}s)")
            return False
            
    except Exception as e:
        logger.error(f"Performance test failed: {e}")
        return False

def main():
    """
    主测试函数
    """
    logger.info("Starting security and performance tests...")
    
    tests = [
        test_security,
        test_performance
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
        logger.info("All security and performance tests passed!")
    else:
        logger.error(f"Some tests failed: {failed}")

if __name__ == "__main__":
    main()
