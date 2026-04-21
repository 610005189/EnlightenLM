import sys
import os
import json
import logging
import time

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 配置日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 导入组件
from src.l1.base_model import L1BaseModel
from src.l2.self_description_model import L2SelfDescriptionModel
from src.l2.cutoff_controller import CutoffController
from src.l3.meta_attention_controller import L3MetaAttentionController
from src.audit.audit_logger import AuditLogger
import yaml

# 加载配置
def load_config():
    config_path = "config/config.yaml"
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise

config = load_config()

def test_l1_model():
    """
    测试L1基座模型
    """
    logger.info("Testing L1 model...")
    
    try:
        # 初始化L1模型
        l1_model = L1BaseModel(config.get("l1", {}))
        
        # 测试生成
        input_text = "什么是人工智能？"
        output_text, snapshot = l1_model.generate(input_text)
        
        logger.info(f"Input: {input_text}")
        logger.info(f"Output: {output_text}")
        logger.info(f"Snapshot keys: {list(snapshot.keys())}")
        
        logger.info("L1 model test passed!")
        return True
    except Exception as e:
        logger.error(f"L1 model test failed: {e}")
        return False

def test_l2_model():
    """
    测试L2自描述小模型
    """
    logger.info("Testing L2 model...")
    
    try:
        # 初始化L2模型
        l2_model = L2SelfDescriptionModel(config.get("l2", {}))
        
        # 创建测试快照
        test_snapshot = {
            "input_text": "什么是人工智能？",
            "output_text": "人工智能是指由人制造出来的系统所表现出来的智能。",
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
        
        # 测试生成元描述
        meta_description = l2_model.generate_meta_description(test_snapshot)
        
        logger.info(f"Meta description: {meta_description}")
        
        logger.info("L2 model test passed!")
        return True
    except Exception as e:
        logger.error(f"L2 model test failed: {e}")
        return False

def test_cutoff_controller():
    """
    测试截断控制器
    """
    logger.info("Testing cutoff controller...")
    
    try:
        # 初始化截断控制器
        cutoff_controller = CutoffController(config.get("l2", {}).get("cutoff_controller", {}))
        
        # 测试语义循环检测
        test_text = "这是一个测试。这是一个测试。这是一个测试。"
        cutoff_result = cutoff_controller.should_cutoff(test_text)
        
        logger.info(f"Cutoff result for semantic cycle: {cutoff_result}")
        
        # 测试自我指涉检测
        test_text = "我注意到这个问题很重要。"
        cutoff_result = cutoff_controller.should_cutoff(test_text)
        
        logger.info(f"Cutoff result for self reference: {cutoff_result}")
        
        # 测试短语重复检测
        test_text = "重要的事情说三遍。重要的事情说三遍。重要的事情说三遍。"
        cutoff_result = cutoff_controller.should_cutoff(test_text)
        
        logger.info(f"Cutoff result for phrase repetition: {cutoff_result}")
        
        logger.info("Cutoff controller test passed!")
        return True
    except Exception as e:
        logger.error(f"Cutoff controller test failed: {e}")
        return False

def test_l3_controller():
    """
    测试L3元注意力控制器
    """
    logger.info("Testing L3 controller...")
    
    try:
        # 初始化L3控制器
        l3_controller = L3MetaAttentionController(config.get("l3", {}))
        
        # 测试任务偏置生成
        input_text = "什么是人工智能？"
        task_bias = l3_controller.generate_task_bias(input_text, 100)
        
        logger.info(f"Task bias layers: {len(task_bias.get('layers', []))}")
        
        # 测试核心价值观偏置生成
        input_tokens = input_text.split()
        core_bias = l3_controller.generate_core_bias(input_tokens)
        
        logger.info(f"Core bias shape: {len(core_bias.get('bias', []))}x{len(core_bias.get('bias', [])[0]) if core_bias.get('bias', []) else 0}")
        
        # 测试用户参数安全投影
        user_params = {
            "creativity": 2.0,  # 超出范围
            "detail": 0.5,
            "safety": 0.1,  # 低于下限
            "role": 0
        }
        safe_user_params = l3_controller.safe_project_user_params(user_params)
        
        logger.info(f"Original user params: {user_params}")
        logger.info(f"Safe user params: {safe_user_params}")
        
        # 测试用户偏置生成
        user_bias = l3_controller.generate_user_bias(safe_user_params, 100)
        
        logger.info(f"User bias shape: {len(user_bias.get('bias', []))}x{len(user_bias.get('bias', [])[0]) if user_bias.get('bias', []) else 0}")
        
        logger.info("L3 controller test passed!")
        return True
    except Exception as e:
        logger.error(f"L3 controller test failed: {e}")
        return False

def test_audit_logger():
    """
    测试审计日志系统
    """
    logger.info("Testing audit logger...")
    
    try:
        # 初始化审计日志系统
        audit_logger = AuditLogger(config.get("audit", {}))
        
        # 测试日志记录
        session_id = f"test_session_{int(time.time())}"
        audit_logger.log_session(
            session_id=session_id,
            input_text="测试输入",
            output_text="测试输出",
            meta_description="测试元描述",
            core_rules={"bitmap": 0, "vocab_version_hash": ""},
            user_params={"creativity": 0.0, "detail": 0.5, "safety": 0.5, "role": 0},
            attention_stats={"bias_norm_quant": 0, "core_overlap_quant": 0, "entropy_quant": 0},
            cutoff_event={"depth": 0, "reason": "", "snapshot_hash": ""}
        )
        
        # 等待日志处理
        time.sleep(1)
        
        # 测试会话摘要查询
        summary = audit_logger.get_session_summary(session_id)
        logger.info(f"Session summary: {summary}")
        
        # 测试哈希链验证
        verify_result = audit_logger.verify_hash_chain()
        logger.info(f"Hash chain verification: {verify_result}")
        
        # 关闭审计日志系统
        audit_logger.shutdown()
        
        logger.info("Audit logger test passed!")
        return True
    except Exception as e:
        logger.error(f"Audit logger test failed: {e}")
        return False

def main():
    """
    主测试函数
    """
    logger.info("Starting system tests...")
    
    tests = [
        test_l1_model,
        test_l2_model,
        test_cutoff_controller,
        test_l3_controller,
        test_audit_logger
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
