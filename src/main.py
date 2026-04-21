import os
import sys
import logging
import yaml
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import Dict, Optional
import uuid

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入组件
from src.l1.base_model import L1BaseModel
from src.l1.attention_bias import AttentionBias
from src.l2.self_description_model import L2SelfDescriptionModel
from src.l2.cutoff_controller import CutoffController
from src.l3.meta_attention_controller import L3MetaAttentionController
from src.audit.audit_logger import AuditLogger
from src.audit.security_isolation import SecurityIsolation
from src.audit.threat_detection import ThreatDetection

# 配置日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

# 初始化组件
logger.info("Initializing components...")

# L1 基座模型
l1_model = L1BaseModel(config.get("l1", {}))

# L2 自描述小模型
l2_model = L2SelfDescriptionModel(config.get("l2", {}))

# 截断控制器
cutoff_controller = CutoffController(config.get("l2", {}).get("cutoff_controller", {}))

# L3 元注意力控制器
l3_controller = L3MetaAttentionController(config.get("l3", {}))

# 注意力偏置
attention_bias = AttentionBias(config.get("l1", {}).get("attention_bias", {}).get("rank", 8))

# 审计日志系统
audit_logger = AuditLogger(config.get("audit", {}))

# 安全隔离模块
security_isolation = SecurityIsolation(config.get("audit", {}).get("security", {}))

# 威胁检测模块
threat_detection = ThreatDetection(config.get("audit", {}).get("threat_detection", {}))

# 创建FastAPI应用
app = FastAPI(
    title="觉悟三层架构 API",
    description="安全、自指、可审计的大模型推理系统",
    version="2.1"
)

# 配置CORS
if config.get("api", {}).get("cors", True):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.get("api", {}).get("allowed_origins", ["*"]),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# 健康检查端点
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# 推理端点
@app.post("/inference")
async def inference(input_data: Dict, client_ip: Optional[str] = None):
    """
    推理端点
    
    Args:
        input_data: 输入数据，包含以下字段：
            - text: 输入文本
            - user_params: 用户参数（可选）
        client_ip: 客户端IP
    """
    try:
        # 生成会话ID
        session_id = str(uuid.uuid4())
        
        # 提取输入文本
        input_text = input_data.get("text", "")
        if not input_text:
            raise HTTPException(status_code=400, detail="Text is required")
        
        # 检测威胁
        is_threat, threat_info = threat_detection.detect_threat(input_text, client_ip)
        if is_threat:
            logger.warning(f"Threat detected: {threat_info}")
            # 记录威胁
            audit_logger.log_session(
                session_id=session_id,
                input_text=input_text,
                output_text="",
                meta_description="",
                core_rules={},
                user_params={},
                attention_stats={},
                cutoff_event={"reason": f"Threat detected: {threat_info['threats']}"}
            )
            raise HTTPException(status_code=400, detail=f"Threat detected: {[t['description'] for t in threat_info['threats']]}")
        
        # 提取用户参数
        user_params = input_data.get("user_params", {})
        
        # 安全投影用户参数
        safe_user_params = l3_controller.safe_project_user_params(user_params)
        
        # 生成任务偏置
        task_bias = l3_controller.generate_task_bias(input_text, 100)  # 假设序列长度为100
        
        # 生成核心价值观偏置
        # 这里需要分词，简化处理
        input_tokens = input_text.split()
        core_bias = l3_controller.generate_core_bias(input_tokens)
        
        # 生成用户偏置
        user_bias = l3_controller.generate_user_bias(safe_user_params, 100)
        
        # 组合偏置
        # 这里需要根据实际情况实现偏置组合
        combined_bias = None
        
        # L1 生成回答
        output_text, snapshot = l1_model.generate(input_text, combined_bias)
        
        # 重置截断控制器
        cutoff_controller.reset()
        
        # L2 生成元描述
        meta_description = l2_model.generate_meta_description(snapshot)
        
        # 检查截断
        cutoff_result = cutoff_controller.should_cutoff(meta_description)
        
        # 构建注意力统计
        attention_stats = {
            "bias_norm_quant": 0,  # 占位符
            "core_overlap_quant": 0,  # 占位符
            "entropy_quant": 0  # 占位符
        }
        
        # 构建核心规则信息
        core_rules = {
            "bitmap": 0,  # 占位符
            "vocab_version_hash": ""  # 占位符
        }
        
        # 构建截断事件
        cutoff_event = {
            "depth": cutoff_controller.delta_cross,
            "reason": cutoff_result.get("reason", ""),
            "snapshot_hash": ""  # 占位符
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
        
        # 返回结果
        return {
            "session_id": session_id,
            "output": output_text,
            "meta_description": meta_description,
            "user_params": safe_user_params,
            "cutoff": cutoff_result.get("cutoff", False),
            "cutoff_reason": cutoff_result.get("reason", "")
        }
        
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# 审计日志查询端点
@app.get("/audit/session/{session_id}/summary")
async def get_session_summary(session_id: str):
    """
    获取会话摘要
    
    Args:
        session_id: 会话ID
    """
    try:
        summary = audit_logger.get_session_summary(session_id)
        if not summary:
            raise HTTPException(status_code=404, detail="Session not found")
        return summary
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session summary: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# 哈希链验证端点
@app.get("/audit/verify")
async def verify_hash_chain():
    """
    验证哈希链的完整性
    """
    try:
        result = audit_logger.verify_hash_chain()
        return {
            "verified": result,
            "message": "Hash chain is valid" if result else "Hash chain is broken"
        }
    except Exception as e:
        logger.error(f"Error verifying hash chain: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# 威胁检测统计端点
@app.get("/audit/threat/statistics")
async def get_threat_statistics():
    """
    获取威胁检测统计信息
    """
    try:
        statistics = threat_detection.get_statistics()
        return statistics
    except Exception as e:
        logger.error(f"Error getting threat statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# 重置威胁检测统计端点
@app.post("/audit/threat/reset")
async def reset_threat_statistics():
    """
    重置威胁检测统计信息
    """
    try:
        threat_detection.reset_statistics()
        return {"message": "Threat detection statistics reset successfully"}
    except Exception as e:
        logger.error(f"Error resetting threat statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# 主函数
if __name__ == "__main__":
    api_config = config.get("api", {})
    host = api_config.get("host", "0.0.0.0")
    port = api_config.get("port", 8000)
    
    logger.info(f"Starting server on {host}:{port}")
    
    try:
        uvicorn.run(
            "src.main:app",
            host=host,
            port=port,
            reload=True
        )
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
    finally:
        # 关闭组件
        audit_logger.shutdown()
        security_isolation.shutdown()
        logger.info("Server shutdown completed")
