"""
FastAPI Service - API服务
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Optional, Any, List
import uvicorn
import logging
import uuid

from .main import EnlightenLM, InferenceResult
from .config import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="EnlightenLM API", version="1.0.0")

_model = None


class InferenceRequest(BaseModel):
    """推理请求"""
    text: str = Field(..., description="输入文本")
    user_params: Optional[Dict[str, Any]] = Field(default=None, description="用户参数")
    session_id: Optional[str] = Field(default=None, description="会话ID")
    max_length: int = Field(default=2048, description="最大生成长度")


class InferenceResponse(BaseModel):
    """推理响应"""
    session_id: str
    output: str
    meta_description: str
    user_params: Dict[str, Any]
    cutoff: bool
    cutoff_reason: Optional[str]
    attention_stats: Dict[str, float]


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    model_loaded: bool
    version: str


class AuditVerifyResponse(BaseModel):
    """审计验证响应"""
    verified: bool
    message: str


def get_model() -> EnlightenLM:
    """获取模型实例"""
    global _model
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    return _model


def init_model():
    """初始化模型"""
    global _model
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM

        logger.info("Loading model...")

        config = load_config()

        tokenizer = AutoTokenizer.from_pretrained(config.l1.model_name)
        model = AutoModelForCausalLM.from_pretrained(config.l1.model_name)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        _model = EnlightenLM(model, tokenizer, config)

        logger.info("Model loaded successfully")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    """启动时初始化模型"""
    init_model()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    model = get_model()
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        version="1.0.0"
    )


@app.post("/inference", response_model=InferenceResponse)
async def inference(request: InferenceRequest):
    """
    推理接口

    Args:
        request: 推理请求

    Returns:
        InferenceResponse: 推理响应
    """
    model = get_model()

    try:
        result = model.generate(
            text=request.text,
            user_params=request.user_params,
            session_id=request.session_id,
            max_length=request.max_length
        )

        return InferenceResponse(
            session_id=result.session_id,
            output=result.output,
            meta_description=result.meta_description,
            user_params=result.user_params,
            cutoff=result.cutoff,
            cutoff_reason=result.cutoff_reason,
            attention_stats=result.attention_stats
        )

    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/audit/verify", response_model=AuditVerifyResponse)
async def verify_audit_chain():
    """验证审计链"""
    model = get_model()

    try:
        verified = model.verify_audit_chain()
        return AuditVerifyResponse(
            verified=verified,
            message="Hash chain is valid" if verified else "Hash chain verification failed"
        )

    except Exception as e:
        logger.error(f"Audit verification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/audit/session/{session_id}/summary")
async def get_session_summary(session_id: str):
    """获取会话摘要"""
    model = get_model()

    try:
        report = model.generate_review_report(session_id)
        return {"report": report}

    except Exception as e:
        logger.error(f"Report generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """根路径"""
    return {
        "name": "EnlightenLM API",
        "version": "1.0.0",
        "description": "基于认知神经科学的大模型安全推理与元认知框架"
    }


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """
    运行服务器

    Args:
        host: 主机地址
        port: 端口
    """
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
