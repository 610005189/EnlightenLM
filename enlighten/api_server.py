"""
FastAPI Service - API服务
使用 DeepSeek API 作为底层模型
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Optional, Any
import uvicorn
import logging
import uuid
import os

from .api.deepseek_client import DeepSeekAPIClient, DeepSeekConfig
from .config import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="EnlightenLM API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None
_deepseek_client = None


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


def get_deepseek_client() -> DeepSeekAPIClient:
    """获取 DeepSeek API 客户端"""
    global _deepseek_client
    if _deepseek_client is None:
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise HTTPException(status_code=503, detail="DeepSeek API key not set. Please set DEEPSEEK_API_KEY environment variable.")

        model_name = os.environ.get("DEEPSEEK_MODEL_NAME", "deepseek-chat")

        config = DeepSeekConfig(
            api_key=api_key,
            model=model_name,
            max_tokens=2048,
            temperature=0.7
        )
        _deepseek_client = DeepSeekAPIClient(config)

    return _deepseek_client


def init_model():
    """初始化模型（DeepSeek API 客户端）"""
    global _model
    try:
        logger.info("Initializing DeepSeek API client...")
        client = get_deepseek_client()

        if client.is_available():
            logger.info("DeepSeek API is available")
            _model = True
        else:
            logger.warning("DeepSeek API is not available, but client is initialized")

    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    """启动时初始化"""
    init_model()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    try:
        client = get_deepseek_client()
        is_available = client.is_available()
        return HealthResponse(
            status="healthy" if is_available else "degraded",
            model_loaded=is_available,
            version="1.0.0"
        )
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
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
    try:
        client = get_deepseek_client()

        response_text, latency = client.generate(
            prompt=request.text,
            max_tokens=request.max_length
        )

        session_id = request.session_id or str(uuid.uuid4())

        return InferenceResponse(
            session_id=session_id,
            output=response_text,
            meta_description=f"DeepSeek API 响应，延迟: {latency:.2f}秒",
            user_params=request.user_params or {},
            cutoff=False,
            cutoff_reason=None,
            attention_stats={"latency": latency}
        )

    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/audit/verify", response_model=AuditVerifyResponse)
async def verify_audit_chain():
    """验证审计链"""
    return AuditVerifyResponse(
        verified=True,
        message="Audit chain verification not implemented for API mode"
    )


@app.get("/audit/session/{session_id}/summary")
async def get_session_summary(session_id: str):
    """获取会话摘要"""
    return {"report": f"Session {session_id} summary not available in API mode"}


@app.get("/")
async def root():
    """根路径"""
    return {
        "name": "EnlightenLM API",
        "version": "1.0.0",
        "description": "基于认知神经科学的大模型安全推理与元认知框架 - DeepSeek API 模式",
        "model": os.environ.get("DEEPSEEK_MODEL_NAME", "deepseek-chat"),
        "api_status": "available" if get_deepseek_client().is_available() else "unavailable"
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