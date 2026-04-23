"""
FastAPI Service - API服务
使用 DeepSeek API 或本地模型作为底层模型
集成完整的 L1/L2/L3 三层架构安全监控

架构说明:
- L1 生成层: DeepSeek API 或本地 Transformer 模型
- L2 工作记忆: 会话历史管理 + 注意力统计 + 熵值追踪
- L3 VAN 监控: 熵值监控 + 自指循环检测 + 变异性分析
"""

from dataclasses import dataclass
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Optional, Any
import uvicorn
import logging
import uuid
import os

from .hybrid_architecture import HybridEnlightenLM, GenerationResult
from .api.deepseek_client import DeepSeekAPIClient, DeepSeekConfig
from .config import load_config, ModeConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="EnlightenLM API", version="1.0.0", description="混合架构三层安全监控")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model: Optional[HybridEnlightenLM] = None
_deepseek_client: Optional[DeepSeekAPIClient] = None
_config: Optional[ModeConfig] = None


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
    tokens: int = Field(default=0, description="生成的token数")
    meta_description: str
    user_params: Dict[str, Any]
    cutoff: bool
    cutoff_reason: Optional[str]
    attention_stats: Dict[str, float]
    entropy_stats: Dict[str, float]
    van_event: bool
    security_verified: bool
    mode: str = Field(default="api", description="运行模式: local 或 api")


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    model_loaded: bool
    version: str
    mode: str
    security_status: Dict[str, Any]


class ModeSwitchRequest(BaseModel):
    """模式切换请求"""
    use_local_model: bool = Field(..., description="是否使用本地模型")


class ModeSwitchResponse(BaseModel):
    """模式切换响应"""
    success: bool
    mode: str
    message: str


@dataclass
class SecurityStats:
    """安全统计信息"""
    total_requests: int = 0
    blocked_requests: int = 0
    van_events: int = 0
    self_loop_events: int = 0
    cooldown_active: bool = False
    cooldown_remaining: int = 0
    block_ratio: float = 0.0


def get_deepseek_client() -> DeepSeekAPIClient:
    """获取 DeepSeek API 客户端"""
    global _deepseek_client
    if _deepseek_client is None:
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise HTTPException(status_code=503, detail="DeepSeek API key not set")

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
    """初始化模型"""
    global _model, _config

    try:
        _config = load_config("balanced")

        model_provider = _config.model_provider
        use_local = model_provider.use_local_model

        logger.info(f"Initializing EnlightenLM in {'local' if use_local else 'API'} mode...")

        api_client = None
        if not use_local:
            api_client = get_deepseek_client()
            if not api_client.is_available():
                raise RuntimeError("DeepSeek API is not available")

        _model = HybridEnlightenLM(
            use_local_model=use_local,
            local_model_name=model_provider.local_model_name,
            api_client=api_client,
            config=_config
        )

        status = _model.get_status()
        logger.info(f"Model initialized successfully: {status}")

    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        raise


def get_model() -> HybridEnlightenLM:
    """获取模型实例"""
    global _model
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    return _model


@app.on_event("startup")
async def startup_event():
    """启动时初始化"""
    init_model()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    global _model

    try:
        if _model is None:
            return HealthResponse(
                status="unhealthy",
                model_loaded=False,
                version="1.0.0",
                mode="unknown",
                security_status={}
            )

        status = _model.get_status()
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            version="1.0.0",
            mode=status["mode"],
            security_status=status["van_stats"]
        )

    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            version="1.0.0",
            mode="error",
            security_status={"error": str(e)}
        )


@app.post("/inference", response_model=InferenceResponse)
async def inference(request: InferenceRequest):
    """
    推理接口 - 完整的 L1/L2/L3 三层架构

    L1 生成层:
    - 本地模型或 DeepSeek API 生成

    L2 工作记忆:
    - 会话历史管理
    - 注意力统计
    - 熵值追踪

    L3 VAN 监控:
    - 输入预检 (敏感词、自指循环)
    - 输出后检 (敏感内容、变异性)
    - 熵值监控 (低熵截断)
    - 冷却机制

    Args:
        request: 推理请求

    Returns:
        InferenceResponse: 推理响应
    """
    try:
        model = get_model()

        result: GenerationResult = model.generate(
            prompt=request.text,
            max_length=request.max_length
        )

        session_id = request.session_id or str(uuid.uuid4())

        meta_parts = []
        if model.use_local_model:
            meta_parts.append(f"本地模型 ({model.local_model_name})")
        else:
            meta_parts.append("DeepSeek API")
        meta_parts.append(f"L3安全监控{'通过' if result.security_verified else '触发'}")
        if result.cutoff:
            meta_parts.append(f"截断: {result.cutoff_reason}")
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "内容被安全监控系统拦截",
                    "reason": result.cutoff_reason,
                    "van_event": result.van_event,
                    "session_id": session_id
                }
            )

        return InferenceResponse(
            session_id=session_id,
            output=result.text,
            tokens=result.tokens,
            meta_description=" | ".join(meta_parts),
            user_params=request.user_params or {},
            cutoff=result.cutoff,
            cutoff_reason=result.cutoff_reason,
            attention_stats=result.entropy_stats,
            entropy_stats=result.entropy_stats,
            van_event=result.van_event,
            security_verified=result.security_verified,
            mode="local" if model.use_local_model else "api"
        )

    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/mode/switch", response_model=ModeSwitchResponse)
async def switch_mode(request: ModeSwitchRequest):
    """
    切换运行模式

    在本地模型模式和API模式之间切换
    """
    global _model, _deepseek_client

    try:
        use_local = request.use_local_model

        logger.info(f"Switching mode to: {'local' if use_local else 'API'}")

        if use_local:
            api_client = None
        else:
            api_client = get_deepseek_client()
            if not api_client.is_available():
                return ModeSwitchResponse(
                    success=False,
                    mode="unchanged",
                    message="DeepSeek API is not available. Please check your API key."
                )

        _model = HybridEnlightenLM(
            use_local_model=use_local,
            local_model_name=_config.model_provider.local_model_name if _config else "distilgpt2",
            api_client=api_client,
            config={
                "max_history": 100,
                "context_window": 4096,
                "entropy_window": 20,
                "van_threshold": 0.7,
                "entropy_threshold": 0.3,
                "variance_threshold": 0.05,
                "cooldown_steps": 5,
                "van_enabled": True
            }
        )

        new_mode = "local" if use_local else "api"
        return ModeSwitchResponse(
            success=True,
            mode=new_mode,
            message=f"Successfully switched to {new_mode} mode"
        )

    except Exception as e:
        logger.error(f"Mode switch error: {e}")
        return ModeSwitchResponse(
            success=False,
            mode="error",
            message=str(e)
        )


@app.get("/status")
async def get_status():
    """获取系统状态"""
    global _model

    if _model is None:
        return {"status": "not_initialized"}

    return _model.get_status()


@app.get("/security/stats")
async def get_security_stats() -> Dict[str, Any]:
    """获取安全统计信息"""
    global _model

    if _model is None:
        return {"error": "Model not initialized"}

    van_stats = _model.get_van_stats()

    return {
        "van_stats": SecurityStats(
            total_requests=van_stats.get("total_requests", 0),
            blocked_requests=van_stats.get("blocked_requests", 0),
            van_events=van_stats.get("van_events", 0),
            self_loop_events=0,
            cooldown_active=van_stats.get("cooldown_active", False),
            cooldown_remaining=van_stats.get("cooldown_remaining", 0),
            block_ratio=van_stats.get("block_ratio", 0.0)
        ),
        "entropy_stats": _model.get_entropy_stats(),
        "attention_stats": {
            "entropy": _model.get_attention_stats().entropy,
            "stability": _model.get_attention_stats().stability_score
        }
    }


@app.post("/security/reset")
async def reset_security():
    """重置安全监控状态"""
    global _model

    if _model is None:
        return {"error": "Model not initialized"}

    _model.reset()
    return {"status": "reset", "message": "Security monitor reset successfully"}


@app.get("/audit/verify")
async def verify_audit_chain():
    """验证审计链"""
    global _model

    if _model is None:
        return {"verified": False, "message": "Model not initialized"}

    van_stats = _model.get_van_stats()
    return {
        "verified": van_stats["block_ratio"] < 0.5,
        "message": f"VAN events: {van_stats['van_events']}, Blocked: {van_stats['blocked_requests']}",
        "details": van_stats
    }


@app.get("/")
async def root():
    """根路径"""
    global _model

    mode = "unknown"
    if _model is not None:
        mode = "local" if _model.use_local_model else "api"

    return {
        "name": "EnlightenLM API",
        "version": "1.0.0",
        "description": "混合架构 L1/L2/L3 三层安全监控系统",
        "mode": mode,
        "architecture": {
            "L1": "本地模型 / DeepSeek API",
            "L2": "工作记忆 (会话历史 + 注意力统计 + 熵值追踪)",
            "L3": "VAN 监控 (熵值 + 自指循环 + 变异性分析)"
        },
        "security": "启用"
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