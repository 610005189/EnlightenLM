"""
Async Review Manager - 异步审核系统
独立进程运行1.5B审核模型，定期复盘生成内容
"""

import os
import time
import queue
import threading
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReviewPriority(Enum):
    """审核优先级"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class ReviewResult:
    """审核结果"""
    session_id: str
    factuality_score: float
    safety_score: float
    issues: List[str]
    timestamp: float
    priority: ReviewPriority
    processed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "factuality_score": self.factuality_score,
            "safety_score": self.safety_score,
            "issues": self.issues,
            "timestamp": self.timestamp,
            "priority": self.priority.name,
            "processed": self.processed
        }


class ReviewRequest:
    """审核请求"""
    def __init__(
        self,
        session_id: str,
        content: str,
        priority: ReviewPriority = ReviewPriority.NORMAL
    ):
        self.session_id = session_id
        self.content = content
        self.priority = priority
        self.timestamp = time.time()


class AsyncReviewManager:
    """
    异步审核进程管理器

    功能:
    1. 启动独立审核进程
    2. 接收审核任务到队列
    3. 使用1.5B模型进行内容审核
    4. 返回审核结果

    Args:
        model_name: 审核模型名称
        interval: 审核间隔（每隔N个请求或固定时间）
        queue_size: 队列最大长度
    """

    def __init__(
        self,
        model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        interval: int = 32,
        queue_size: int = 100
    ):
        self.model_name = model_name
        self.interval = interval
        self.queue_size = queue_size

        self.review_queue: queue.Queue = queue.PriorityQueue(maxsize=queue_size)
        self.results: Dict[str, ReviewResult] = {}
        self.results_lock = threading.Lock()

        self.review_process = None
        self.worker_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

        self.request_counter = 0
        self.last_review_time = time.time()

        self.model = None
        self.tokenizer = None
        self.model_loaded = False

        self._load_model_sync()

    def _load_model_sync(self) -> None:
        """
        同步加载模型（仅加载，不启动独立进程）
        用于快速原型验证
        """
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers not available, using rule-based review")
            self.model_loaded = False
            return

        try:
            logger.info(f"Loading review model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.eval()
            self.model_loaded = True
            logger.info("Review model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load model: {e}, using rule-based review")
            self.model_loaded = False

    def start(self) -> None:
        """
        启动异步审核线程
        """
        if self.worker_thread is not None and self.worker_thread.is_alive():
            logger.warning("Review manager already running")
            return

        self.stop_event.clear()
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        logger.info("Async review manager started")

    def stop(self) -> None:
        """
        停止异步审核线程
        """
        if self.worker_thread is None:
            return

        self.stop_event.set()
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5.0)
        logger.info("Async review manager stopped")

    def _worker_loop(self) -> None:
        """
        审核工作线程循环
        """
        while not self.stop_event.is_set():
            try:
                request = self.review_queue.get(timeout=1.0)
                self._process_review(request)
                self.review_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in review worker: {e}")

    def _process_review(self, request: ReviewRequest) -> None:
        """
        处理审核请求

        Args:
            request: 审核请求
        """
        if self.model_loaded and self.model is not None:
            result = self._model_based_review(request)
        else:
            result = self._rule_based_review(request)

        with self.results_lock:
            self.results[request.session_id] = result

        logger.debug(f"Review completed for session {request.session_id}: "
                    f"factuality={result.factuality_score:.2f}, safety={result.safety_score:.2f}")

    def _model_based_review(self, request: ReviewRequest) -> ReviewResult:
        """
        基于模型的审核

        Args:
            request: 审核请求

        Returns:
            ReviewResult: 审核结果
        """
        try:
            inputs = self.tokenizer(
                request.content,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

            probs = torch.softmax(logits, dim=-1)
            safety_score = probs[0, 1].item() if probs.shape[1] > 1 else 0.5

            factuality_score = 1.0 - abs(safety_score - 0.5) * 2

            issues = []
            if safety_score > 0.8:
                issues.append("可能存在有害内容")
            if factuality_score < 0.5:
                issues.append("可能存在事实性错误")

            return ReviewResult(
                session_id=request.session_id,
                factuality_score=factuality_score,
                safety_score=safety_score,
                issues=issues,
                timestamp=request.timestamp,
                priority=request.priority,
                processed=True
            )
        except Exception as e:
            logger.error(f"Model-based review failed: {e}")
            return self._rule_based_review(request)

    def _rule_based_review(self, request: ReviewRequest) -> ReviewResult:
        """
        基于规则的简单审核

        Args:
            request: 审核请求

        Returns:
            ReviewResult: 审核结果
        """
        content_lower = request.content.lower()

        sensitive_keywords = [
            "暴力", "犯罪", "毒品", "赌博", "色情",
            "欺诈", "仇恨", "自伤", "武器", "炸弹"
        ]

        issues = []
        for keyword in sensitive_keywords:
            if keyword in content_lower:
                issues.append(f"检测到敏感词: {keyword}")

        safety_score = 1.0 if not issues else 0.3
        factuality_score = 0.8 if len(request.content) > 50 else 0.5

        return ReviewResult(
            session_id=request.session_id,
            factuality_score=factuality_score,
            safety_score=safety_score,
            issues=issues,
            timestamp=request.timestamp,
            priority=request.priority,
            processed=True
        )

    def submit_for_review(
        self,
        session_id: str,
        content: str,
        priority: ReviewPriority = ReviewPriority.NORMAL
    ) -> bool:
        """
        提交审核任务

        Args:
            session_id: 会话ID
            content: 待审核内容
            priority: 审核优先级

        Returns:
            是否提交成功
        """
        try:
            request = ReviewRequest(session_id, content, priority)
            self.review_queue.put((priority.value, request), timeout=1.0)
            self.request_counter += 1
            logger.debug(f"Review submitted for session {session_id}")
            return True
        except queue.Full:
            logger.warning("Review queue full, rejecting request")
            return False

    def get_review_result(self, session_id: str) -> Optional[ReviewResult]:
        """
        获取审核结果

        Args:
            session_id: 会话ID

        Returns:
            审核结果，不存在则返回None
        """
        with self.results_lock:
            return self.results.get(session_id)

    def has_result(self, session_id: str) -> bool:
        """
        检查是否有审核结果

        Args:
            session_id: 会话ID

        Returns:
            是否有结果
        """
        with self.results_lock:
            return session_id in self.results

    def get_pending_count(self) -> int:
        """
        获取待处理请求数量

        Returns:
            队列中的请求数
        """
        return self.review_queue.qsize()

    def clear_old_results(self, max_age_seconds: float = 3600) -> int:
        """
        清理旧的审核结果

        Args:
            max_age_seconds: 最大保留时间

        Returns:
            清理的结果数量
        """
        current_time = time.time()
        cleared = 0

        with self.results_lock:
            to_remove = []
            for session_id, result in self.results.items():
                if current_time - result.timestamp > max_age_seconds:
                    to_remove.append(session_id)

            for session_id in to_remove:
                del self.results[session_id]
                cleared += 1

        return cleared

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取审核统计信息

        Returns:
            统计信息字典
        """
        total_reviews = len(self.results)
        avg_factuality = 0.0
        avg_safety = 0.0
        issues_count = 0

        if total_reviews > 0:
            for result in self.results.values():
                avg_factuality += result.factuality_score
                avg_safety += result.safety_score
                issues_count += len(result.issues)

            avg_factuality /= total_reviews
            avg_safety /= total_reviews

        return {
            "model_loaded": self.model_loaded,
            "model_name": self.model_name,
            "total_reviews": total_reviews,
            "pending_reviews": self.get_pending_count(),
            "avg_factuality_score": avg_factuality,
            "avg_safety_score": avg_safety,
            "total_issues": issues_count,
            "request_counter": self.request_counter,
            "last_review_time": self.last_review_time
        }


class ReviewSession:
    """
    审核会话
    用于管理单个会话的审核状态
    """

    def __init__(self, session_id: str, manager: AsyncReviewManager):
        self.session_id = session_id
        self.manager = manager
        self.submitted_content: List[str] = []

    def submit(self, content: str, priority: ReviewPriority = ReviewPriority.NORMAL) -> bool:
        """
        提交内容进行审核

        Args:
            content: 内容
            priority: 优先级

        Returns:
            是否提交成功
        """
        self.submitted_content.append(content)
        return self.manager.submit_for_review(self.session_id, content, priority)

    def get_result(self) -> Optional[ReviewResult]:
        """
        获取审核结果

        Returns:
            审核结果
        """
        return self.manager.get_review_result(self.session_id)

    def wait_for_result(self, timeout: float = 30.0) -> Optional[ReviewResult]:
        """
        等待审核结果

        Args:
            timeout: 超时时间（秒）

        Returns:
            审核结果
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            result = self.get_result()
            if result is not None:
                return result
            time.sleep(0.1)
        return None

    def is_complete(self) -> bool:
        """
        检查审核是否完成

        Returns:
            是否完成
        """
        return self.manager.has_result(self.session_id)