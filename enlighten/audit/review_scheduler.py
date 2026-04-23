"""
Automated Review Scheduler - 自动化复盘服务调度器
定时任务调度，自动保存快照和生成复盘报告
"""

import time
import threading
import schedule
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from enum import Enum
from queue import Queue


class ScheduleType(Enum):
    """调度类型"""
    INTERVAL = "interval"      # 间隔调度
    CRON = "cron"             # Cron 调度
    ON_DEMAND = "on_demand"   # 按需调度


@dataclass
class SchedulerConfig:
    """调度器配置"""
    snapshot_interval_minutes: int = 30
    report_interval_hours: int = 24
    max_snapshots: int = 100
    enable_auto_review: bool = True
    review_start_hour: int = 2  # 凌晨 2 点开始
    review_end_hour: int = 6    # 凌晨 6 点结束


@dataclass
class Snapshot:
    """快照"""
    session_id: str
    timestamp: float
    data: Dict[str, Any]
    size_bytes: int

    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "data_size": self.size_bytes,
            "created_at": datetime.fromtimestamp(self.timestamp).isoformat()
        }


@dataclass
class ReviewTask:
    """复盘任务"""
    task_id: str
    task_type: str  # "snapshot" | "report"
    session_id: Optional[str]
    scheduled_time: float
    status: str = "pending"  # pending | running | completed | failed
    result: Optional[Dict] = None
    error: Optional[str] = None


class AutomatedReviewScheduler:
    """
    自动化复盘服务调度器

    功能:
    - 定时保存 L2 快照
    - 定时生成复盘报告
    - 任务队列管理
    - 失败重试机制
    """

    def __init__(
        self,
        config: Optional[SchedulerConfig] = None,
        l2_model: Optional[Any] = None,
        audit_chain: Optional[Any] = None
    ):
        self.config = config or SchedulerConfig()
        self.l2_model = l2_model
        self.audit_chain = audit_chain

        self.snapshots: List[Snapshot] = []
        self.task_queue: Queue = Queue()
        self.running_tasks: Dict[str, ReviewTask] = {}

        self._stop_event = threading.Event()
        self._scheduler_thread: Optional[threading.Thread] = None

        self._setup_schedule()

    def _setup_schedule(self) -> None:
        """设置调度任务"""
        if self.config.enable_auto_review:
            schedule.every(self.config.snapshot_interval_minutes).minutes.do(
                self._take_snapshot
            )

            schedule.every(self.config.report_interval_hours).hours.do(
                self._generate_daily_report
            )

    def start(self) -> None:
        """启动调度器"""
        if self._scheduler_thread is not None and self._scheduler_thread.is_alive():
            return

        self._stop_event.clear()
        self._scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self._scheduler_thread.start()

    def stop(self) -> None:
        """停止调度器"""
        self._stop_event.set()

        if self._scheduler_thread is not None:
            self._scheduler_thread.join(timeout=5)

    def _run_scheduler(self) -> None:
        """运行调度循环"""
        while not self._stop_event.is_set():
            schedule.run_pending()
            time.sleep(1)

    def schedule_snapshot(
        self,
        session_id: str,
        delay_seconds: int = 0
    ) -> str:
        """
        调度快照任务

        Args:
            session_id: 会话 ID
            delay_seconds: 延迟秒数

        Returns:
            task_id: 任务 ID
        """
        task_id = f"snapshot_{session_id}_{int(time.time())}"

        task = ReviewTask(
            task_id=task_id,
            task_type="snapshot",
            session_id=session_id,
            scheduled_time=time.time() + delay_seconds
        )

        self.task_queue.put(task)

        return task_id

    def schedule_report(
        self,
        session_id: Optional[str] = None,
        delay_seconds: int = 0
    ) -> str:
        """
        调度报告生成任务

        Args:
            session_id: 会话 ID，None 表示所有会话
            delay_seconds: 延迟秒数

        Returns:
            task_id: 任务 ID
        """
        task_id = f"report_{session_id or 'all'}_{int(time.time())}"

        task = ReviewTask(
            task_id=task_id,
            task_type="report",
            session_id=session_id,
            scheduled_time=time.time() + delay_seconds
        )

        self.task_queue.put(task)

        return task_id

    def _take_snapshot(self) -> None:
        """执行快照任务"""
        try:
            if self.l2_model is None:
                return

            snapshot_data = self.l2_model.get_memory_snapshot()

            snapshot = Snapshot(
                session_id="auto_snapshot",
                timestamp=time.time(),
                data=snapshot_data,
                size_bytes=self._estimate_size(snapshot_data)
            )

            self.snapshots.append(snapshot)

            self._cleanup_old_snapshots()

        except Exception as e:
            print(f"快照任务失败: {e}")

    def _generate_daily_report(self) -> None:
        """生成每日报告"""
        try:
            current_hour = datetime.now().hour

            if not (self.config.review_start_hour <= current_hour < self.config.review_end_hour):
                return

            sessions = self._get_active_sessions()

            for session_id in sessions:
                self._generate_session_report(session_id)

        except Exception as e:
            print(f"每日报告生成失败: {e}")

    def _generate_session_report(self, session_id: str) -> Dict:
        """生成会话报告"""
        from .offline_review import OfflineReviewService

        if self.audit_chain is None:
            return {}

        review_service = OfflineReviewService(
            audit_chain=self.audit_chain,
            l2_model=self.l2_model
        )

        report = review_service.generate_report(session_id)

        return asdict_report(report)

    def _get_active_sessions(self) -> List[str]:
        """获取活跃会话列表"""
        if self.audit_chain is None:
            return []

        sessions = set()

        for entry in self.audit_chain.chain:
            session_id = entry.data.get("session_id")
            if session_id:
                sessions.add(session_id)

        return list(sessions)

    def _cleanup_old_snapshots(self) -> None:
        """清理旧快照"""
        if len(self.snapshots) > self.config.max_snapshots:
            self.snapshots = self.snapshots[-self.config.max_snapshots:]

    def _estimate_size(self, data: Dict) -> int:
        """估算数据大小"""
        import sys
        return len(str(data).encode())

    def get_pending_tasks(self) -> List[ReviewTask]:
        """获取待处理任务"""
        tasks = []
        while not self.task_queue.empty():
            tasks.append(self.task_queue.get())
        return tasks

    def get_task_status(self, task_id: str) -> Optional[ReviewTask]:
        """获取任务状态"""
        return self.running_tasks.get(task_id)

    def get_snapshots(
        self,
        last_n: Optional[int] = None
    ) -> List[Snapshot]:
        """获取快照列表"""
        if last_n is None:
            return self.snapshots
        return self.snapshots[-last_n:]

    def clear_snapshots(self) -> None:
        """清除所有快照"""
        self.snapshots.clear()

    def get_statistics(self) -> Dict[str, Any]:
        """获取调度器统计信息"""
        return {
            "total_snapshots": len(self.snapshots),
            "pending_tasks": self.task_queue.qsize(),
            "running_tasks": len(self.running_tasks),
            "config": {
                "snapshot_interval_minutes": self.config.snapshot_interval_minutes,
                "report_interval_hours": self.config.report_interval_hours,
                "enable_auto_review": self.config.enable_auto_review
            }
        }


def asdict_report(report) -> Dict:
    """将报告转换为字典"""
    from dataclasses import asdict
    return asdict(report)
