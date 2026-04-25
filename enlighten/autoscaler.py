"""
Autoscaler - 基于负载的自动缩放机制

功能:
- 负载监控: CPU、内存、请求队列、响应时间等指标
- 平滑扩缩容: 基于阈值的策略，防止抖动和震荡
- 多指标支持: 支持多种负载指标组合判断
- 可配置的冷却期: 缩放后防止立即反向缩放

架构:
LoadMonitor -> ScalingStrategy -> SmoothScalingController -> ScaleAction
"""

import time
import psutil
import threading
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from collections import deque
from enum import Enum
import statistics

logger = logging.getLogger(__name__)


class ScalingDirection(Enum):
    """缩放方向"""
    NONE = "none"
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"


@dataclass
class LoadMetrics:
    """负载指标数据"""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    request_queue_size: int = 0
    avg_response_time: float = 0.0
    requests_per_second: float = 0.0
    active_connections: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class ScalingAction:
    """缩放动作"""
    direction: ScalingDirection
    reason: str
    metrics: LoadMetrics
    target_replicas: Optional[int] = None


@dataclass
class ScalingConfig:
    """缩放配置"""
    min_replicas: int = 1
    max_replicas: int = 10

    cpu_threshold_up: float = 70.0
    cpu_threshold_down: float = 30.0

    memory_threshold_up: float = 75.0
    memory_threshold_down: float = 40.0

    queue_threshold_up: int = 10
    queue_threshold_down: int = 3

    response_time_threshold_up: float = 1.0
    response_time_threshold_down: float = 0.3

    scale_up_cool_down: float = 60.0
    scale_down_cool_down: float = 120.0

    stabilization_window: int = 5
    scale_up_stabilization_window: int = 3
    scale_down_stabilization_window: int = 5

    scale_factor: float = 1.0
    min_scale_step: int = 1
    max_scale_step: int = 3

    enable_smooth_scaling: bool = True
    predictive_scaling: bool = False

    def __post_init__(self):
        if self.min_replicas < 1:
            self.min_replicas = 1
        if self.max_replicas < self.min_replicas:
            self.max_replicas = self.min_replicas


class LoadMonitor:
    """
    负载监控器

    收集和跟踪系统负载指标:
    - CPU 使用率
    - 内存使用率
    - 请求队列大小
    - 平均响应时间
    - 请求速率
    - 活跃连接数
    """

    def __init__(
        self,
        window_size: int = 60,
        sample_interval: float = 1.0,
        queue: Optional[Any] = None
    ):
        self.window_size = window_size
        self.sample_interval = sample_interval
        self.queue = queue

        self._metrics_history: deque = deque(maxlen=window_size)
        self._response_times: deque = deque(maxlen=1000)
        self._request_timestamps: deque = deque(maxlen=1000)

        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        self._request_count = 0
        self._request_count_lock = threading.Lock()

    def start(self):
        """启动监控"""
        if self._running:
            return

        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Load monitor started")

    def stop(self):
        """停止监控"""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("Load monitor stopped")

    def _monitor_loop(self):
        """监控循环"""
        while self._running:
            try:
                metrics = self._collect_metrics()
                with self._lock:
                    self._metrics_history.append(metrics)
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")

            time.sleep(self.sample_interval)

    def _collect_metrics(self) -> LoadMetrics:
        """收集当前指标"""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent

        return LoadMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            request_queue_size=self._get_queue_size(),
            avg_response_time=self._compute_avg_response_time(),
            requests_per_second=self._compute_rps(),
            active_connections=self._get_active_connections(),
            timestamp=time.time()
        )

    def _get_queue_size(self) -> int:
        """获取请求队列大小（子类可重写）"""
        if self.queue and hasattr(self.queue, 'size'):
            try:
                return self.queue.size()
            except Exception:
                return 0
        return 0

    def _get_active_connections(self) -> int:
        """获取活跃连接数"""
        try:
            return len(psutil.net_connections())
        except Exception:
            return 0

    def _compute_avg_response_time(self) -> float:
        """计算平均响应时间"""
        with self._lock:
            if not self._response_times:
                return 0.0
            return statistics.mean(self._response_times)

    def _compute_rps(self) -> float:
        """计算每秒请求数"""
        current_time = time.time()
        with self._lock:
            cutoff_time = current_time - 1.0
            self._request_timestamps = deque(
                [ts for ts in self._request_timestamps if ts > cutoff_time],
                maxlen=1000
            )
            return len(self._request_timestamps)

    def record_request(self, response_time: float):
        """记录请求和响应时间"""
        with self._request_count_lock:
            self._request_count += 1
            current_time = time.time()
            self._request_timestamps.append(current_time)
            self._response_times.append(response_time)

    def record_load_metrics(self, metrics: LoadMetrics):
        """记录完整的负载指标"""
        with self._lock:
            self._metrics_history.append(metrics)

    def get_current_metrics(self) -> LoadMetrics:
        """获取当前指标"""
        with self._lock:
            if self._metrics_history:
                return self._metrics_history[-1]
            return self._collect_metrics()

    def get_average_metrics(self, last_n: Optional[int] = None) -> LoadMetrics:
        """获取平均指标"""
        with self._lock:
            if not self._metrics_history:
                return LoadMetrics()

            if last_n is None:
                metrics_list = list(self._metrics_history)
            else:
                metrics_list = list(self._metrics_history)[-last_n:]

            if not metrics_list:
                return LoadMetrics()

            return LoadMetrics(
                cpu_percent=statistics.mean(m.cpu_percent for m in metrics_list),
                memory_percent=statistics.mean(m.memory_percent for m in metrics_list),
                request_queue_size=int(statistics.mean(m.request_queue_size for m in metrics_list)),
                avg_response_time=statistics.mean(m.avg_response_time for m in metrics_list),
                requests_per_second=statistics.mean(m.requests_per_second for m in metrics_list),
                active_connections=int(statistics.mean(m.active_connections for m in metrics_list)),
                timestamp=time.time()
            )

    def get_metrics_trend(self, window: int = 5) -> Dict[str, float]:
        """获取指标趋势"""
        with self._lock:
            if len(self._metrics_history) < window:
                return {"cpu_trend": 0.0, "memory_trend": 0.0, "queue_trend": 0.0}

            recent = list(self._metrics_history)[-window:]
            if len(recent) < 2:
                return {"cpu_trend": 0.0, "memory_trend": 0.0, "queue_trend": 0.0}

            def compute_trend(values: List[float]) -> float:
                if len(values) < 2:
                    return 0.0
                n = len(values)
                x = list(range(n))
                mean_x = sum(x) / n
                mean_y = sum(values) / n
                numerator = sum((x[i] - mean_x) * (values[i] - mean_y) for i in range(n))
                denominator = sum((x[i] - mean_x) ** 2 for i in range(n))
                if denominator == 0:
                    return 0.0
                return numerator / denominator

            cpu_values = [m.cpu_percent for m in recent]
            memory_values = [m.memory_percent for m in recent]
            queue_values = [float(m.request_queue_size) for m in recent]

            return {
                "cpu_trend": compute_trend(cpu_values),
                "memory_trend": compute_trend(memory_values),
                "queue_trend": compute_trend(queue_values)
            }

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            if not self._metrics_history:
                return {
                    "total_requests": self._request_count,
                    "samples_collected": 0
                }

            cpu_values = [m.cpu_percent for m in self._metrics_history]
            memory_values = [m.memory_percent for m in self._metrics_history]

            return {
                "total_requests": self._request_count,
                "samples_collected": len(self._metrics_history),
                "cpu": {
                    "current": cpu_values[-1] if cpu_values else 0,
                    "avg": statistics.mean(cpu_values) if cpu_values else 0,
                    "max": max(cpu_values) if cpu_values else 0,
                    "min": min(cpu_values) if cpu_values else 0
                },
                "memory": {
                    "current": memory_values[-1] if memory_values else 0,
                    "avg": statistics.mean(memory_values) if memory_values else 0,
                    "max": max(memory_values) if memory_values else 0,
                    "min": min(memory_values) if memory_values else 0
                }
            }


class ScalingStrategy:
    """缩放策略基类"""

    def __init__(self, config: ScalingConfig):
        self.config = config

    def should_scale(
        self,
        current_metrics: LoadMetrics,
        avg_metrics: LoadMetrics,
        current_replicas: int
    ) -> ScalingAction:
        """判断是否应该缩放"""
        raise NotImplementedError


class ThresholdBasedStrategy(ScalingStrategy):
    """
    基于阈值的缩放策略

    使用多个指标的阈值组合判断是否需要缩放:
    - 扩容条件: 任意指标超过上限阈值
    - 缩容条件: 所有指标都低于下限阈值
    """

    def should_scale(
        self,
        current_metrics: LoadMetrics,
        avg_metrics: LoadMetrics,
        current_replicas: int
    ) -> ScalingAction:
        """判断是否应该缩放"""
        config = self.config

        should_up = self._should_scale_up(current_metrics, avg_metrics)
        should_down = self._should_scale_down(current_metrics, avg_metrics, current_replicas)

        if should_up and not should_down:
            return ScalingAction(
                direction=ScalingDirection.SCALE_UP,
                reason=self._get_scale_up_reason(current_metrics, avg_metrics),
                metrics=current_metrics,
                target_replicas=min(
                    current_replicas + self._compute_scale_step(avg_metrics, True),
                    config.max_replicas
                )
            )
        elif should_down and not should_up:
            return ScalingAction(
                direction=ScalingDirection.SCALE_DOWN,
                reason=self._get_scale_down_reason(current_metrics, avg_metrics),
                metrics=current_metrics,
                target_replicas=max(
                    current_replicas - self._compute_scale_step(avg_metrics, False),
                    config.min_replicas
                )
            )
        else:
            return ScalingAction(
                direction=ScalingDirection.NONE,
                reason="Metrics within acceptable range",
                metrics=current_metrics
            )

    def _should_scale_up(
        self,
        current_metrics: LoadMetrics,
        avg_metrics: LoadMetrics
    ) -> bool:
        """判断是否应该扩容"""
        config = self.config

        conditions = [
            avg_metrics.cpu_percent >= config.cpu_threshold_up,
            avg_metrics.memory_percent >= config.memory_threshold_up,
            avg_metrics.request_queue_size >= config.queue_threshold_up,
            avg_metrics.avg_response_time >= config.response_time_threshold_up
        ]

        return sum(conditions) >= 1

    def _should_scale_down(
        self,
        current_metrics: LoadMetrics,
        avg_metrics: LoadMetrics,
        current_replicas: int
    ) -> bool:
        """判断是否应该缩容"""
        config = self.config

        if current_replicas <= config.min_replicas:
            return False

        all_conditions = [
            avg_metrics.cpu_percent <= config.cpu_threshold_down,
            avg_metrics.memory_percent <= config.memory_threshold_down,
            avg_metrics.request_queue_size <= config.queue_threshold_down,
            avg_metrics.avg_response_time <= config.response_time_threshold_down
        ]

        return all(all_conditions)

    def _compute_scale_step(self, metrics: LoadMetrics, scaling_up: bool) -> int:
        """计算缩放步长"""
        config = self.config

        cpu_ratio = metrics.cpu_percent / 100.0
        memory_ratio = metrics.memory_percent / 100.0

        base_load = max(cpu_ratio, memory_ratio)

        if scaling_up:
            scale_factor = max(base_load * config.scale_factor, 1.0)
        else:
            scale_factor = max((1.0 - base_load) * config.scale_factor, 1.0)

        step = max(int(scale_factor), config.min_scale_step)
        return min(step, config.max_scale_step)

    def _get_scale_up_reason(self, current: LoadMetrics, avg: LoadMetrics) -> str:
        """获取扩容原因"""
        reasons = []
        if avg.cpu_percent >= self.config.cpu_threshold_up:
            reasons.append(f"CPU({avg.cpu_percent:.1f}%)")
        if avg.memory_percent >= self.config.memory_threshold_up:
            reasons.append(f"Memory({avg.memory_percent:.1f}%)")
        if avg.request_queue_size >= self.config.queue_threshold_up:
            reasons.append(f"Queue({avg.request_queue_size})")
        if avg.avg_response_time >= self.config.response_time_threshold_up:
            reasons.append(f"ResponseTime({avg.avg_response_time:.2f}s)")

        return f"Scale up: {', '.join(reasons)}" if reasons else "Scale up"

    def _get_scale_down_reason(self, current: LoadMetrics, avg: LoadMetrics) -> str:
        """获取缩容原因"""
        reasons = []
        if avg.cpu_percent <= self.config.cpu_threshold_down:
            reasons.append(f"CPU({avg.cpu_percent:.1f}%)")
        if avg.memory_percent <= self.config.memory_threshold_down:
            reasons.append(f"Memory({avg.memory_percent:.1f}%)")
        if avg.request_queue_size <= self.config.queue_threshold_down:
            reasons.append(f"Queue({avg.request_queue_size})")
        if avg.avg_response_time <= self.config.response_time_threshold_down:
            reasons.append(f"ResponseTime({avg.avg_response_time:.2f}s)")

        return f"Scale down: {', '.join(reasons)}" if reasons else "Scale down"


class SmoothScalingController:
    """
    平滑缩放控制器

    防止缩放抖动和不稳定:
    - 冷却期控制: 缩放后等待一段时间再进行下一次缩放
    - 稳定窗口: 基于历史数据判断是否稳定
    - 预测性缩放: 基于趋势预测提前缩放
    """

    def __init__(self, config: ScalingConfig):
        self.config = config

        self._last_scale_time: float = 0.0
        self._last_scale_direction: ScalingDirection = ScalingDirection.NONE
        self._scale_history: deque = deque(maxlen=config.stabilization_window)

        self._scale_up_cooldown_remaining: float = 0.0
        self._scale_down_cooldown_remaining: float = 0.0

    def update(self):
        """更新冷却计时器"""
        current_time = time.time()

        if self._scale_up_cooldown_remaining > 0:
            self._scale_up_cooldown_remaining -= (current_time - self._last_scale_time)
            self._scale_up_cooldown_remaining = max(0.0, self._scale_up_cooldown_remaining)

        if self._scale_down_cooldown_remaining > 0:
            self._scale_down_cooldown_remaining -= (current_time - self._last_scale_time)
            self._scale_down_cooldown_remaining = max(0.0, self._scale_down_cooldown_remaining)

        self._last_scale_time = current_time

    def can_scale(self, action: ScalingAction) -> bool:
        """检查是否可以执行缩放"""
        config = self.config

        if not config.enable_smooth_scaling:
            return True

        self.update()

        if action.direction == ScalingDirection.NONE:
            return False

        if action.direction == ScalingDirection.SCALE_UP:
            if self._scale_up_cooldown_remaining > 0:
                logger.debug(f"Scale up blocked: cooldown {self._scale_up_cooldown_remaining:.1f}s remaining")
                return False

            if self._last_scale_direction == ScalingDirection.SCALE_DOWN:
                if self._scale_down_cooldown_remaining > 0:
                    logger.debug("Scale up blocked: recent scale down, in stabilization period")
                    return False

        elif action.direction == ScalingDirection.SCALE_DOWN:
            if self._scale_down_cooldown_remaining > 0:
                logger.debug(f"Scale down blocked: cooldown {self._scale_down_cooldown_remaining:.1f}s remaining")
                return False

            if self._last_scale_direction == ScalingDirection.SCALE_UP:
                if self._scale_up_cooldown_remaining > 0:
                    logger.debug("Scale down blocked: recent scale up, in stabilization period")
                    return False

        if not self._is_stable(action.direction):
            logger.debug("Scaling blocked: metrics not stable enough")
            return False

        return True

    def _is_stable(self, direction: ScalingDirection) -> bool:
        """检查指标是否稳定"""
        config = self.config

        if len(self._scale_history) < config.stabilization_window:
            return True

        recent_actions = list(self._scale_history)[-config.stabilization_window:]

        direction_counts = {
            ScalingDirection.SCALE_UP: 0,
            ScalingDirection.SCALE_DOWN: 0,
            ScalingDirection.NONE: 0
        }

        for action in recent_actions:
            if isinstance(action, ScalingAction):
                direction_counts[action.direction] += 1
            else:
                direction_counts[action] += 1

        if direction == ScalingDirection.SCALE_UP:
            if direction_counts[ScalingDirection.SCALE_DOWN] >= config.scale_up_stabilization_window:
                return False
        elif direction == ScalingDirection.SCALE_DOWN:
            if direction_counts[ScalingDirection.SCALE_UP] >= config.scale_down_stabilization_window:
                return False

        return True

    def record_scale(self, action: ScalingAction):
        """记录缩放事件"""
        config = self.config

        self._last_scale_time = time.time()
        self._last_scale_direction = action.direction

        if action.direction == ScalingDirection.SCALE_UP:
            self._scale_up_cooldown_remaining = config.scale_up_cool_down
        elif action.direction == ScalingDirection.SCALE_DOWN:
            self._scale_down_cooldown_remaining = config.scale_down_cool_down

        self._scale_history.append(action.direction)

        logger.info(f"Scale recorded: {action.direction.value} - {action.reason}")

    def get_status(self) -> Dict[str, Any]:
        """获取控制器状态"""
        return {
            "last_direction": self._last_scale_direction.value,
            "last_scale_time": self._last_scale_time,
            "scale_up_cooldown_remaining": self._scale_up_cooldown_remaining,
            "scale_down_cooldown_remaining": self._scale_down_cooldown_remaining,
            "history_length": len(self._scale_history)
        }


class Autoscaler:
    """
    自动缩放器主类

    协调负载监控、缩放策略和平滑控制:
    - 定期检查负载指标
    - 基于策略决策是否缩放
    - 通过平滑控制器确保稳定
    - 调用回调函数执行缩放
    """

    def __init__(
        self,
        config: Optional[ScalingConfig] = None,
        strategy: Optional[ScalingStrategy] = None,
        scale_callback: Optional[Callable[[int], None]] = None,
        queue: Optional[Any] = None
    ):
        self.config = config or ScalingConfig()
        self.strategy = strategy or ThresholdBasedStrategy(self.config)
        self.scale_callback = scale_callback

        self.load_monitor = LoadMonitor(queue=queue)
        self.smooth_controller = SmoothScalingController(self.config)

        self._current_replicas: int = self.config.min_replicas
        self._target_replicas: int = self.config.min_replicas

        self._running = False
        self._autoscaler_thread: Optional[threading.Thread] = None
        self._check_interval: float = 10.0

        self._scaling_actions: deque = deque(maxlen=100)
        self._lock = threading.Lock()

    def start(self, check_interval: float = 10.0):
        """启动自动缩放"""
        if self._running:
            return

        self._running = True
        self._check_interval = check_interval

        self.load_monitor.start()

        self._autoscaler_thread = threading.Thread(
            target=self._scaling_loop,
            daemon=True
        )
        self._autoscaler_thread.start()

        logger.info(f"Autoscaler started with check interval {check_interval}s")

    def stop(self):
        """停止自动缩放"""
        self._running = False
        self.load_monitor.stop()

        if self._autoscaler_thread:
            self._autoscaler_thread.join(timeout=15)

        logger.info("Autoscaler stopped")

    def _scaling_loop(self):
        """缩放决策循环"""
        while self._running:
            try:
                self._check_and_scale()
            except Exception as e:
                logger.error(f"Error in scaling loop: {e}")

            time.sleep(self._check_interval)

    def _check_and_scale(self):
        """检查并执行缩放"""
        current_metrics = self.load_monitor.get_current_metrics()
        avg_metrics = self.load_monitor.get_average_metrics(last_n=10)

        action = self.strategy.should_scale(
            current_metrics=current_metrics,
            avg_metrics=avg_metrics,
            current_replicas=self._current_replicas
        )

        with self._lock:
            self._scaling_actions.append(action)

        if action.direction == ScalingDirection.NONE:
            return

        if self.smooth_controller.can_scale(action):
            self._execute_scale(action)

    def _execute_scale(self, action: ScalingAction):
        """执行缩放"""
        if action.target_replicas is None:
            return

        if action.target_replicas == self._current_replicas:
            return

        old_replicas = self._current_replicas
        self._current_replicas = action.target_replicas
        self._target_replicas = action.target_replicas

        self.smooth_controller.record_scale(action)

        logger.info(
            f"Scaling {old_replicas} -> {self._current_replicas}: {action.reason}"
        )

        if self.scale_callback:
            try:
                self.scale_callback(self._current_replicas)
            except Exception as e:
                logger.error(f"Error in scale callback: {e}")

    def record_request(self, response_time: float):
        """记录请求以更新指标"""
        self.load_monitor.record_request(response_time)

    def record_load_metrics(self, metrics: LoadMetrics):
        """记录完整的负载指标"""
        self.load_monitor.record_load_metrics(metrics)

    def set_replicas(self, replicas: int):
        """手动设置副本数"""
        with self._lock:
            self._current_replicas = max(
                self.config.min_replicas,
                min(replicas, self.config.max_replicas)
            )
            self._target_replicas = self._current_replicas
        logger.info(f"Replicas manually set to {self._current_replicas}")

    def get_current_replicas(self) -> int:
        """获取当前副本数"""
        return self._current_replicas

    def get_status(self) -> Dict[str, Any]:
        """获取自动缩放器状态"""
        return {
            "running": self._running,
            "current_replicas": self._current_replicas,
            "target_replicas": self._target_replicas,
            "config": {
                "min_replicas": self.config.min_replicas,
                "max_replicas": self.config.max_replicas,
                "enable_smooth_scaling": self.config.enable_smooth_scaling,
                "scale_up_cool_down": self.config.scale_up_cool_down,
                "scale_down_cool_down": self.config.scale_down_cool_down
            },
            "load_monitor": self.load_monitor.get_statistics(),
            "smooth_controller": self.smooth_controller.get_status(),
            "recent_actions": [
                {
                    "direction": a.direction.value,
                    "reason": a.reason,
                    "target_replicas": a.target_replicas
                }
                for a in list(self._scaling_actions)[-5:]
            ]
        }

    def get_load_metrics(self) -> Dict[str, Any]:
        """获取负载指标"""
        return {
            "current": self.load_monitor.get_current_metrics().__dict__,
            "average": self.load_monitor.get_average_metrics().__dict__,
            "trend": self.load_monitor.get_metrics_trend()
        }

    def simulate_load(
        self,
        cpu_percent: float,
        memory_percent: float,
        queue_size: int,
        response_time: float
    ):
        """模拟负载（用于测试）"""
        metrics = LoadMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            request_queue_size=queue_size,
            avg_response_time=response_time,
            requests_per_second=0.0,
            active_connections=0,
            timestamp=time.time()
        )

        with self.load_monitor._lock:
            self.load_monitor._metrics_history.append(metrics)

        self.record_request(response_time)


class RequestQueue:
    """
    请求队列（用于跟踪挂起的请求）

    在API服务器中使用，跟踪请求队列大小
    """

    def __init__(self, max_size: int = 1000):
        self._queue: deque = deque(maxlen=max_size)
        self._lock = threading.Lock()

    def put(self, request_id: str) -> bool:
        """添加请求到队列"""
        with self._lock:
            if len(self._queue) >= self._queue.maxlen:
                return False
            self._queue.append({
                "request_id": request_id,
                "start_time": time.time()
            })
            return True

    def get(self, request_id: str) -> Optional[float]:
        """移除请求并返回等待时间"""
        with self._lock:
            for i, req in enumerate(self._queue):
                if req["request_id"] == request_id:
                    self._queue.remove(req)
                    return time.time() - req["start_time"]
            return None

    def size(self) -> int:
        """获取队列大小"""
        with self._lock:
            return len(self._queue)

    def get_waiting_requests(self) -> int:
        """获取等待中的请求数"""
        return self.size()
