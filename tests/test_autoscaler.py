"""
Test Autoscaler - 基于负载的自动缩放机制测试

测试内容:
1. 负载监控器功能测试
2. 缩放策略测试
3. 平滑缩放控制器测试
4. 自动缩放器集成测试
5. 不同负载场景下的缩放效果测试
"""

import pytest
import time
import threading
from collections import deque

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enlighten.autoscaler import (
    Autoscaler,
    ScalingConfig,
    ThresholdBasedStrategy,
    SmoothScalingController,
    LoadMonitor,
    LoadMetrics,
    ScalingAction,
    ScalingDirection,
    RequestQueue
)


class TestLoadMetrics:
    """测试负载指标数据"""

    def test_metrics_creation(self):
        """测试负载指标创建"""
        metrics = LoadMetrics(
            cpu_percent=50.0,
            memory_percent=60.0,
            request_queue_size=5,
            avg_response_time=0.5,
            requests_per_second=10.0,
            active_connections=20,
            timestamp=time.time()
        )

        assert metrics.cpu_percent == 50.0
        assert metrics.memory_percent == 60.0
        assert metrics.request_queue_size == 5
        assert metrics.avg_response_time == 0.5
        assert metrics.requests_per_second == 10.0
        assert metrics.active_connections == 20

    def test_metrics_defaults(self):
        """测试负载指标默认值"""
        metrics = LoadMetrics()

        assert metrics.cpu_percent == 0.0
        assert metrics.memory_percent == 0.0
        assert metrics.request_queue_size == 0
        assert metrics.avg_response_time == 0.0
        assert metrics.requests_per_second == 0.0
        assert metrics.active_connections == 0


class TestScalingConfig:
    """测试缩放配置"""

    def test_config_defaults(self):
        """测试默认配置"""
        config = ScalingConfig()

        assert config.min_replicas == 1
        assert config.max_replicas == 10
        assert config.cpu_threshold_up == 70.0
        assert config.cpu_threshold_down == 30.0
        assert config.scale_up_cool_down == 60.0
        assert config.scale_down_cool_down == 120.0
        assert config.enable_smooth_scaling is True

    def test_config_custom_values(self):
        """测试自定义配置"""
        config = ScalingConfig(
            min_replicas=2,
            max_replicas=20,
            cpu_threshold_up=80.0,
            cpu_threshold_down=20.0,
            scale_up_cool_down=30.0,
            scale_down_cool_down=60.0
        )

        assert config.min_replicas == 2
        assert config.max_replicas == 20
        assert config.cpu_threshold_up == 80.0
        assert config.cpu_threshold_down == 20.0
        assert config.scale_up_cool_down == 30.0
        assert config.scale_down_cool_down == 60.0

    def test_config_bounds_enforcement(self):
        """测试配置边界强制"""
        config = ScalingConfig(min_replicas=0, max_replicas=5)

        assert config.min_replicas == 1

    def test_config_max_replicas_enforcement(self):
        """测试最大副本数边界强制"""
        config = ScalingConfig(min_replicas=10, max_replicas=5)

        assert config.max_replicas >= config.min_replicas


class TestLoadMonitor:
    """测试负载监控器"""

    def test_monitor_initialization(self):
        """测试监控器初始化"""
        monitor = LoadMonitor(window_size=10, sample_interval=0.1)

        assert monitor.window_size == 10
        assert monitor.sample_interval == 0.1
        assert len(monitor._metrics_history) == 0

    def test_record_request(self):
        """测试记录请求"""
        monitor = LoadMonitor()
        monitor.record_request(0.5)
        monitor.record_request(0.3)
        monitor.record_request(0.7)

        assert len(monitor._response_times) == 3
        assert len(monitor._request_timestamps) == 3

    def test_get_current_metrics(self):
        """测试获取当前指标"""
        monitor = LoadMonitor()

        metrics = monitor.get_current_metrics()

        assert isinstance(metrics, LoadMetrics)
        assert hasattr(metrics, 'cpu_percent')
        assert hasattr(metrics, 'memory_percent')

    def test_simulate_metrics(self):
        """测试模拟负载"""
        monitor = LoadMonitor()

        for _ in range(5):
            monitor.simulate_load(
                cpu_percent=50.0,
                memory_percent=60.0,
                queue_size=5,
                response_time=0.5
            )

        assert len(monitor._metrics_history) == 5

        avg = monitor.get_average_metrics()
        assert avg.cpu_percent == 50.0
        assert avg.memory_percent == 60.0

    def test_metrics_trend(self):
        """测试指标趋势计算"""
        monitor = LoadMonitor()

        for i in range(10):
            monitor.simulate_load(
                cpu_percent=30.0 + i * 5,
                memory_percent=40.0 + i * 3,
                queue_size=i,
                response_time=0.1 * i
            )

        trend = monitor.get_metrics_trend(window=5)

        assert 'cpu_trend' in trend
        assert 'memory_trend' in trend
        assert 'queue_trend' in trend

    def test_monitor_start_stop(self):
        """测试监控器启动和停止"""
        monitor = LoadMonitor(sample_interval=0.05)

        monitor.start()
        time.sleep(0.2)
        monitor.stop()

        assert not monitor._running


class TestThresholdBasedStrategy:
    """测试基于阈值的缩放策略"""

    def test_scale_up_cpu_threshold(self):
        """测试CPU扩容阈值"""
        config = ScalingConfig(
            cpu_threshold_up=70.0,
            cpu_threshold_down=30.0
        )
        strategy = ThresholdBasedStrategy(config)

        high_cpu_metrics = LoadMetrics(
            cpu_percent=80.0,
            memory_percent=50.0,
            request_queue_size=0,
            avg_response_time=0.2
        )

        avg_metrics = LoadMetrics(
            cpu_percent=75.0,
            memory_percent=50.0,
            request_queue_size=0,
            avg_response_time=0.2
        )

        action = strategy.should_scale(high_cpu_metrics, avg_metrics, current_replicas=3)

        assert action.direction == ScalingDirection.SCALE_UP
        assert action.target_replicas > 3

    def test_scale_down_low_metrics(self):
        """测试低负载缩容"""
        config = ScalingConfig(
            cpu_threshold_up=70.0,
            cpu_threshold_down=30.0,
            memory_threshold_up=75.0,
            memory_threshold_down=40.0,
            queue_threshold_up=10,
            queue_threshold_down=3
        )
        strategy = ThresholdBasedStrategy(config)

        low_metrics = LoadMetrics(
            cpu_percent=20.0,
            memory_percent=30.0,
            request_queue_size=1,
            avg_response_time=0.1
        )

        avg_metrics = LoadMetrics(
            cpu_percent=25.0,
            memory_percent=35.0,
            request_queue_size=2,
            avg_response_time=0.1
        )

        action = strategy.should_scale(low_metrics, avg_metrics, current_replicas=5)

        assert action.direction == ScalingDirection.SCALE_DOWN

    def test_no_scale_within_thresholds(self):
        """测试阈值内不缩放"""
        config = ScalingConfig(
            cpu_threshold_up=70.0,
            cpu_threshold_down=30.0,
            memory_threshold_up=75.0,
            memory_threshold_down=40.0
        )
        strategy = ThresholdBasedStrategy(config)

        mid_metrics = LoadMetrics(
            cpu_percent=50.0,
            memory_percent=50.0,
            request_queue_size=5,
            avg_response_time=0.5
        )

        avg_metrics = mid_metrics

        action = strategy.should_scale(mid_metrics, avg_metrics, current_replicas=3)

        assert action.direction == ScalingDirection.NONE

    def test_scale_up_queue_threshold(self):
        """测试队列扩容阈值"""
        config = ScalingConfig(queue_threshold_up=10)
        strategy = ThresholdBasedStrategy(config)

        high_queue_metrics = LoadMetrics(
            cpu_percent=50.0,
            memory_percent=50.0,
            request_queue_size=15,
            avg_response_time=0.5
        )

        avg_metrics = high_queue_metrics

        action = strategy.should_scale(high_queue_metrics, avg_metrics, current_replicas=3)

        assert action.direction == ScalingDirection.SCALE_UP


class TestSmoothScalingController:
    """测试平滑缩放控制器"""

    def test_controller_initialization(self):
        """测试控制器初始化"""
        config = ScalingConfig(
            scale_up_cool_down=60.0,
            scale_down_cool_down=120.0
        )
        controller = SmoothScalingController(config)

        assert controller.config == config

    def test_cooldown_blocks_scale_up(self):
        """测试冷却期阻止扩容"""
        config = ScalingConfig(
            scale_up_cool_down=10.0,
            enable_smooth_scaling=True
        )
        controller = SmoothScalingController(config)

        controller._last_scale_time = time.time() - 1.0
        controller._scale_up_cooldown_remaining = 10.0

        action = ScalingAction(
            direction=ScalingDirection.SCALE_UP,
            reason="Test",
            metrics=LoadMetrics()
        )

        can_scale = controller.can_scale(action)

        assert can_scale is False

    def test_cooldown_allows_after_expiry(self):
        """测试冷却期后允许缩放"""
        config = ScalingConfig(
            scale_up_cool_down=5.0,
            enable_smooth_scaling=True
        )
        controller = SmoothScalingController(config)

        controller._last_scale_time = time.time() - 10.0
        controller._scale_up_cooldown_remaining = 0.0

        action = ScalingAction(
            direction=ScalingDirection.SCALE_UP,
            reason="Test",
            metrics=LoadMetrics()
        )

        can_scale = controller.can_scale(action)

        assert can_scale is True

    def test_record_scale_updates_cooldown(self):
        """测试记录缩放更新冷却时间"""
        config = ScalingConfig(
            scale_up_cool_down=60.0,
            scale_down_cool_down=120.0
        )
        controller = SmoothScalingController(config)

        action = ScalingAction(
            direction=ScalingDirection.SCALE_UP,
            reason="Test",
            metrics=LoadMetrics()
        )

        controller.record_scale(action)

        assert controller._last_scale_direction == ScalingDirection.SCALE_UP
        assert controller._scale_up_cooldown_remaining == 60.0

    def test_smooth_scaling_disabled(self):
        """测试平滑缩放禁用"""
        config = ScalingConfig(enable_smooth_scaling=False)
        controller = SmoothScalingController(config)

        controller._scale_up_cooldown_remaining = 1000.0

        action = ScalingAction(
            direction=ScalingDirection.SCALE_UP,
            reason="Test",
            metrics=LoadMetrics()
        )

        can_scale = controller.can_scale(action)

        assert can_scale is True

    def test_get_status(self):
        """测试获取控制器状态"""
        config = ScalingConfig()
        controller = SmoothScalingController(config)

        status = controller.get_status()

        assert 'last_direction' in status
        assert 'scale_up_cooldown_remaining' in status
        assert 'scale_down_cooldown_remaining' in status


class TestAutoscaler:
    """测试自动缩放器"""

    def test_autoscaler_initialization(self):
        """测试自动缩放器初始化"""
        config = ScalingConfig()
        autoscaler = Autoscaler(config=config)

        assert autoscaler.config == config
        assert autoscaler._current_replicas == config.min_replicas
        assert autoscaler.load_monitor is not None
        assert autoscaler.smooth_controller is not None

    def test_set_replicas(self):
        """测试设置副本数"""
        config = ScalingConfig(min_replicas=1, max_replicas=10)
        autoscaler = Autoscaler(config=config)

        autoscaler.set_replicas(5)

        assert autoscaler.get_current_replicas() == 5

    def test_set_replicas_enforces_bounds(self):
        """测试设置副本数强制边界"""
        config = ScalingConfig(min_replicas=2, max_replicas=8)
        autoscaler = Autoscaler(config=config)

        autoscaler.set_replicas(100)
        assert autoscaler.get_current_replicas() == 8

        autoscaler.set_replicas(0)
        assert autoscaler.get_current_replicas() == 2

    def test_record_request(self):
        """测试记录请求"""
        config = ScalingConfig()
        autoscaler = Autoscaler(config=config)

        autoscaler.record_request(0.5)
        autoscaler.record_request(0.3)

        assert len(autoscaler.load_monitor._response_times) == 2

    def test_simulate_load(self):
        """测试模拟负载"""
        config = ScalingConfig()
        autoscaler = Autoscaler(config=config)

        autoscaler.simulate_load(
            cpu_percent=80.0,
            memory_percent=70.0,
            queue_size=15,
            response_time=1.5
        )

        assert len(autoscaler.load_monitor._metrics_history) == 1

    def test_get_status(self):
        """测试获取自动缩放器状态"""
        config = ScalingConfig()
        autoscaler = Autoscaler(config=config)

        status = autoscaler.get_status()

        assert 'running' in status
        assert 'current_replicas' in status
        assert 'target_replicas' in status
        assert 'config' in status
        assert 'load_monitor' in status

    def test_get_load_metrics(self):
        """测试获取负载指标"""
        config = ScalingConfig()
        autoscaler = Autoscaler(config=config)

        autoscaler.simulate_load(50.0, 60.0, 5, 0.5)

        metrics = autoscaler.get_load_metrics()

        assert 'current' in metrics
        assert 'average' in metrics
        assert 'trend' in metrics

    def test_scale_callback(self):
        """测试缩放回调"""
        config = ScalingConfig()
        callback_results = []

        def scale_callback(replicas):
            callback_results.append(replicas)

        autoscaler = Autoscaler(
            config=config,
            scale_callback=scale_callback
        )

        autoscaler._execute_scale(ScalingAction(
            direction=ScalingDirection.SCALE_UP,
            reason="Test",
            metrics=LoadMetrics(),
            target_replicas=3
        ))

        assert len(callback_results) == 1
        assert callback_results[0] == 3


class TestRequestQueue:
    """测试请求队列"""

    def test_queue_initialization(self):
        """测试队列初始化"""
        queue = RequestQueue(max_size=100)

        assert queue.size() == 0

    def test_put_and_get(self):
        """测试添加和移除请求"""
        queue = RequestQueue()

        request_id = "test-request-123"
        queue.put(request_id)

        assert queue.size() == 1

        wait_time = queue.get(request_id)

        assert wait_time is not None
        assert wait_time >= 0
        assert queue.size() == 0

    def test_get_nonexistent(self):
        """测试获取不存在的请求"""
        queue = RequestQueue()

        wait_time = queue.get("nonexistent")

        assert wait_time is None

    def test_queue_full(self):
        """测试队列满"""
        queue = RequestQueue(max_size=3)

        queue.put("req1")
        queue.put("req2")
        queue.put("req3")

        result = queue.put("req4")

        assert result is False
        assert queue.size() == 3


class TestScalingUnderDifferentLoads:
    """测试不同负载下的缩放效果"""

    def test_gradual_load_increase(self):
        """测试渐进负载增加"""
        config = ScalingConfig(
            cpu_threshold_up=50.0,
            cpu_threshold_down=30.0,
            enable_smooth_scaling=True,
            scale_up_cool_down=1.0
        )
        autoscaler = Autoscaler(config=config)

        replicas_history = []

        for i in range(20):
            cpu = 20.0 + i * 5
            autoscaler.simulate_load(cpu, 30.0, 2, 0.2)

            if autoscaler._current_replicas != replicas_history[-1] if replicas_history else True:
                replicas_history.append(autoscaler._current_replicas)

            time.sleep(0.05)

        print(f"Replicas history: {replicas_history}")
        assert len(replicas_history) >= 1

    def test_sudden_high_load(self):
        """测试突然高负载"""
        config = ScalingConfig(
            cpu_threshold_up=70.0,
            cpu_threshold_down=30.0,
            enable_smooth_scaling=True
        )
        autoscaler = Autoscaler(config=config)

        autoscaler.simulate_load(85.0, 80.0, 20, 2.0)

        current = autoscaler.load_monitor.get_current_metrics()
        avg = autoscaler.load_monitor.get_average_metrics()

        assert current.cpu_percent == 85.0
        assert avg.cpu_percent == 85.0

    def test_load_spike_recovery(self):
        """测试负载突增后恢复"""
        config = ScalingConfig(
            cpu_threshold_up=60.0,
            cpu_threshold_down=30.0,
            enable_smooth_scaling=True,
            scale_up_cool_down=2.0,
            scale_down_cool_down=2.0
        )
        autoscaler = Autoscaler(config=config)

        for i in range(10):
            autoscaler.simulate_load(80.0, 70.0, 15, 1.0)
            time.sleep(0.05)

        for i in range(10):
            autoscaler.simulate_load(20.0, 30.0, 1, 0.1)
            time.sleep(0.05)

        avg = autoscaler.load_monitor.get_average_metrics(last_n=5)
        assert avg.cpu_percent < 30.0

    def test_scale_up_stabilization(self):
        """测试扩容稳定性"""
        config = ScalingConfig(
            cpu_threshold_up=70.0,
            cpu_threshold_down=30.0,
            enable_smooth_scaling=True,
            scale_up_stabilization_window=3
        )
        autoscaler = Autoscaler(config=config)

        autoscaler.set_replicas(5)

        for _ in range(10):
            autoscaler.simulate_load(80.0, 70.0, 15, 1.0)
            time.sleep(0.05)

        assert autoscaler._current_replicas == 5

    def test_min_max_replicas_bounds(self):
        """测试最小/最大副本数边界"""
        config = ScalingConfig(
            min_replicas=2,
            max_replicas=5,
            cpu_threshold_up=50.0,
            cpu_threshold_down=30.0
        )
        autoscaler = Autoscaler(config=config)

        autoscaler.set_replicas(10)
        assert autoscaler.get_current_replicas() == 5

        autoscaler.set_replicas(1)
        assert autoscaler.get_current_replicas() == 2

    def test_concurrent_scaling_decisions(self):
        """测试并发缩放决策"""
        config = ScalingConfig(
            cpu_threshold_up=50.0,
            cpu_threshold_down=30.0,
            enable_smooth_scaling=False
        )
        autoscaler = Autoscaler(config=config)

        def simulate_high_load():
            for _ in range(10):
                autoscaler.simulate_load(80.0, 70.0, 15, 1.0)
                time.sleep(0.02)

        def simulate_low_load():
            for _ in range(10):
                autoscaler.simulate_load(20.0, 30.0, 1, 0.1)
                time.sleep(0.02)

        thread1 = threading.Thread(target=simulate_high_load)
        thread2 = threading.Thread(target=simulate_low_load)

        thread1.start()
        thread2.start()

        thread1.join()
        thread2.join()

        assert len(autoscaler.load_monitor._metrics_history) >= 10


class TestAutoscalerStressScenario:
    """压力场景测试"""

    def test_rapid_load_changes(self):
        """测试快速负载变化"""
        config = ScalingConfig(
            cpu_threshold_up=60.0,
            cpu_threshold_down=40.0,
            enable_smooth_scaling=True,
            scale_up_cool_down=1.0,
            scale_down_cool_down=1.0
        )
        autoscaler = Autoscaler(config=config)

        for cycle in range(5):
            for i in range(10):
                autoscaler.simulate_load(80.0, 70.0, 15, 1.0)
                time.sleep(0.01)

            for i in range(10):
                autoscaler.simulate_load(20.0, 30.0, 1, 0.1)
                time.sleep(0.01)

        assert len(autoscaler.load_monitor._metrics_history) >= 50

    def test_sustained_high_load(self):
        """测试持续高负载"""
        config = ScalingConfig(
            cpu_threshold_up=70.0,
            cpu_threshold_down=30.0,
            enable_smooth_scaling=True
        )
        autoscaler = Autoscaler(config=config)

        for i in range(30):
            autoscaler.simulate_load(
                cpu_percent=75.0 + (i % 10),
                memory_percent=70.0 + (i % 5),
                queue_size=12 + (i % 5),
                response_time=0.8 + (i % 3) * 0.1
            )
            time.sleep(0.02)

        avg = autoscaler.load_monitor.get_average_metrics(last_n=10)
        assert avg.cpu_percent >= 70.0

    def test_zero_load_scenario(self):
        """测试零负载场景"""
        config = ScalingConfig(
            cpu_threshold_up=70.0,
            cpu_threshold_down=30.0
        )
        autoscaler = Autoscaler(config=config)

        for _ in range(10):
            autoscaler.simulate_load(5.0, 20.0, 0, 0.05)
            time.sleep(0.02)

        avg = autoscaler.load_monitor.get_average_metrics()
        assert avg.cpu_percent < 30.0


def run_all_tests():
    """运行所有测试"""
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_all_tests()
