"""
Performance Benchmark - 性能基准测试套件
验证 EnlightenLM 的 +5%~+15% 性能开销目标
"""

import time
import torch
import psutil
import os
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from enum import Enum
import json


class BenchmarkType(Enum):
    """基准测试类型"""
    LATENCY = "latency"           # 延迟测试
    MEMORY = "memory"             # 内存测试
    THROUGHPUT = "throughput"     # 吞吐量测试
    OVERHEAD = "overhead"         # 开销测试


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    test_name: str
    benchmark_type: BenchmarkType
    value: float
    unit: str
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "test_name": self.test_name,
            "benchmark_type": self.benchmark_type.value,
            "value": self.value,
            "unit": self.unit,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }


@dataclass
class PerformanceMetrics:
    """性能指标"""
    latency_ms: float = 0.0
    memory_gb: float = 0.0
    throughput_tokens_per_sec: float = 0.0
    overhead_percent: float = 0.0


class LatencyBenchmark:
    """延迟基准测试"""

    def __init__(self, warmup_iterations: int = 3, test_iterations: int = 10):
        self.warmup_iterations = warmup_iterations
        self.test_iterations = test_iterations

    def benchmark_generation(
        self,
        generate_fn: Callable,
        input_text: str,
        max_tokens: int = 100
    ) -> BenchmarkResult:
        """
        测试生成延迟

        Args:
            generate_fn: 生成函数
            input_text: 输入文本
            max_tokens: 最大 token 数

        Returns:
            BenchmarkResult: 测试结果
        """
        for _ in range(self.warmup_iterations):
            generate_fn(input_text, max_tokens=max_tokens)

        latencies = []

        for _ in range(self.test_iterations):
            start_time = time.perf_counter()
            generate_fn(input_text, max_tokens=max_tokens)
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)

        avg_latency = sum(latencies) / len(latencies)
        p50_latency = sorted(latencies)[len(latencies) // 2]
        p99_latency = sorted(latencies)[int(len(latencies) * 0.99)]

        return BenchmarkResult(
            test_name="generation_latency",
            benchmark_type=BenchmarkType.LATENCY,
            value=avg_latency,
            unit="ms",
            timestamp=time.time(),
            metadata={
                "p50_ms": p50_latency,
                "p99_ms": p99_latency,
                "iterations": self.test_iterations,
                "max_tokens": max_tokens
            }
        )

    def benchmark_per_token(
        self,
        generate_fn: Callable,
        input_text: str,
        max_tokens: int = 100
    ) -> BenchmarkResult:
        """测试每 token 延迟"""
        start_time = time.perf_counter()
        result = generate_fn(input_text, max_tokens=max_tokens)
        end_time = time.perf_counter()

        total_time = (end_time - start_time) * 1000
        num_tokens = len(result) if isinstance(result, str) else max_tokens
        per_token_latency = total_time / num_tokens

        return BenchmarkResult(
            test_name="per_token_latency",
            benchmark_type=BenchmarkType.LATENCY,
            value=per_token_latency,
            unit="ms/token",
            timestamp=time.time(),
            metadata={"total_tokens": num_tokens}
        )


class MemoryBenchmark:
    """内存基准测试"""

    def __init__(self):
        self.process = psutil.Process(os.getpid())

    def get_memory_usage(self) -> float:
        """获取当前内存使用量 (GB)"""
        mem_info = self.process.memory_info()
        return mem_info.rss / (1024 ** 3)

    def get_gpu_memory_usage(self) -> float:
        """获取 GPU 内存使用量 (GB)"""
        if not torch.cuda.is_available():
            return 0.0

        return torch.cuda.memory_allocated() / (1024 ** 3)

    def benchmark_model_memory(
        self,
        model_fn: Callable,
        model_name: str
    ) -> BenchmarkResult:
        """测试模型内存占用"""
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        model = model_fn()
        model_memory = self.get_memory_usage()
        gpu_memory = self.get_gpu_memory_usage()

        return BenchmarkResult(
            test_name="model_memory",
            benchmark_type=BenchmarkType.MEMORY,
            value=model_memory,
            unit="GB",
            timestamp=time.time(),
            metadata={
                "model_name": model_name,
                "gpu_memory_gb": gpu_memory,
                "total_memory_gb": model_memory + gpu_memory
            }
        )


class ThroughputBenchmark:
    """吞吐量基准测试"""

    def __init__(self, batch_sizes: List[int] = None):
        self.batch_sizes = batch_sizes or [1, 4, 8, 16]

    def benchmark_batch_throughput(
        self,
        generate_fn: Callable,
        input_text: str,
        max_tokens: int = 50,
        duration_seconds: float = 10.0
    ) -> List[BenchmarkResult]:
        """测试批处理吞吐量"""
        results = []

        for batch_size in self.batch_sizes:
            num_tokens_generated = 0
            start_time = time.time()
            end_time = start_time + duration_seconds

            while time.time() < end_time:
                for _ in range(batch_size):
                    result = generate_fn(input_text, max_tokens=max_tokens)
                    num_tokens_generated += len(result) if isinstance(result, str) else max_tokens

            actual_duration = time.time() - start_time
            tokens_per_sec = num_tokens_generated / actual_duration

            results.append(BenchmarkResult(
                test_name=f"batch_throughput_{batch_size}",
                benchmark_type=BenchmarkType.THROUGHPUT,
                value=tokens_per_sec,
                unit="tokens/sec",
                timestamp=time.time(),
                metadata={
                    "batch_size": batch_size,
                    "total_tokens": num_tokens_generated,
                    "duration_sec": actual_duration
                }
            ))

        return results


class OverheadBenchmark:
    """性能开销基准测试"""

    def __init__(self):
        self.baseline_latency = 40.0  # 标准 Transformer 7B 基准 (ms)

    def benchmark_overhead(
        self,
        enlighten_latency_ms: float,
        baseline_latency_ms: Optional[float] = None
    ) -> BenchmarkResult:
        """
        测试 EnlightenLM 相对标准 Transformer 的性能开销

        Args:
            enlighten_latency_ms: EnlightenLM 延迟 (ms)
            baseline_latency_ms: 基准延迟，None 使用默认 40ms

        Returns:
            BenchmarkResult: 开销测试结果
        """
        baseline = baseline_latency_ms or self.baseline_latency
        overhead_percent = ((enlighten_latency_ms - baseline) / baseline) * 100

        return BenchmarkResult(
            test_name="enlighten_overhead",
            benchmark_type=BenchmarkType.OVERHEAD,
            value=overhead_percent,
            unit="%",
            timestamp=time.time(),
            metadata={
                "enlighten_latency_ms": enlighten_latency_ms,
                "baseline_latency_ms": baseline,
                "target_overhead_range": "+5% to +15%",
                "within_target": 5 <= overhead_percent <= 15
            }
        )


class PerformanceBenchmark:
    """
    综合性能基准测试套件

    验证 EnlightenLM 是否满足以下目标:
    - 延迟增加: +5% ~ +15%
    - 显存增加: +0.5GB ~ +1.5GB
    - 幻觉率: < 2% (full), < 3% (balanced), < 5% (lightweight)
    """

    def __init__(
        self,
        mode: str = "balanced",
        baseline_latency_ms: float = 40.0
    ):
        self.mode = mode
        self.baseline_latency_ms = baseline_latency_ms

        self.latency_benchmark = LatencyBenchmark()
        self.memory_benchmark = MemoryBenchmark()
        self.throughput_benchmark = ThroughputBenchmark()
        self.overhead_benchmark = OverheadBenchmark()

        self.results: List[BenchmarkResult] = []

    def run_all_benchmarks(
        self,
        generate_fn: Callable,
        test_input: str = "解释量子计算的基本原理"
    ) -> Dict[str, Any]:
        """
        运行所有基准测试

        Args:
            generate_fn: 生成函数
            test_input: 测试输入

        Returns:
            测试结果摘要
        """
        print(f"运行 {self.mode} 模式性能基准测试...")

        latency_result = self.latency_benchmark.benchmark_generation(
            generate_fn, test_input, max_tokens=100
        )
        self.results.append(latency_result)
        print(f"延迟测试完成: {latency_result.value:.2f}ms")

        overhead_result = self.overhead_benchmark.benchmark_overhead(
            latency_result.value
        )
        self.results.append(overhead_result)
        print(f"开销测试完成: {overhead_result.value:.2f}%")

        throughput_results = self.throughput_benchmark.benchmark_batch_throughput(
            generate_fn, test_input, max_tokens=50, duration_seconds=5.0
        )
        self.results.extend(throughput_results)
        print(f"吞吐量测试完成: {len(throughput_results)} 项")

        return self.get_summary()

    def get_summary(self) -> Dict[str, Any]:
        """获取测试结果摘要"""
        summary = {
            "mode": self.mode,
            "timestamp": datetime.now().isoformat(),
            "tests": [r.to_dict() for r in self.results],
            "targets": self._get_targets()
        }

        overhead_results = [r for r in self.results if r.benchmark_type == BenchmarkType.OVERHEAD]
        if overhead_results:
            latest_overhead = overhead_results[-1]
            summary["status"] = "PASS" if latest_overhead.metadata.get("within_target") else "FAIL"

        return summary

    def _get_targets(self) -> Dict:
        """获取性能目标"""
        targets = {
            "lightweight": {
                "overhead_percent": "+5%",
                "memory_overhead_gb": 0.5,
                "repetition_rate": "< 5%"
            },
            "balanced": {
                "overhead_percent": "+10%",
                "memory_overhead_gb": 1.0,
                "repetition_rate": "< 3%"
            },
            "full": {
                "overhead_percent": "+15%",
                "memory_overhead_gb": 1.5,
                "repetition_rate": "< 2%"
            }
        }

        return targets.get(self.mode, targets["balanced"])

    def save_results(self, path: str) -> None:
        """保存测试结果到文件"""
        summary = self.get_summary()

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    @staticmethod
    def compare_modes(results: Dict[str, Dict]) -> Dict:
        """比较不同模式的性能"""
        comparison = {
            "modes": list(results.keys()),
            "comparison": {}
        }

        for mode, result in results.items():
            overhead_results = [r for r in result.get("tests", []) if r.get("benchmark_type") == "overhead"]
            if overhead_results:
                comparison["comparison"][mode] = {
                    "overhead_percent": overhead_results[0].get("value"),
                    "status": result.get("status")
                }

        return comparison
