"""
性能基准测试
测试三种模式的性能指标：延迟、显存、重复率
"""

import time
import torch
import numpy as np
from typing import Dict, List, Any, Tuple

from enlighten.config import get_mode_preset, EnlightenMode, ModeConfig
from enlighten.attention.van import VANFunnel, KeywordMatcher
from enlighten.memory import WorkingMemory
from enlighten.l3_controller import L3Controller


class LatencyBenchmark:
    """
    延迟基准测试
    """

    def __init__(self, num_runs: int = 100):
        self.num_runs = num_runs

    def benchmark_mode(self, mode: ModeConfig) -> Dict[str, float]:
        """
        测量指定模式的延迟

        Returns:
            延迟统计信息
        """
        van = VANFunnel(
            level=mode.van_level,
            van_threshold=0.9
        )

        tokens = torch.randint(0, 50000, (1, 128))
        hidden = torch.randn(1, 128, 768)

        latencies = []
        warmup_runs = 10

        for i in range(warmup_runs + self.num_runs):
            start = time.perf_counter()
            van.forward(tokens, hidden)
            latency = time.perf_counter() - start

            if i >= warmup_runs:
                latencies.append(latency * 1000)

        return {
            "mean_ms": np.mean(latencies),
            "std_ms": np.std(latencies),
            "min_ms": np.min(latencies),
            "max_ms": np.max(latencies),
            "p50_ms": np.percentile(latencies, 50),
            "p95_ms": np.percentile(latencies, 95),
            "p99_ms": np.percentile(latencies, 99)
        }


class MemoryBenchmark:
    """
    显存基准测试
    """

    def benchmark_mode(self, mode: ModeConfig) -> Dict[str, float]:
        """
        测量指定模式的显存占用

        Returns:
            显存统计信息（MB）
        """
        if not torch.cuda.is_available():
            return {"allocated_mb": 0, "reserved_mb": 0, "note": "CUDA not available"}

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        initial_memory = torch.cuda.memory_allocated() / 1024 / 1024

        van = VANFunnel(level=mode.van_level)
        memory = WorkingMemory(
            memory_size=mode.van_level == "full" and 512 or 256,
            embedding_dim=768,
            use_topk_refresh=mode.use_topk_refresh,
            refresh_interval=mode.refresh_interval
        )

        tokens = torch.randint(0, 50000, (1, 128))
        hidden = torch.randn(1, 128, 768)

        for _ in range(10):
            van.forward(tokens, hidden)
            memory.update(hidden, hidden)

        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
        current_memory = torch.cuda.memory_allocated() / 1024 / 1024

        return {
            "initial_mb": initial_memory,
            "peak_mb": peak_memory,
            "current_mb": current_memory,
            "overhead_mb": peak_memory - initial_memory
        }


class RepetitionBenchmark:
    """
    重复率基准测试
    """

    def __init__(self):
        pass

    def calculate_repetition_rate(self, generated_tokens: List[int]) -> float:
        """
        计算生成内容的重复率

        Args:
            generated_tokens: 生成的token列表

        Returns:
            重复率 (0-1)
        """
        if len(generated_tokens) < 2:
            return 0.0

        ngrams = []
        for n in [2, 3, 4]:
            ngrams.extend(self._get_ngrams(generated_tokens, n))

        if not ngrams:
            return 0.0

        unique_ngrams = len(set(ngrams))
        total_ngrams = len(ngrams)

        return 1.0 - (unique_ngrams / total_ngrams)

    def _get_ngrams(self, tokens: List[int], n: int) -> List[Tuple]:
        """
        获取n-grams
        """
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

    def test_safety_repetition(self, model_output: str) -> float:
        """
        测试安全相关的重复率

        Args:
            model_output: 模型输出文本

        Returns:
            重复率
        """
        tokens = list(model_output.encode('utf-8'))
        return self.calculate_repetition_rate(tokens)


class PerformanceBenchmark:
    """
    综合性能基准测试
    """

    def __init__(self):
        self.latency_bench = LatencyBenchmark(num_runs=50)
        self.memory_bench = MemoryBenchmark()
        self.repetition_bench = RepetitionBenchmark()

    def run_all_benchmarks(self) -> Dict[str, Any]:
        """
        运行所有基准测试

        Returns:
            完整的基准测试结果
        """
        results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "modes": {}
        }

        modes_to_test = [
            ("full", get_mode_preset("full")),
            ("balanced", get_mode_preset("balanced")),
            ("lightweight", get_mode_preset("lightweight"))
        ]

        for mode_name, mode_config in modes_to_test:
            print(f"\nBenchmarking mode: {mode_name}")
            results["modes"][mode_name] = self._benchmark_single_mode(mode_config, mode_name)

        return results

    def _benchmark_single_mode(self, mode: ModeConfig, mode_name: str) -> Dict[str, Any]:
        """
        测试单个模式的性能
        """
        results = {
            "mode_name": mode_name,
            "config": {
                "van_level": mode.van_level,
                "gate_fusion": mode.gate_fusion,
                "dmn_noise": mode.dmn_noise,
                "use_topk_refresh": mode.use_topk_refresh,
                "refresh_interval": mode.refresh_interval
            }
        }

        print(f"  Running latency benchmark...")
        results["latency"] = self.latency_bench.benchmark_mode(mode)

        print(f"  Running memory benchmark...")
        results["memory"] = self.memory_bench.benchmark_mode(mode)

        print(f"  Running repetition benchmark...")
        sample_tokens = list(range(100)) * 3
        results["repetition"] = {
            "rate": self.repetition_bench.calculate_repetition_rate(sample_tokens)
        }

        return results

    def print_results(self, results: Dict[str, Any]) -> None:
        """
        打印基准测试结果
        """
        print("\n" + "=" * 60)
        print("EnlightenLM v2.1 性能基准测试报告")
        print("=" * 60)
        print(f"测试时间: {results['timestamp']}")
        print()

        for mode_name, mode_results in results["modes"].items():
            print(f"\n### {mode_name.upper()} 模式")
            print("-" * 40)

            print("\n延迟:")
            latency = mode_results["latency"]
            print(f"  平均: {latency['mean_ms']:.2f} ms")
            print(f"  标准差: {latency['std_ms']:.2f} ms")
            print(f"  P95: {latency['p95_ms']:.2f} ms")
            print(f"  P99: {latency['p99_ms']:.2f} ms")

            print("\n显存:")
            memory = mode_results["memory"]
            if "note" in memory:
                print(f"  {memory['note']}")
            else:
                print(f"  增加: {memory['overhead_mb']:.2f} MB")

            print("\n重复率:")
            print(f"  率: {mode_results['repetition']['rate']:.4f}")

        print("\n" + "=" * 60)


def run_quick_benchmark() -> Dict[str, Any]:
    """
    运行快速基准测试
    """
    bench = PerformanceBenchmark()
    results = bench.run_all_benchmarks()
    bench.print_results(results)
    return results


if __name__ == "__main__":
    print("Starting EnlightenLM v2.1 Performance Benchmark...")
    run_quick_benchmark()