"""
性能测试 - 测试系统的响应时间和资源使用情况

测试场景：
1. 响应时间测试
2. 内存使用测试
3. CPU 使用率测试
4. 并发请求测试
"""

import pytest
import time
import psutil
import concurrent.futures
from typing import List, Dict, Any

from enlighten.hybrid_architecture import HybridEnlightenLM
from enlighten.config.loader import load_config


class TestPerformance:
    """性能测试"""

    @pytest.fixture
    def hybrid_model(self):
        """创建 HybridEnlightenLM 实例"""
        config = load_config("balanced")
        config.use_l1_adapter = False
        config.use_l2_adapter = True
        config.use_l3_controller = True
        config.use_bayesian_l3 = True
        
        model = HybridEnlightenLM(
            config=config,
            model_name="llama3:latest",
            model_type="ollama"
        )
        return model

    def test_response_time(self, hybrid_model):
        """测试响应时间"""
        prompts = [
            "请解释什么是人工智能",
            "写一个简短的故事",
            "计算 12345 * 67890",
            "解释量子计算的基本原理",
            "如何学习Python编程"
        ]
        
        response_times = []
        
        for prompt in prompts:
            start_time = time.time()
            result = hybrid_model.generate(
                prompt=prompt,
                session_id=f"perf_test_{time.time()}",
                max_new_tokens=50,
                temperature=0.7
            )
            end_time = time.time()
            response_time = end_time - start_time
            response_times.append(response_time)
            
            assert result is not None
            assert "text" in result
        
        # 计算平均响应时间
        avg_response_time = sum(response_times) / len(response_times)
        print(f"平均响应时间: {avg_response_time:.2f} 秒")
        
        # 响应时间应该在合理范围内（根据模型和硬件不同，阈值可调整）
        assert avg_response_time < 30, f"平均响应时间过长: {avg_response_time:.2f} 秒"

    def test_memory_usage(self, hybrid_model):
        """测试内存使用"""
        # 测试前内存使用
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 执行多次推理
        for i in range(5):
            prompt = f"内存测试 {i+1}"
            result = hybrid_model.generate(
                prompt=prompt,
                session_id=f"memory_test_{i+1}",
                max_new_tokens=50
            )
            assert result is not None
        
        # 测试后内存使用
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"初始内存: {initial_memory:.2f} MB")
        print(f"最终内存: {final_memory:.2f} MB")
        print(f"内存增长: {memory_increase:.2f} MB")
        
        # 内存增长应该在合理范围内
        assert memory_increase < 500, f"内存增长过大: {memory_increase:.2f} MB"

    def test_cpu_usage(self, hybrid_model):
        """测试 CPU 使用率"""
        # 执行推理并监控 CPU 使用率
        prompt = "请详细解释机器学习的各种算法"
        
        # 开始监控
        process = psutil.Process()
        cpu_percentages = []
        
        # 执行推理
        start_time = time.time()
        result = hybrid_model.generate(
            prompt=prompt,
            session_id="cpu_test",
            max_new_tokens=100
        )
        end_time = time.time()
        
        # 计算平均 CPU 使用率（这里使用简化的方法）
        avg_cpu = process.cpu_percent(interval=end_time - start_time)
        
        print(f"CPU 使用率: {avg_cpu:.2f}%")
        print(f"推理时间: {end_time - start_time:.2f} 秒")
        
        assert result is not None

    def test_concurrent_requests(self, hybrid_model):
        """测试并发请求"""
        prompts = [
            "请解释什么是深度学习",
            "写一个关于未来的故事",
            "如何提高编程技能",
            "解释区块链技术",
            "什么是云计算"
        ]
        
        results = []
        response_times = []
        
        def process_request(prompt, session_id):
            start_time = time.time()
            result = hybrid_model.generate(
                prompt=prompt,
                session_id=session_id,
                max_new_tokens=30,
                temperature=0.7
            )
            end_time = time.time()
            return result, end_time - start_time
        
        # 使用线程池并发处理
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_to_prompt = {
                executor.submit(process_request, prompt, f"concurrent_{i}"):
                prompt for i, prompt in enumerate(prompts)
            }
            
            for future in concurrent.futures.as_completed(future_to_prompt):
                result, response_time = future.result()
                results.append(result)
                response_times.append(response_time)
        
        # 验证所有请求都成功
        for result in results:
            assert result is not None
            assert "text" in result
        
        # 计算统计信息
        avg_response_time = sum(response_times) / len(response_times)
        max_response_time = max(response_times)
        min_response_time = min(response_times)
        
        print(f"并发测试结果:")
        print(f"平均响应时间: {avg_response_time:.2f} 秒")
        print(f"最大响应时间: {max_response_time:.2f} 秒")
        print(f"最小响应时间: {min_response_time:.2f} 秒")
        
        # 并发响应时间应该在合理范围内
        assert avg_response_time < 45, f"并发平均响应时间过长: {avg_response_time:.2f} 秒"

    def test_throughput(self, hybrid_model):
        """测试吞吐量"""
        num_requests = 10
        prompts = [f"吞吐量测试 {i+1}" for i in range(num_requests)]
        
        start_time = time.time()
        
        for i, prompt in enumerate(prompts):
            result = hybrid_model.generate(
                prompt=prompt,
                session_id=f"throughput_{i}",
                max_new_tokens=20
            )
            assert result is not None
        
        end_time = time.time()
        total_time = end_time - start_time
        throughput = num_requests / total_time
        
        print(f"吞吐量测试结果:")
        print(f"总请求数: {num_requests}")
        print(f"总时间: {total_time:.2f} 秒")
        print(f"吞吐量: {throughput:.2f} 请求/秒")
        
        # 吞吐量应该大于 0.1 请求/秒
        assert throughput > 0.1, f"吞吐量过低: {throughput:.2f} 请求/秒"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])