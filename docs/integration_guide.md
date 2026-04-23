# EnlightenLM 集成指南

> 版本: v1.0
> 更新日期: 2026-04-23

---

## 目录

1. [环境准备](#1-环境准备)
2. [快速开始](#2-快速开始)
3. [HuggingFace集成](#3-huggingface集成)
4. [vLLM集成](#4-vllm集成)
5. [自定义配置](#5-自定义配置)
6. [性能优化](#6-性能优化)
7. [生产部署](#7-生产部署)
8. [故障排除](#8-故障排除)

---

## 1. 环境准备

### 1.1 系统要求

| 要求 | 最低配置 | 推荐配置 |
|------|---------|---------|
| Python | 3.8+ | 3.10 |
| 内存 | 16GB | 32GB |
| 显存 | 8GB (可选) | 16GB+ |
| 磁盘 | 10GB | 50GB SSD |

### 1.2 依赖安装

```bash
# 基础依赖
pip install torch>=2.0.0
pip install transformers>=4.30.0
pip install accelerate>=0.20.0

# 可选依赖
pip install vllm>=0.2.0  # 生产级推理
pip install fastapi>=0.100.0  # API服务
pip install uvicorn>=0.23.0
```

### 1.3 目录结构

```
EnlightenLM/
├── enlighten/           # 核心框架
├── configs/            # 配置文件
├── docs/              # 文档
├── examples/          # 示例
└── tests/            # 测试
```

---

## 2. 快速开始

### 2.1 基础使用

```python
from enlighten import EnlightenLM

# 初始化
model = EnlightenLM(
    model_name="deepseek-ai/DeepSeek-V3",
    device="cuda"  # 或 "cpu"
)

# 推理
result = model.generate(
    text="解释量子计算的基本原理",
    user_params={
        "creativity": 0.5,
        "detail": 0.8,
        "safety": 0.7
    }
)

print(result.output)
print(result.meta_description)
```

### 2.2 API服务启动

```bash
# 启动API服务
python -m enlighten.api_server --host 0.0.0.0 --port 8000

# 调用API
curl -X POST http://localhost:8000/inference \
  -H "Content-Type: application/json" \
  -d '{
    "text": "什么是人工智能？",
    "user_params": {
      "creativity": 0.5,
      "detail": 0.8
    }
  }'
```

---

## 3. HuggingFace集成

### 3.1 模型加载

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from enlighten import L1Generation, L2WorkingMemory, L3Controller

# 加载基础模型
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3")
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-V3")

# 包装为EnlightenLM组件
l1 = L1Generation(model, tokenizer)
l2 = L2WorkingMemory(memory_size=512, embedding_dim=1024)
l3 = L3Controller(config="configs/hyperparameters.yaml")

# 组装
enlighten = EnlightenLM(l1, l2, l3)
```

### 3.2 自定义任务偏置

```python
# 定义任务类型
TASKBias = {
    "math": {
        "dan_strength": 1.0,
        "van_threshold": 0.8,
        "temperature": 0.3
    },
    "creative": {
        "dan_strength": 0.5,
        "van_threshold": 0.6,
        "temperature": 1.0
    },
    "safety_critical": {
        "dan_strength": 1.5,
        "van_threshold": 0.9,
        "temperature": 0.2
    }
}

# 应用任务偏置
result = enlighten.generate(
    text="数学问题",
    task_type="math",
    task_bias=TASKBias["math"]
)
```

### 3.3 注意力可视化

```python
from enlighten.utils import visualize_attention

# 获取注意力权重
attn_weights = result.attention_weights

# 可视化
visualize_attention(
    attn_weights,
    tokens=result.tokens,
    output_file="attention_heatmap.png"
)
```

---

## 4. vLLM集成

### 4.1 安装vLLM

```bash
pip install vllm>=0.2.0
```

### 4.2 vLLM后端配置

```python
from enlighten.integrations.vllm import EnlightenVLLM

# 初始化vLLM后端
vllm_backend = EnlightenVLLM(
    model_name="deepseek-ai/DeepSeek-V3",
    tensor_parallel_size=2,  # 多GPU并行
    max_model_len=4096
)

# 包装为EnlightenLM
enlighten = EnlightenLM(
    backend=vllm_backend,
    l2_working_memory=L2WorkingMemory(memory_size=512),
    l3_controller=L3Controller()
)
```

### 4.3 性能对比

| 配置 | HuggingFace | vLLM | 提升 |
|------|-------------|------|------|
| Throughput (tokens/s) | 50 | 200 | 4x |
| Latency P50 (ms) | 100 | 30 | 3.3x |
| Latency P99 (ms) | 500 | 80 | 6.25x |

---

## 5. 自定义配置

### 5.1 工作记忆配置

```yaml
# configs/working_memory.yaml
working_memory:
  memory_size: 512       # m: 活跃token数量
  embedding_dim: 1024    # d: 嵌入维度
  update_strategy: "topk"  # 或 "threshold", "random"
  momentum: 0.9          # 动量更新

entropy_tracker:
  window_size: 100       # 滑动窗口大小
  compute_interval: 1    # 每N步计算一次
```

```python
from enlighten import L2WorkingMemory

l2 = L2WorkingMemory(
    memory_size=512,
    embedding_dim=1024,
    config_path="configs/working_memory.yaml"
)
```

### 5.2 元控制器配置

```yaml
# configs/controller.yaml
l3_controller:
  entropy_threshold: 0.5    # μ_H < 0.5 触发截断
  variance_threshold: 0.05  # σ_H < 0.05 触发截断

  tau_range: [0.1, 2.0]    # 温度范围
  theta_range: [0.5, 0.9]  # 稀疏阈值范围
  alpha_range: [0.0, 1.0]  # DMN系数范围

  van_priority: true        # VAN事件优先处理
  cutoff_cool_down: 10      # 截断冷却步数
```

### 5.3 核心价值观配置

```yaml
# configs/core_rules.yaml
core_rules:
  negative_vocab:
    - path: "configs/vocab/negative.txt"
      weight: -1e9

  positive_vocab:
    - path: "configs/vocab/positive.txt"
      weight: 2.0

  exception_tokens:          # 特殊处理token
    - token: "[PAD]"
      action: "ignore"
    - token: "[UNK]"
      action: "mask"
```

---

## 6. 性能优化

### 6.1 显存优化

```python
from enlighten.optimization import MemoryOptimizer

# 启用量化
optimizer = MemoryOptimizer()
optimizer.apply_quantization(model, bits=4)

# 启用梯度检查点
optimizer.apply_gradient_checkpointing(model)

# 启用混合精度
optimizer.enable_fp16()
```

### 6.2 延迟优化

```python
# 启用批处理
enlighten.enable_batch_inference(batch_size=8)

# 启用投机解码
enlighten.enable_speculative_decoding(draft_model="Qwen2.5-0.5B")

# 启用缓存
enlighten.enable_kv_cache()
```

### 6.3 长上下文优化

```python
# 配置稀疏注意力
enlighten.configure_sparse_attention(
    mode="flash",      # 或 "linear", "softmax"
    sparse_ratio=0.1   # 保留10%的注意力
)
```

---

## 7. 生产部署

### 7.1 Docker部署

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "-m", "enlighten.api_server", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# 构建镜像
docker build -t enlightenlm:latest .

# 运行容器
docker run -p 8000:8000 \
  --gpus all \
  -v /path/to/configs:/app/configs \
  enlightenlm:latest
```

### 7.2 Kubernetes部署

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: enlightenlm
spec:
  replicas: 3
  selector:
    matchLabels:
      app: enlightenlm
  template:
    metadata:
      labels:
        app: enlightenlm
    spec:
      containers:
      - name: enlightenlm
        image: enlightenlm:latest
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: 32Gi
          requests:
            nvidia.com/gpu: 1
            memory: 16Gi
```

### 7.3 高可用配置

```python
# 多实例负载均衡
from enlighten.deployment import HAProxy

haproxy = HAProxy(
    backend="enlightenlm",
    servers=[
        {"host": "10.0.0.1", "port": 8000},
        {"host": "10.0.0.2", "port": 8000},
        {"host": "10.0.0.3", "port": 8000}
    ],
    algorithm="round_robin"
)
```

---

## 8. 故障排除

### 8.1 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| CUDA OOM | 显存不足 | 启用量化或减少batch_size |
| 模型加载慢 | 网络问题 | 使用本地模型缓存 |
| 截断频繁 | 熵阈值过低 | 调高entropy_threshold |
| 延迟高 | 批处理未启用 | 启用batch_inference |

### 8.2 日志调试

```python
import logging

# 启用详细日志
logging.basicConfig(level=logging.DEBUG)

# 查看截断决策
enlighten.enable_cutoff_debug()

# 查看注意力权重
enlighten.enable_attention_debug()
```

### 8.3 性能诊断

```python
from enlighten.utils import PerformanceProfiler

profiler = PerformanceProfiler()
profiler.start()

# 运行推理
result = enlighten.generate(text)

profiler.stop()
profiler.report()
```

---

## 附录

### A. 配置文件参考

完整配置示例见 [hyperparameters.yaml](../configs/hyperparameters.yaml)

### B. 环境变量

| 变量 | 说明 | 默认值 |
|------|------|-------|
| ENLIGHTEN_MODEL_PATH | 模型路径 | HuggingFace Hub |
| ENLIGHTEN_DEVICE | 运行设备 | auto |
| ENLIGHTEN_LOG_LEVEL | 日志级别 | INFO |
| ENLIGHTEN_CACHE_DIR | 缓存目录 | ~/.cache |

### C. 联系方式

- GitHub Issues: https://github.com/610005189/EnlightenLM/issues
- 文档: https://enlightenlm.readthedocs.io

---

*文档版本: v1.0*
*最后更新: 2026-04-23*
