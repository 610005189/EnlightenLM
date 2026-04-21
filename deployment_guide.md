# 觉悟三层架构部署指南

## 1. 系统要求

### 1.1 硬件要求
- **CPU**: 至少 8 核
- **内存**: 至少 32GB
- **GPU**: 推荐使用 NVIDIA GPU，至少 16GB 显存
- **存储空间**: 至少 100GB

### 1.2 软件要求
- **操作系统**: Ubuntu 20.04 或 Windows 10/11
- **Python**: 3.8 或更高版本
- **CUDA**: 11.7 或更高版本（如果使用 GPU）

## 2. 安装步骤

### 2.1 克隆代码库
```bash
git clone https://github.com/your-username/enlightenlm.git
cd enlightenlm
```

### 2.2 创建虚拟环境
```bash
# 使用 venv
python -m venv venv

# 激活虚拟环境
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 2.3 安装依赖
```bash
pip install -r requirements.txt
```

### 2.4 下载模型
系统默认使用以下模型：
- L1 基座模型: Qwen/Qwen2-7B
- L2 自描述模型: distilbert-base-uncased
- L3 元注意力控制器: google/bert-mini-uncased

首次运行时会自动下载这些模型。

## 3. 配置说明

### 3.1 核心配置文件
配置文件位于 `config/config.yaml`，包含以下部分：

```yaml
# API 配置
api:
  host: "0.0.0.0"
  port: 8000
  cors: true
  allowed_origins: ["*"]

# L1 基座模型配置
l1:
  model_name: "Qwen/Qwen2-7B"
  device: "cuda"
  inference:
    max_new_tokens: 512
    temperature: 0.7
    top_p: 0.9
  attention_bias:
    rank: 8
    strategy: "layer-wise"
  optimization:
    quantization: true
    quantization_bits: 4
    use_cache: true
    batch_size: 1

# L2 自描述模型配置
l2:
  model_name: "distilbert-base-uncased"
  device: "cuda"
  inference:
    max_new_tokens: 256
    temperature: 0.6
  cutoff_controller:
    max_depth: 1
    similarity_threshold: 0.85
    self_reference_weight: 1.0

# L3 元注意力控制器配置
l3:
  model_name: "google/bert-mini-uncased"
  device: "cuda"
  core_values:
    negative_vocab_path: "config/negative_vocab.txt"
    positive_vocab_path: "config/positive_vocab.txt"
    negative_bias: -1000000000.0
    positive_bias: 2.0
  user_params:
    creativity_range: [-1, 1]
    detail_attention_range: [0, 1]
    safety_margin_range: [0.3, 1]
    role_presets: ["teacher", "assistant", "analyst"]

# 审计日志配置
audit:
  storage_path: "logs/audit"
  compression:
    enabled: true
    level: 6
  hash_chain:
    algorithm: "sha256"
    initial_hash: "0000000000000000000000000000000000000000000000000000000000000000"
  security:
    enabled: true
    type: "process"
    storage_path: "logs/secure"
    hash_algorithm: "sha256"
  threat_detection:
    enabled: true
    thresholds:
      request_per_minute: 60
      max_input_length: 10000
      max_special_char_ratio: 0.5
```

### 3.2 词汇表配置
- `config/negative_vocab.txt`: 负向词汇表，包含需要拒绝的词汇
- `config/positive_vocab.txt`: 正向词汇表，包含需要增强的词汇

## 4. 部署方式

### 4.1 本地开发部署
```bash
python src/main.py
```

### 4.2 生产环境部署

#### 4.2.1 使用 Gunicorn
```bash
pip install gunicorn uvicorn

# 启动服务器
gunicorn -w 4 -k uvicorn.workers.UvicornWorker src.main:app
```

#### 4.2.2 使用 Docker

**Dockerfile**:
```dockerfile
FROM python:3.8

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["python", "src/main.py"]
```

**构建和运行**:
```bash
docker build -t enlightenlm .
docker run -p 8000:8000 --gpus all enlightenlm
```

### 4.3 集群部署

对于大规模部署，建议使用 Kubernetes 进行容器编排：

**deployment.yaml**:
```yaml
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
        image: your-registry/enlightenlm:latest
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 1
---
apiVersion: v1
kind: Service
metadata:
  name: enlightenlm
spec:
  selector:
    app: enlightenlm
  ports:
  - port: 8000
    targetPort: 8000
  type: LoadBalancer
```

## 5. 监控和维护

### 5.1 日志管理
- 应用日志: `logs/app.log`
- 审计日志: `logs/audit/`
- 安全日志: `logs/secure/`

### 5.2 性能监控
使用 Prometheus 和 Grafana 监控系统性能：

1. 安装 Prometheus 和 Grafana
2. 配置 Prometheus 抓取应用指标
3. 创建 Grafana 仪表板

### 5.3 定期维护
- 每周更新词汇表
- 每月检查哈希链完整性
- 每季度进行安全测试

## 6. 故障排查

### 6.1 常见问题

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| 模型加载失败 | 网络问题或模型不存在 | 检查网络连接，确保模型名称正确 |
| 内存不足 | GPU 显存不够 | 启用量化，减少 batch_size |
| 推理速度慢 | 模型太大或硬件不足 | 启用缓存，使用更强大的硬件 |
| 威胁检测误报 | 规则过于严格 | 调整威胁检测阈值 |

### 6.2 调试技巧
- 启用详细日志: `export LOG_LEVEL=DEBUG`
- 检查 GPU 使用情况: `nvidia-smi`
- 验证哈希链: `curl http://localhost:8000/audit/verify`
- 查看威胁检测统计: `curl http://localhost:8000/audit/threat/statistics`

## 7. 安全建议

- 定期更新词汇表
- 启用安全隔离
- 配置合适的威胁检测阈值
- 限制 API 访问
- 使用 HTTPS
- 定期备份审计日志

## 8. 扩展性

### 8.1 模型替换
可以通过修改 `config/config.yaml` 中的模型名称来替换不同的模型：

```yaml
l1:
  model_name: "meta-llama/Llama-2-7b-chat-hf"
```

### 8.2 水平扩展
通过增加实例数量来提高系统吞吐量：
- 使用负载均衡器
- 配置自动缩放
- 优化数据库连接池

### 8.3 垂直扩展
通过升级硬件来提高系统性能：
- 使用更强大的 GPU
- 增加内存
- 使用更快的存储

## 9. 结论

觉悟三层架构提供了一套完整的安全、自指、可审计的大模型推理系统。通过本指南的部署步骤，您可以快速搭建和运行该系统，为您的应用提供安全可靠的 AI 服务。