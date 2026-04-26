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

#### 4.2.2.1 快速开始

**1. 克隆代码库**
```bash
git clone https://github.com/610005189/enlightenlm.git
cd enlightenlm
```

**2. 配置环境变量**
```bash
cp .env.example .env
# 编辑 .env 文件，填入 DeepSeek API Key
vim .env
```

**3. 构建镜像**
```bash
# API 模式 (CPU)
docker build -t enlightenlm:api --target production .

# 本地 GPU 模式
docker build -t enlightenlm:gpu --target production-gpu .
```

**4. 运行容器**
```bash
# API 模式
docker run -d \
  --name enlightenlm-api \
  -p 8000:8000 \
  --env-file .env \
  -v $(pwd)/logs:/app/logs \
  enlightenlm:api

# GPU 模式
docker run -d \
  --name enlightenlm-api-gpu \
  --gpus all \
  -p 8001:8000 \
  --env-file .env \
  -v $(pwd)/logs:/app/logs \
  enlightenlm:gpu
```

#### 4.2.2.2 使用 docker-compose 部署

**API 模式部署（推荐用于开发/测试）**
```bash
# 启动服务
docker-compose up -d api

# 查看日志
docker-compose logs -f api

# 停止服务
docker-compose down
```

**本地 GPU 模式部署**
```bash
# 启动 GPU 服务
docker-compose up -d api-gpu

# 查看日志
docker-compose logs -f api-gpu
```

**完整生产环境部署**
```bash
# 启动所有服务 (API + Nginx + Redis + Prometheus + Grafana)
docker-compose --profile production up -d

# 访问服务
# - API: http://localhost:8000
# - Nginx: http://localhost:80
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000 (admin/admin)
```

#### 4.2.2.3 多阶段 Dockerfile 说明

| Stage | 用途 | 基础镜像 | 大小 |
|-------|------|----------|------|
| builder | 依赖安装 | python:3.11-slim | ~200MB |
| development | 开发环境 | python:3.11-slim | ~500MB |
| production | 生产环境 (CPU) | python:3.11-slim | ~400MB |
| production-gpu | 生产环境 (GPU) | nvidia/cuda:12.1 | ~2GB |

#### 4.2.2.4 镜像构建优化

**构建时使用 BuildKit 加速**
```bash
export DOCKER_BUILDKIT=1
docker build -t enlightenlm:api --target production .
```

**查看镜像大小**
```bash
docker images enlightenlm
```

**清理未使用的构建缓存**
```bash
docker builder prune -f
```

#### 4.2.2.5 生产环境配置

**1. SSL 证书配置**
```bash
# 将证书放入 deploy/ssl/ 目录
cp your-cert.crt deploy/ssl/cert.pem
cp your-key.key deploy/ssl/key.pem
```

**2. 环境变量配置**
```bash
# 生产环境必须设置
DEEPSEEK_API_KEY=your-production-api-key
LOG_LEVEL=WARNING
ENLIGHTEN_MODE=full
```

**3. 资源限制**
```bash
# CPU 模式资源限制
docker run -d \
  --name enlightenlm-api \
  --memory="4g" \
  --cpus="2" \
  -p 8000:8000 \
  --restart unless-stopped \
  --env-file .env \
  enlightenlm:api
```

**4. 健康检查验证**
```bash
# 检查容器健康状态
docker inspect --format='{{.State.Health.Status}}' enlightenlm-api

# 手动触发健康检查
docker exec enlightenlm-api curl -f http://localhost:8000/health
```

### 4.3 集群部署

对于大规模部署，建议使用 Kubernetes 进行容器编排：

#### 4.3.1 Kubernetes 部署清单

**deployment.yaml**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: enlightenlm
  labels:
    app: enlightenlm
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
        env:
        - name: DEEPSEEK_API_KEY
          valueFrom:
            secretKeyRef:
              name: enlightenlm-secrets
              key: deepseek-api-key
        - name: ENLIGHTEN_MODE
          value: "full"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
            nvidia.com/gpu: 1
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: enlightenlm
spec:
  selector:
    app: enlightenlm
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
---
apiVersion: v1
kind: Secret
metadata:
  name: enlightenlm-secrets
type: Opaque
stringData:
  deepseek-api-key: "your-api-key-here"
```

#### 4.3.2 Horizontal Pod Autoscaler (HPA)
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: enlightenlm-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: enlightenlm
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

#### 4.3.3 PodDisruptionBudget
```yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: enlightenlm-pdb
spec:
  minAvailable: 2
  selector:
    matchLabels:
      app: enlightenlm
```

### 4.4 容器化故障排查

#### 4.4.1 常见问题

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| 容器启动失败 | 环境变量未配置 | 检查 .env 文件，确保 DEEPSEEK_API_KEY 已设置 |
| 健康检查失败 | 服务启动慢 | 增加 start_period 时间，或检查日志 `docker-compose logs api` |
| GPU 不可用 | nvidia-docker 未安装 | 安装 nvidia-docker2 并配置 runtime |
| 内存不足 | 资源限制过低 | 调整 docker-compose 中的 mem_limit |
| 端口冲突 | 8000 端口被占用 | 修改端口映射，如 8001:8000 |

#### 4.4.2 调试命令

```bash
# 查看容器日志
docker-compose logs -f api

# 进入容器调试
docker exec -it enlightenlm-api /bin/bash

# 检查容器状态
docker ps -a | grep enlighten

# 检查网络连通性
docker exec enlightenlm-api curl -f http://localhost:8000/health

# 检查资源使用
docker stats

# 重启服务
docker-compose restart api

# 完全重建
docker-compose down -v && docker-compose up -d --build
```

#### 4.4.3 GPU 调试

```bash
# 检查 NVIDIA 驱动
nvidia-smi

# 验证 Docker GPU 支持
docker run --rm --gpus all nvidia/cuda:12.1.0-runtime-ubuntu22.04 nvidia-smi

# 检查容器内 GPU
docker exec enlightenlm-api-gpu nvidia-smi
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
| 容器启动失败 | 环境变量未配置 | 检查 .env 文件，确保 DEEPSEEK_API_KEY 已设置 |
| GPU 不可用 | nvidia-docker 未安装 | 安装 nvidia-docker2 并配置 runtime |

### 6.2 容器环境调试

```bash
# 查看容器日志
docker-compose logs -f api

# 进入容器调试
docker exec -it enlightenlm-api /bin/bash

# 检查容器状态
docker ps -a | grep enlighten

# 检查健康状态
docker inspect --format='{{.State.Health.Status}}' enlightenlm-api

# 检查资源使用
docker stats

# 重启服务
docker-compose restart api
```

### 6.3 调试技巧
- 启用详细日志: `export LOG_LEVEL=DEBUG`
- 检查 GPU 使用情况: `nvidia-smi`
- 验证哈希链: `curl http://localhost:8000/audit/verify`
- 查看威胁检测统计: `curl http://localhost:8000/audit/threat/statistics`

## 7. 安全建议

### 7.1 应用安全
- 定期更新词汇表
- 启用安全隔离
- 配置合适的威胁检测阈值
- 限制 API 访问
- 使用 HTTPS
- 定期备份审计日志

### 7.2 容器安全
- 使用非 root 用户运行容器 (已配置 `USER appuser`)
- 限制容器资源使用
- 定期更新基础镜像
- 启用容器健康检查
- 扫描镜像漏洞: `trivy image enlightenlm:api`

### 7.3 生产环境安全
```bash
# 使用只读文件系统
docker run --read-only --security-opt=no-new-privileges ...

# 限制网络访问
docker network create --internal enlighten-internal
docker network connect enlighten-internal enlightenlm-api

# 定期轮换密钥
kubectl rollout restart deployment enlightenlm
```

## 8. 扩展性

### 8.1 模型替换
可以通过修改 `config/config.yaml` 中的模型名称来替换不同的模型：

```yaml
l1:
  model_name: "meta-llama/Llama-2-7b-chat-hf"
```

### 8.2 Docker 水平扩展
通过增加容器实例数量来提高系统吞吐量：

```bash
# 启动多个 API 实例
docker-compose up -d --scale api=3

# 或使用 docker-compose.yml 中预设的 replicas
```

使用 Nginx 负载均衡（已配置在 deploy/nginx/nginx.conf）自动分发请求到多个容器。

### 8.3 Kubernetes 水平扩展
```bash
# 自动扩缩容
kubectl autoscale deployment enlightenlm --cpu-percent=70 --min=2 --max=10

# 手动扩缩容
kubectl scale deployment enlightenlm --replicas=5
```

### 8.4 垂直扩展
通过升级硬件来提高系统性能：
- 使用更强大的 GPU
- 增加内存
- 使用更快的存储

## 9. 结论

觉悟三层架构提供了一套完整的安全、自指、可审计的大模型推理系统。通过本指南的部署步骤，您可以快速搭建和运行该系统，为您的应用提供安全可靠的 AI 服务。