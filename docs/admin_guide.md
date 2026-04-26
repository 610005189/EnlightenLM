# EnlightenLM 管理员指南

> 版本: v3.0
> 更新日期: 2026-04-26
> 状态: 正式版

---

## 1. 系统架构

### 1.1 三层架构

EnlightenLM 采用模型无关的三层架构：

- **L1 生成层**：负责与底层模型交互，支持 Ollama、DeepSeek API 等
- **L2 工作记忆层**：负责会话管理、注意力统计和熵值追踪
- **L3 安全监控层**：负责敏感内容检测、自指循环检测和幻觉风险评估

### 1.2 核心组件

| 组件 | 功能 | 位置 |
|------|------|------|
| HybridEnlightenLM | 核心推理引擎 | `enlighten/hybrid_architecture.py` |
| L3Controller | 安全监控控制器 | `enlighten/l3_controller.py` |
| EntropyTracker | 熵值监控 | `enlighten/memory/entropy_tracker.py` |
| TEEAuditWriter | 审计链管理 | `enlighten/audit/tee_audit.py` |
| API 服务器 | REST API 和 WebSocket | `enlighten/api_server.py` |

---

## 2. 部署指南

### 2.1 本地部署

**步骤**：
1. **环境准备**
   ```bash
   # 安装依赖
   pip install -r requirements.txt
   
   # 安装 Prometheus 依赖（监控）
   pip install prometheus-client
   ```

2. **配置文件**
   - 编辑 `configs/balanced.yaml`
   - 设置 `model_name` 和 `model_type`

3. **启动服务**
   ```bash
   # 启动 Ollama 服务
   ollama serve
   
   # 启动 API 服务器
   python -m enlighten.api_server
   ```

### 2.2 Docker 部署

**步骤**：
1. **构建镜像**
   ```bash
   docker build -t enlightenlm .
   ```

2. **启动容器**
   ```bash
   docker-compose up -d
   ```

3. **访问服务**
   - Web 界面：`http://localhost:8000/chat.html`
   - API 接口：`http://localhost:8000/inference`
   - 监控：`http://localhost:9090` (Prometheus)
   - 仪表板：`http://localhost:3000` (Grafana)

---

## 3. 配置管理

### 3.1 配置文件结构

**主要配置文件**：
- `configs/balanced.yaml`：平衡模式
- `configs/full.yaml`：完整功能模式
- `configs/lightweight.yaml`：轻量级模式

**核心配置参数**：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `model_name` | 模型名称 | `llama3:latest` |
| `model_type` | 模型类型 | `ollama` |
| `use_l1_adapter` | 是否启用 L1 适配器 | `false` |
| `use_l2_adapter` | 是否启用 L2 适配器 | `true` |
| `use_l3_controller` | 是否启用 L3 控制器 | `true` |
| `use_bayesian_l3` | 是否启用贝叶斯 L3 | `true` |
| `van_sensitivity` | VAN 敏感词检测灵敏度 | `0.7` |
| `hallucination_threshold` | 幻觉风险阈值 | `0.7` |

### 3.2 安全配置

**安全相关参数**：
- `van_sensitivity`：敏感词检测灵敏度 (0-1)
- `self_reference_threshold`：自指循环检测阈值
- `max_repetition_ratio`：最大词汇重复比例
- `cooling_window_seconds`：冷却时间窗口
- `entropy_threshold`：熵异常检测阈值
- `variance_threshold`：熵方差异常阈值

**配置示例**：
```yaml
# 高安全模式
van_sensitivity: 0.9
self_reference_threshold: 0.7
max_repetition_ratio: 0.2
cooling_window_seconds: 30

# 平衡模式
van_sensitivity: 0.7
self_reference_threshold: 0.8
max_repetition_ratio: 0.3
cooling_window_seconds: 60

# 低安全模式
van_sensitivity: 0.5
self_reference_threshold: 0.9
max_repetition_ratio: 0.4
cooling_window_seconds: 120
```

---

## 4. 监控与日志

### 4.1 Prometheus 监控

**指标端点**：`http://localhost:8001/metrics`

**主要指标**：
- `enlightenlm_requests_total`：总请求数
- `enlightenlm_errors_total`：总错误数
- `enlightenlm_van_events_total`：VAN 事件数
- `enlightenlm_cutoffs_total`：截断事件数
- `enlightenlm_response_time_seconds_avg`：平均响应时间
- `enlightenlm_entropy_mean`：熵均值
- `enlightenlm_cpu_usage_percent`：CPU 使用率
- `enlightenlm_memory_usage_percent`：内存使用率

### 4.2 Grafana 仪表板

**访问地址**：`http://localhost:3000`

**默认仪表板**：`EnlightenLM 监控`

**仪表板包含**：
- API 请求率
- 响应时间
- VAN 事件率
- 截断率
- 熵均值
- 审计事件率
- CPU 使用率
- 内存使用率
- 错误率

### 4.3 日志管理

**日志位置**：`logs/` 目录

**日志类型**：
- `api_server.log`：API 服务器日志
- `audit.log`：审计日志
- `error.log`：错误日志

**日志级别**：
- DEBUG：详细调试信息
- INFO：普通信息
- WARNING：警告信息
- ERROR：错误信息
- CRITICAL：严重错误

---

## 5. 安全管理

### 5.1 敏感词库

**词库位置**：`configs/core_rules.yaml`

**词库分类**：
- 暴力犯罪
- 色情内容
- 隐私信息
- 歧视偏见
- 危险建议

**更新词库**：
1. 编辑 `configs/core_rules.yaml`
2. 重启服务：`python -m enlighten.api_server`

### 5.2 审计链管理

**审计链功能**：
- 哈希链记录
- HMAC 签名
- 完整性验证

**审计日志查询**：
- API：`GET /api/v1/audit/logs?session_id=session_123`
- 本地文件：`logs/audit.log`

**验证审计链**：
- API：`GET /audit/verify`
- 代码：`model.audit_writer.verify_chain()`

### 5.3 安全事件处理

**VAN 事件**：
- 敏感词检测
- 自指循环检测
- 词汇重复检测

**处理流程**：
1. 检测到安全事件
2. 触发冷却机制
3. 记录审计日志
4. 向客户端返回截断信息

---

## 6. 性能优化

### 6.1 硬件优化

**CPU**：
- 推荐 4+ 核心 CPU
- 支持 AVX2 指令集

**内存**：
- 7B 模型：8GB+ 内存
- 14B 模型：16GB+ 内存
- 70B 模型：32GB+ 内存

**存储**：
- SSD 存储推荐
- 10GB+ 磁盘空间

### 6.2 软件优化

**模型选择**：
- 快速响应：`llama3:8b` 或 `qwen2.5:7b`
- 平衡模式：`llama3:70b` 或 `qwen2.5:14b`
- 高质量：`llama3:70b` 或 `claude-3-opus`

**配置优化**：
- 调整 `memory_size` 适应硬件
- 调整 `max_concurrent_requests` 控制并发
- 启用会话缓存减少重复计算

**部署策略**：
- 使用 Docker 容器化部署
- 配置自动缩放
- 启用负载均衡

---

## 7. 故障排查

### 7.1 常见问题

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| 模型无响应 | Ollama 服务未启动 | 启动 Ollama 服务：`ollama serve` |
| API 调用失败 | 模型未安装 | 安装模型：`ollama pull llama3:latest` |
| 内容被截断 | 敏感词检测触发 | 调整安全配置 |
| 响应时间长 | 模型加载慢 | 使用较小的模型 |
| 内存使用高 | 模型占用大 | 调整 `memory_size` 参数 |
| 审计链验证失败 | 数据被篡改 | 检查系统安全 |

### 7.2 日志分析

**错误日志示例**：
```
2026-04-26 10:00:00 ERROR: Ollama API 调用失败: 404 Not Found
2026-04-26 10:00:30 ERROR: 模型未找到: llama3:latest
```

**审计日志示例**：
```
2026-04-26 10:00:00 INFO: 会话 session_123 开始
2026-04-26 10:00:05 INFO: 生成完成，tokens: 150
2026-04-26 10:00:05 INFO: 审计链验证通过
```

### 7.3 健康检查

**API 健康检查**：
- 端点：`GET /health`
- 预期响应：`{"status": "healthy", "model": "llama3:latest"}`

**Ollama 健康检查**：
- 命令：`ollama status`
- 预期输出：`Ollama is running`

**系统资源检查**：
- CPU：`top` 或 `htop`
- 内存：`free -h`
- 磁盘：`df -h`

---

## 8. 扩展与定制

### 8.1 添加新模型

**步骤**：
1. 安装模型：`ollama pull model_name`
2. 修改配置：`configs/balanced.yaml`
3. 重启服务：`python -m enlighten.api_server`

**支持的模型**：
- `llama3:*`
- `qwen2.5:*`
- `mistral:*`
- `gemma:*`
- `claude-3-*`

### 8.2 自定义安全规则

**步骤**：
1. 编辑 `configs/core_rules.yaml`
2. 添加新的敏感词规则
3. 重启服务

**规则格式**：
```yaml
sensitive_categories:
  violence:
    - "武器制作"
    - "攻击方法"
  pornography:
    - "色情"
    - "成人内容"
```

### 8.3 开发新功能

**开发流程**：
1. 创建功能分支：`git checkout -b feature/new-feature`
2. 实现功能
3. 编写测试：`tests/test_new_feature.py`
4. 运行测试：`pytest tests/test_new_feature.py`
5. 提交代码：`git commit -m "feat: add new feature"`
6. 创建 PR

**代码规范**：
- 遵循 `black` 代码格式
- 使用 `isort` 整理导入
- 添加类型注解
- 编写文档字符串

---

## 9. 版本管理

### 9.1 版本号格式

`v<主版本>.<次版本>.<补丁版本>`

- **主版本**：重大架构变更
- **次版本**：新功能添加
- **补丁版本**：bug 修复

### 9.2 发布流程

**步骤**：
1. 更新版本号：`pyproject.toml`
2. 更新 CHANGELOG.md
3. 运行测试：`pytest tests/`
4. 构建发布：`python -m build`
5. 上传发布：`twine upload dist/*`

### 9.3 升级指南

**从 v2.x 升级到 v3.0**：
1. 备份配置文件
2. 更新代码：`git pull`
3. 更新依赖：`pip install -r requirements.txt`
4. 更新配置文件格式
5. 重启服务

---

## 10. 安全最佳实践

### 10.1 部署安全

- **网络隔离**：将服务部署在受保护的网络区域
- **API 认证**：启用 API 密钥认证
- **HTTPS**：使用 TLS 加密
- **防火墙**：配置适当的防火墙规则
- **定期更新**：定期更新模型和依赖

### 10.2 运营安全

- **监控告警**：配置 Prometheus 告警
- **日志分析**：定期分析安全日志
- **漏洞扫描**：定期进行安全扫描
- **安全审计**：定期进行安全审计
- **应急响应**：建立安全事件响应流程

### 10.3 数据安全

- **数据加密**：加密敏感数据
- **数据保留**：设置合理的数据保留策略
- **数据脱敏**：对敏感信息进行脱敏
- **访问控制**：严格的访问控制
- **合规性**：遵守相关数据保护法规

---

## 11. 联系支持

### 11.1 技术支持

- **GitHub Issues**：https://github.com/610005189/EnlightenLM/issues
- **邮件**：support@enlightenlm.com
- **Discord**：https://discord.gg/enlightenlm

### 11.2 贡献指南

- **提交 Issue**：详细描述问题
- **提交 PR**：遵循代码规范
- **文档贡献**：改进文档
- **测试贡献**：添加测试用例

---

**© 2026 EnlightenLM 团队**