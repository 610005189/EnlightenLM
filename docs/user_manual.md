# EnlightenLM 用户手册

> 版本: v3.0
> 更新日期: 2026-04-26
> 状态: 正式版

---

## 1. 系统概述

EnlightenLM 是一个**模型无关**的三层推理安全框架，为大语言模型提供实时安全监控、幻觉截断与偏见缓解功能。

### 1.1 核心优势

- **安全保障**：实时检测并拦截有害内容、自指循环和幻觉
- **模型无关**：支持 Ollama、DeepSeek API 等多种模型后端
- **可解释性**：提供详细的安全监控数据和审计链
- **实时监控**：通过 WebSocket 实现实时监控数据传输
- **易于部署**：提供完整的 Docker 部署方案

### 1.2 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    安全监控层 (L3)                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ 偏见检测    │  │ 幻觉风险    │  │ 敏感词/自指检测    │  │
│  │ 词库+探针   │  │ MLP判别器   │  │ 正则+AC自动机      │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                    信号提取层 (L2)                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ 熵值追踪   │  │ 重复率分析   │  │ 置信度监控         │  │
│  │ 后验熵 H_t │  │ n-gram频率   │  │ max(p), margin_t  │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                    模型调用层 (L1)                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Ollama      │  │ DeepSeek    │  │ vLLM               │  │
│  │ 本地模型    │  │ API         │  │ 高性能推理          │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 快速开始

### 2.1 环境要求

- Python 3.8+
- Ollama 服务（推荐本地部署）
- 8GB+ 内存
- 10GB+ 磁盘空间

### 2.2 安装步骤

1. **克隆仓库**
   ```bash
   git clone https://github.com/610005189/EnlightenLM.git
   cd EnlightenLM
   ```

2. **创建虚拟环境**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # 或 venv\Scripts\activate  # Windows
   ```

3. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

4. **启动 Ollama 服务**
   ```bash
   ollama serve
   ```

5. **安装模型**
   ```bash
   ollama pull llama3:latest
   ```

6. **启动 API 服务器**
   ```bash
   python -m enlighten.api_server
   ```

7. **访问 Web 界面**
   打开浏览器访问：`http://localhost:8000/chat.html`

---

## 3. Web 界面使用

### 3.1 聊天界面

1. **输入框**：在底部输入框中输入您的问题
2. **发送按钮**：点击发送按钮或按 Enter 键发送消息
3. **对话历史**：左侧显示对话历史
4. **L3 监控栏**：右侧显示实时监控数据

### 3.2 L3 监控栏

L3 监控栏显示以下信息：

- **置信度分析**：模型生成内容的置信度
- **贝叶斯分析**：幻觉风险评估
- **温度控制**：实时温度调整
- **系统日志**：安全事件记录

### 3.3 示例对话

```
用户: 请解释什么是人工智能

模型: 人工智能（Artificial Intelligence, AI）是指计算机系统模拟人类智能行为的能力...

[L3 监控] 置信度: 0.92, 幻觉风险: 低, 温度: 0.7
```

---

## 4. API 使用

### 4.1 推理接口

**端点**：`POST /inference`

**请求体**：
```json
{
  "text": "请解释什么是人工智能",
  "max_length": 500,
  "session_id": "optional_session_id"
}
```

**响应**：
```json
{
  "session_id": "session_123",
  "output": "人工智能是...",
  "tokens": 150,
  "meta_description": "Ollama API (llama3:latest) | L3安全监控通过",
  "cutoff": false,
  "van_event": false,
  "security_verified": true,
  "entropy_stats": {
    "mean": 0.65,
    "variance": 0.03,
    "trend": 0.02
  }
}
```

### 4.2 安全配置接口

**获取配置**：`GET /api/v1/safety/config`

**更新配置**：`PUT /api/v1/safety/config`

**请求体**：
```json
{
  "van_sensitivity": 0.7,
  "hallucination_threshold": 0.7
}
```

### 4.3 审计日志接口

**获取审计日志**：`GET /api/v1/audit/logs?session_id=session_123`

---

## 5. 模型管理

### 5.1 支持的模型

- **本地模型**：通过 Ollama 提供
  - llama3:latest
  - qwen2.5:7b
  - mistral:latest

- **API 模型**：
  - DeepSeek API
  - 其他兼容 OpenAI API 的服务

### 5.2 切换模型

1. **修改配置文件**：`configs/balanced.yaml`
2. **重启服务**：`python -m enlighten.api_server`

---

## 6. 常见问题

### 6.1 模型无响应

**可能原因**：
- Ollama 服务未启动
- 模型未正确安装
- 网络连接问题

**解决方案**：
- 检查 Ollama 服务状态：`ollama status`
- 重新安装模型：`ollama pull llama3:latest`
- 检查网络连接

### 6.2 内容被截断

**可能原因**：
- 触发了敏感词检测
- 检测到自指循环
- 幻觉风险过高

**解决方案**：
- 调整问题表述
- 避免敏感话题
- 分解复杂问题

### 6.3 响应时间过长

**可能原因**：
- 模型加载时间长
- 硬件性能限制
- 并发请求过多

**解决方案**：
- 使用较小的模型（如 7B 模型）
- 增加硬件资源
- 减少并发请求

---

## 7. 高级功能

### 7.1 自定义安全配置

编辑 `configs/custom.yaml` 文件：

```yaml
van_sensitivity: 0.7
self_reference_threshold: 0.8
hallucination_threshold: 0.7
```

### 7.2 启用 WebSocket 实时监控

WebSocket 端点：`ws://localhost:8000/ws/l3/stats`

**使用示例**：
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/l3/stats');
ws.onmessage = function(event) {
  const data = JSON.parse(event.data);
  console.log('L3 监控数据:', data);
};
```

### 7.3 审计链验证

**端点**：`GET /audit/verify`

**响应**：
```json
{
  "verified": true,
  "message": "VAN events: 0, Blocked: 0",
  "details": {
    "van_events": 0,
    "blocked_requests": 0,
    "block_ratio": 0
  }
}
```

---

## 8. 性能优化

### 8.1 模型选择

- **快速响应**：使用 `llama3:8b` 或 `qwen2.5:7b`
- **高质量**：使用 `llama3:70b` 或 `qwen2.5:14b`

### 8.2 配置优化

- **内存限制**：根据硬件调整 `memory_size` 参数
- **并发限制**：调整 `max_concurrent_requests` 参数
- **缓存策略**：启用会话缓存减少重复计算

---

## 9. 安全注意事项

- **敏感信息**：不要在对话中输入个人敏感信息
- **内容审核**：系统会自动审核生成内容
- **合规使用**：遵守相关法律法规
- **定期更新**：定期更新模型和安全规则

---

## 10. 联系支持

- **GitHub Issues**：https://github.com/610005189/EnlightenLM/issues
- **文档**：https://github.com/610005189/EnlightenLM/tree/main/docs
- **邮件**：support@enlightenlm.com

---

**© 2026 EnlightenLM 团队**