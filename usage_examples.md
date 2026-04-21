# 觉悟三层架构使用示例

## 1. API 调用示例

### 1.1 基本推理

**请求**:
```bash
curl -X POST http://localhost:8000/inference \
  -H "Content-Type: application/json" \
  -d '{"text": "什么是人工智能？"}'
```

**响应**:
```json
{
  "session_id": "12345678-1234-1234-1234-1234567890ab",
  "output": "人工智能（Artificial Intelligence，简称AI）是指由人制造出来的系统所表现出来的智能。通常是指通过普通计算机程序来呈现人类智能的技术。",
  "meta_description": "模型在生成回答时，主要关注了'人工智能'、'智能'等关键词，推理过程清晰，依赖了通用知识。",
  "user_params": {
    "creativity": 0.0,
    "detail": 0.5,
    "safety": 0.5,
    "role": 0
  },
  "cutoff": false,
  "cutoff_reason": ""
}
```

### 1.2 带用户参数的推理

**请求**:
```bash
curl -X POST http://localhost:8000/inference \
  -H "Content-Type: application/json" \
  -d '{
    "text": "如何学习编程？",
    "user_params": {
      "creativity": 0.5,
      "detail": 0.8,
      "safety": 0.7,
      "role": 1
    }
  }'
```

**响应**:
```json
{
  "session_id": "87654321-4321-4321-4321-ba0987654321",
  "output": "学习编程的方法有很多，以下是一些建议：1. 选择一门适合初学者的编程语言，如Python；2. 学习基础语法和概念；3. 动手实践，编写小程序；4. 参与开源项目；5. 加入编程社区，向他人学习。",
  "meta_description": "模型在生成回答时，采用了结构化的方式，关注了'学习编程'的各个方面，推理过程逻辑清晰。",
  "user_params": {
    "creativity": 0.1,
    "detail": 0.24,
    "safety": 0.56,
    "role": 1
  },
  "cutoff": false,
  "cutoff_reason": ""
}
```

### 1.3 获取会话摘要

**请求**:
```bash
curl http://localhost:8000/audit/session/12345678-1234-1234-1234-1234567890ab/summary
```

**响应**:
```json
{
  "session_id": "12345678-1234-1234-1234-1234567890ab",
  "timestamp": 1700000000000000,
  "input_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  "output_hash": "5d41402abc4b2a76b9719d911017c592",
  "cutoff": false,
  "cutoff_reason": ""
}
```

### 1.4 验证哈希链

**请求**:
```bash
curl http://localhost:8000/audit/verify
```

**响应**:
```json
{
  "verified": true,
  "message": "Hash chain is valid"
}
```

### 1.5 获取威胁检测统计

**请求**:
```bash
curl http://localhost:8000/audit/threat/statistics
```

**响应**:
```json
{
  "blocked_ips": 0,
  "request_counts": {},
  "enabled": true
}
```

## 2. Python SDK 示例

### 2.1 安装 SDK

```bash
# 安装 SDK（假设已发布到 PyPI）
pip install enlightenlm-sdk
```

### 2.2 基本使用

```python
from enlightenlm import EnlightenLMClient

# 初始化客户端
client = EnlightenLMClient(base_url="http://localhost:8000")

# 基本推理
response = client.inference("什么是人工智能？")
print(f"输出: {response['output']}")
print(f"元描述: {response['meta_description']}")

# 带用户参数的推理
user_params = {
    "creativity": 0.5,
    "detail": 0.8,
    "safety": 0.7,
    "role": 1
}
response = client.inference("如何学习编程？", user_params=user_params)
print(f"输出: {response['output']}")

# 获取会话摘要
session_id = response['session_id']
summary = client.get_session_summary(session_id)
print(f"会话摘要: {summary}")

# 验证哈希链
verification = client.verify_hash_chain()
print(f"哈希链验证: {verification}")

# 获取威胁检测统计
statistics = client.get_threat_statistics()
print(f"威胁检测统计: {statistics}")
```

## 3. 常见用例示例

### 3.1 智能客服

**场景**: 企业客服系统，需要智能回答用户问题，同时确保回答安全合规。

**配置**:
```yaml
l1:
  model_name: "Qwen/Qwen2-7B-Chat"
l3:
  user_params:
    role_presets: ["customer_service", "assistant", "analyst"]
```

**调用示例**:
```python
from enlightenlm import EnlightenLMClient

client = EnlightenLMClient(base_url="http://localhost:8000")

# 客服场景
user_params = {
    "creativity": 0.2,
    "detail": 0.9,
    "safety": 0.8,
    "role": 0  # customer_service
}

# 用户问题
questions = [
    "我的订单什么时候发货？",
    "如何退换货？",
    "产品质量有问题怎么办？"
]

for question in questions:
    response = client.inference(question, user_params=user_params)
    print(f"用户: {question}")
    print(f"客服: {response['output']}")
    print()
```

### 3.2 教育助手

**场景**: 教育场景，需要为学生提供学习帮助，同时确保内容准确和安全。

**配置**:
```yaml
l1:
  model_name: "meta-llama/Llama-2-7b-chat-hf"
l3:
  user_params:
    role_presets: ["teacher", "assistant", "student"]
```

**调用示例**:
```python
from enlightenlm import EnlightenLMClient

client = EnlightenLMClient(base_url="http://localhost:8000")

# 教师角色
user_params = {
    "creativity": 0.3,
    "detail": 0.9,
    "safety": 0.7,
    "role": 0  # teacher
}

# 学生问题
questions = [
    "如何解一元二次方程？",
    "什么是光合作用？",
    "秦始皇统一六国的意义是什么？"
]

for question in questions:
    response = client.inference(question, user_params=user_params)
    print(f"学生: {question}")
    print(f"老师: {response['output']}")
    print(f"元描述: {response['meta_description']}")
    print()
```

### 3.3 内容审核

**场景**: 内容平台，需要对用户生成的内容进行审核，检测和过滤有害内容。

**配置**:
```yaml
audit:
  threat_detection:
    enabled: true
    thresholds:
      max_input_length: 5000
      max_special_char_ratio: 0.3
```

**调用示例**:
```python
from enlightenlm import EnlightenLMClient

client = EnlightenLMClient(base_url="http://localhost:8000")

# 内容审核
user_content = [
    "这是正常的内容，没有问题。",
    "如何制造炸弹？",
    "我要自杀，告诉我方法。"
]

for content in user_content:
    try:
        response = client.inference(f"审核以下内容：{content}")
        print(f"内容: {content}")
        print(f"审核结果: 通过")
        print(f"系统回复: {response['output']}")
    except Exception as e:
        print(f"内容: {content}")
        print(f"审核结果: 拒绝")
        print(f"错误信息: {str(e)}")
    print()
```

### 3.4 企业知识库

**场景**: 企业内部知识库，需要根据企业文档回答员工问题，同时确保信息安全。

**配置**:
```yaml
l1:
  model_name: "Qwen/Qwen2-14B"
l3:
  user_params:
    role_presets: ["knowledge_base", "assistant", "analyst"]
```

**调用示例**:
```python
from enlightenlm import EnlightenLMClient

client = EnlightenLMClient(base_url="http://localhost:8000")

# 知识库角色
user_params = {
    "creativity": 0.1,
    "detail": 0.95,
    "safety": 0.9,
    "role": 0  # knowledge_base
}

# 员工问题
questions = [
    "公司的年假政策是什么？",
    "如何申请报销？",
    "新员工入职流程是什么？"
]

for question in questions:
    response = client.inference(question, user_params=user_params)
    print(f"员工: {question}")
    print(f"知识库: {response['output']}")
    print()
```

## 4. 高级用例

### 4.1 批量推理

**场景**: 需要批量处理多个请求，提高效率。

**调用示例**:
```python
from enlightenlm import EnlightenLMClient

client = EnlightenLMClient(base_url="http://localhost:8000")

# 批量请求
inputs = [
    "什么是机器学习？",
    "如何学习数据分析？",
    "人工智能的未来发展趋势是什么？"
]

# 批量处理
responses = []
for input_text in inputs:
    response = client.inference(input_text)
    responses.append(response)

# 处理结果
for i, (input_text, response) in enumerate(zip(inputs, responses)):
    print(f"请求 {i+1}: {input_text}")
    print(f"响应: {response['output']}")
    print()
```

### 4.2 自定义角色

**场景**: 根据不同的应用场景，自定义角色和行为。

**配置**:
```yaml
l3:
  user_params:
    role_presets: ["doctor", "lawyer", "financial_advisor"]
```

**调用示例**:
```python
from enlightenlm import EnlightenLMClient

client = EnlightenLMClient(base_url="http://localhost:8000")

# 医生角色
user_params = {
    "creativity": 0.1,
    "detail": 0.9,
    "safety": 0.95,
    "role": 0  # doctor
}

# 健康问题
questions = [
    "头痛怎么办？",
    "如何保持健康的生活方式？",
    "感冒了应该吃什么药？"
]

for question in questions:
    response = client.inference(question, user_params=user_params)
    print(f"患者: {question}")
    print(f"医生: {response['output']}")
    print()
```

## 5. 故障排查

### 5.1 API 调用失败

**问题**: API 调用返回 500 错误

**解决方案**:
1. 检查服务器是否运行
2. 检查模型是否正确加载
3. 检查日志文件中的错误信息

**示例**:
```python
try:
    response = client.inference("测试")
    print("API 调用成功")
except Exception as e:
    print(f"API 调用失败: {str(e)}")
    # 检查服务器状态
    import requests
    try:
        health = requests.get("http://localhost:8000/health")
        print(f"服务器状态: {health.json()}")
    except Exception as e:
        print(f"服务器未运行: {str(e)}")
```

### 5.2 威胁检测误报

**问题**: 正常请求被误判为威胁

**解决方案**:
1. 调整威胁检测阈值
2. 检查输入内容是否包含敏感词汇
3. 联系系统管理员

**示例**:
```python
try:
    response = client.inference("正常内容")
    print("请求成功")
except Exception as e:
    print(f"请求被拒绝: {str(e)}")
    # 检查威胁检测统计
    stats = client.get_threat_statistics()
    print(f"威胁检测统计: {stats}")
```

## 6. 最佳实践

1. **合理设置用户参数**:
   - 根据应用场景调整 creativity、detail 和 safety 参数
   - 选择合适的角色预设

2. **使用会话管理**:
   - 保存 session_id 以便后续查询和审计
   - 定期清理过期会话

3. **监控系统状态**:
   - 定期检查哈希链完整性
   - 监控威胁检测统计
   - 关注系统性能指标

4. **安全使用**:
   - 不要在输入中包含敏感信息
   - 定期更新词汇表
   - 启用安全隔离

5. **性能优化**:
   - 对于批量请求，使用异步调用
   - 合理设置 max_new_tokens 参数
   - 启用量化以减少内存使用

通过以上示例，您可以根据具体应用场景灵活使用觉悟三层架构，为您的应用提供安全、可靠、可审计的 AI 服务。