# EnlightenLM（觉悟三层架构）

安全、自指、可审计的大模型推理系统

## 项目简介

觉悟三层架构是一个用于解决大语言模型在实用化部署中面临的深层问题的系统架构，包括：

- **无限自指递归问题**：通过截断控制器防止模型陷入自我参照循环
- **过程黑盒问题**：通过L2自描述模型生成元描述，提高模型透明度
- **价值观可篡改问题**：通过核心价值观内嵌和保护机制确保价值观安全
- **注意力不可控问题**：通过元注意力控制器引导注意力方向
- **用户定制与安全边界矛盾**：通过安全投影机制在用户定制和安全边界间取得平衡
- **审计不可行问题**：通过哈希链和Merkle树实现可验证的审计日志
- **审计日志数据爆炸问题**：通过压缩和聚合机制优化存储

## 系统架构

系统采用三层架构设计：

1. **L1（基座模型）**：负责生成回答并记录内部状态
2. **L2（自描述小模型）**：基于L1的快照生成元描述，解释推理过程
3. **L3（元注意力控制器）**：生成注意力偏置，管理核心价值观内嵌

此外，还包含截断控制器、审计日志系统和安全隔离模块等辅助组件。

## 核心特性

- **时间解耦**：确保L1生成期间不受上层干扰
- **注意力偏置注入**：通过L3控制注意力，引导模型生成
- **自描述能力**：L2生成元描述，提高模型透明度
- **截断控制**：防止自指递归，避免模型瘫痪
- **审计日志**：可验证的记录，支持第三方审计
- **安全隔离**：保护核心价值观，防止篡改
- **威胁检测**：实时检测和防御各种威胁

## 技术栈

- Python 3.8+
- PyTorch
- Transformers
- FastAPI
- Uvicorn

## 安装

```bash
# 克隆仓库
git clone https://github.com/your-username/EnlightenLM.git
cd EnlightenLM

# 安装依赖
pip install -r requirements.txt
```

## 使用方法

### 启动服务器

```bash
python src/main.py
```

服务器将在 `http://localhost:8000` 启动。

### API 调用

```bash
# 基本推理
curl -X POST http://localhost:8000/inference \
  -H "Content-Type: application/json" \
  -d '{"text": "什么是人工智能？"}'

# 带用户参数的推理
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

## 项目结构

```
EnlightenLM/
├── config/                  # 配置文件
│   ├── config.yaml         # 主配置文件
│   ├── negative_vocab.txt   # 负向词汇表
│   └── positive_vocab.txt  # 正向词汇表
├── src/                    # 源代码
│   ├── l1/                # L1 基座模型
│   │   ├── base_model.py
│   │   └── attention_bias.py
│   ├── l2/                # L2 自描述模型
│   │   ├── self_description_model.py
│   │   └── cutoff_controller.py
│   ├── l3/                # L3 元注意力控制器
│   │   └── meta_attention_controller.py
│   ├── audit/             # 审计模块
│   │   ├── audit_logger.py
│   │   ├── security_isolation.py
│   │   └── threat_detection.py
│   └── main.py            # 主入口
├── tests/                  # 测试文件
├── deployment_guide.md     # 部署指南
├── usage_examples.md       # 使用示例
└── requirements.txt        # 依赖列表
```

## 配置说明

配置文件位于 `config/config.yaml`，包含以下主要部分：

- `l1`: L1 基座模型配置
- `l2`: L2 自描述模型配置
- `l3`: L3 元注意力控制器配置
- `audit`: 审计日志配置
- `api`: API 配置

详细配置说明请参考 [deployment_guide.md](deployment_guide.md)。

## 安全特性

1. **核心价值观保护**：通过词汇表和注意力偏置确保模型遵循核心价值观
2. **威胁检测**：实时检测越狱攻击、注入攻击、隐私侵犯等威胁
3. **安全隔离**：支持进程级、SGX和SEV等多种隔离方案
4. **审计日志**：完整的哈希链和Merkle树验证，确保日志不可篡改

## 部署方式

### 本地开发部署

```bash
python src/main.py
```

### 使用 Gunicorn

```bash
pip install gunicorn uvicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker src.main:app
```

### Docker 部署

```dockerfile
FROM python:3.8

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["python", "src/main.py"]
```

```bash
docker build -t enlightenlm .
docker run -p 8000:8000 --gpus all enlightenlm
```

详细部署说明请参考 [deployment_guide.md](deployment_guide.md)。

## 测试

```bash
# 运行所有测试
pytest tests/

# 运行特定测试
pytest tests/test_attention_bias.py

# 运行安全测试
python tests/test_jailbreak.py
```

## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

MIT License

## 联系方式

- GitHub Issues: https://github.com/your-username/EnlightenLM/issues
