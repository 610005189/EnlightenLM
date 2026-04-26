# EnlightenLM 项目发布指南

## 完成的工作

### 1. 项目结构准备

✅ 创建了 `pyproject.toml` 配置文件
✅ 更新了 `enlighten/__init__.py` 导出正确的模块
✅ 添加了 `api_server.py` 中的 `main()` 入口函数
✅ 创建了 `.gitignore` 过滤构建文件

### 2. 包配置

**包名**: `enlightenlm`  
**版本**: `2.3.0`  
**核心依赖**: `numpy>=1.20`, `loguru>=0.7.0`

**可选依赖**:
- `api`: FastAPI 服务依赖
- `local`: 本地模型运行依赖
- `security`: 安全增强依赖
- `full`: 所有可选依赖

### 3. 命令行工具

```bash
# 安装后可用
enlightenlm-server --host 0.0.0.0 --port 8000
```

### 4. 已构建的文件

```
dist/
├── enlightenlm-2.3.0-py3-none-any.whl
└── enlightenlm-2.3.0.tar.gz
```

## 本地测试

### 开发模式安装

```bash
# 在项目根目录执行
pip install -e .
```

### 测试导入

```python
import enlighten
print(enlighten.__version__)  # 2.3.0

from enlighten import HybridEnlightenLM, load_config
```

### 测试构建

```bash
# 安装 build 工具
pip install --upgrade build

# 构建包
python -m build
```

## PyPI 发布步骤

### 步骤 1: 注册 PyPI 账号

访问 https://pypi.org 注册账号并验证邮箱。

### 步骤 2: 获取 API Token

1. 登录 PyPI 后，访问 Account Settings
2. 在 "API tokens" 部分点击 "Add API token"
3. 输入 Token 名称 (如 "enlightenlm-release")
4. 选择 "Scope: Entire account"
5. 保存 Token (只能查看一次！)

### 步骤 3: 安装 twine

```bash
pip install --upgrade twine
```

### 步骤 4: 上传到 TestPyPI (推荐先测试)

```bash
# 使用 API Token 登录
# 用户名: __token__
# 密码: <你的 API Token>

twine upload --repository testpypi dist/*
```

访问 https://test.pypi.org/project/enlightenlm/ 验证发布是否成功。

### 步骤 5: 测试安装 TestPyPI 包

```bash
pip install --index-url https://test.pypi.org/simple/ enlightenlm
```

### 步骤 6: 发布到正式 PyPI

```bash
twine upload dist/*
```

访问 https://pypi.org/project/enlightenlm/ 验证发布。

## 用户安装指南

### 基本安装

```bash
pip install enlightenlm
```

### 带可选依赖

```bash
# 安装 API 服务依赖
pip install "enlightenlm[api]"

# 安装本地模型依赖
pip install "enlightenlm[local]"

# 安全增强
pip install "enlightenlm[security]"

# 完整安装
pip install "enlightenlm[full]"
```

### 使用示例

```python
from enlighten import HybridEnlightenLM, load_config

# 加载配置
config = load_config("balanced")

# 初始化模型
model = HybridEnlightenLM(
    use_local_model=False,  # 使用 DeepSeek API
    config=config
)

# 生成
result = model.generate(prompt="Hello world!")
print(result.text)
```

### 启动 API 服务

```bash
# 设置 API Key (可选)
$env:DEEPSEEK_API_KEY = "your-api-key"

# 启动服务
enlightenlm-server --host 0.0.0.0 --port 8000
```

## 重要提醒

### 1. PyPI 包名检查

在正式发布前，请确认 `enlightenlm` 包名在 PyPI 上未被占用：
https://pypi.org/project/enlightenlm/

如果包名已被占用，请考虑以下名称：
- `enlighten-lm`
- `enlighten-ai`
- `enlighten-core`

### 2. 更新版本号

在 `pyproject.toml` 和 `enlighten/__init__.py` 中同步更新版本号。

### 3. 发布 Checklist

- [ ] 所有测试通过
- [ ] 文档更新完毕
- [ ] 版本号正确
- [ ] 先在 TestPyPI 测试
- [ ] 正式发布
- [ ] 验证安装成功
- [ ] 更新 GitHub Release

## 问题排查

### 问题: 导入错误

```
ModuleNotFoundError: No module named 'enlighten'
```

**解决**: 确保安装了包 (`pip install enlightenlm`)，然后使用 `import enlighten`。

### 问题: 命令行工具找不到

```
'enlightenlm-server' is not recognized...
```

**解决**: 确保 Scripts 目录在 PATH 中，或使用 `python -m pip install enlightenlm` 重新安装。

### 问题: 构建时权限错误

**解决**: 确保不在系统 Python 环境中，使用虚拟环境或 `--user` 安装。
