# EnlightenLM CI/CD 指南

## GitHub Actions 工作流

本项目使用 GitHub Actions 实现自动化 CI/CD。

### 工作流文件

```
.github/
├── workflows/
│   ├── ci.yml           # 持续集成
│   └── release.yml      # 发布到 PyPI
└── ISSUES_TEMPLATE/
    ├── bug_report_template.md
    ├── feature_request_template.md
    └── pull_request_template.md
```

---

## CI 工作流 (ci.yml)

### 触发条件

- Push 到 `main` 或 `develop` 分支
- Pull Request 到 `main` 或 `develop` 分支

### 测试矩阵

- Python 版本: 3.8, 3.9, 3.10, 3.11
- 操作系统: Ubuntu Latest

### 工作流程

#### 1. Test Job

```yaml
- 检出代码
- 设置 Python 环境
- 缓存 pip 依赖
- 安装依赖: pip install -e ".[full]"
- 运行测试: pytest tests/ -v --cov=enlighten
- 上传覆盖率到 Codecov
```

#### 2. Lint Job

```yaml
- 检出代码
- 设置 Python 3.10
- 安装代码检查工具: flake8, black, mypy
- 运行代码检查:
  - flake8: 检查致命错误
  - black: 检查代码格式
  - mypy: 类型检查
```

#### 3. Build Job

```yaml
- 依赖: test + lint jobs
- 检出代码
- 设置 Python
- 构建包: python -m build
- 上传构建产物
```

---

## Release 工作流 (release.yml)

### 触发方式

手动触发 (workflow_dispatch)

### 输入参数

| 参数 | 描述 | 必填 | 默认值 |
|------|------|------|--------|
| version | 版本号 (如 2.3.0) | 是 | - |
| environment | 目标环境 | 是 | test |

### 工作流程

#### 1. Release Job

根据 `environment` 选择目标:

- **Test PyPI** (test):
  ```bash
  twine upload --repository testpypi dist/*
  ```

- **PyPI** (prod):
  ```bash
  twine upload dist/*
  ```

#### 2. Verify Job

安装并验证发布的包:
```bash
pip install enlightenlm==<version>
python -c "import enlighten; print(enlighten.__version__)"
```

---

## 使用方法

### 本地测试

```bash
# 安装开发依赖
pip install -e ".[full]"

# 运行测试
pytest tests/ -v

# 代码检查
flake8 enlighten tests
black --check enlighten tests
mypy enlighten

# 构建包
python -m build
```

### 设置 GitHub Secrets

在 GitHub 仓库设置以下 secrets:

| Secret 名称 | 描述 |
|------------|------|
| `PYPI_TOKEN` | PyPI API Token |
| `TEST_PYPI_TOKEN` | Test PyPI API Token |

### 创建 PyPI Token

1. 登录 https://pypi.org
2. 进入 Account Settings → API tokens
3. 点击 "Add API token"
4. 设置 token name 和 scope
5. 复制生成的 token

### 发布新版本

1. 确保所有测试通过
2. 更新版本号:
   ```bash
   # pyproject.toml
   version = "2.3.0"
   
   # enlighten/__init__.py
   __version__ = "2.3.0"
   ```
3. 创建 Git tag:
   ```bash
   git tag v2.3.0
   git push origin v2.3.0
   ```
4. 在 GitHub Actions 页面手动触发 `Release to PyPI` workflow
5. 填写版本号和目标环境 (先选择 `test`)
6. 验证 Test PyPI 安装成功
7. 重新触发，选择 `prod` 发布到正式 PyPI

---

## Badge 徽章

在 README 中添加以下徽章:

```markdown
[![CI](https://github.com/610005189/enlightenlm/actions/workflows/ci.yml/badge.svg)](https://github.com/610005189/enlightenlm/actions)
[![PyPI version](https://img.shields.io/pypi/v/enlightenlm.svg)](https://pypi.org/project/enlightenlm/)
[![Python versions](https://img.shields.io/pypi/pyversions/enlightenlm.svg)](https://pypi.org/project/enlightenlm/)
```

---

## 维护清单

### 发布前检查

- [ ] 所有测试通过
- [ ] 代码覆盖率高
- [ ] 无 flake8/black/mypy 错误
- [ ] 文档已更新
- [ ] CHANGELOG 已更新
- [ ] 版本号已更新

### 发布后检查

- [ ] Test PyPI 安装验证成功
- [ ] PyPI 安装验证成功
- [ ] GitHub Release 已创建
- [ ] GitHub Actions 运行成功

---

## 故障排除

### CI 构建失败

1. 检查 Python 版本兼容性
2. 查看 GitHub Actions 日志
3. 本地运行相同的测试命令

### PyPI 发布失败

1. 验证 Token 权限
2. 检查包名是否冲突
3. 确认版本号未使用

### 覆盖率上传失败

- Codecov 集成是可选的，失败不影响 CI 状态
