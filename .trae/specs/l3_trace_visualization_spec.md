# EnlightenLM PyPI 发布计划

## 项目现状分析

**当前状态**: 项目有完整的核心功能，API模式运行正常，文档完善

**需要处理**:

* [ ] 缺少 pyproject.toml

* [ ] 缺少包版本管理

* [ ] 需要清理开发文件

* [ ] 需要配置打包和发布流程

***

## 实施步骤

### 1. 准备项目结构

**需要检查/清理**:

* [ ] 确定实际运行的包入口（hybrid\_architecture.py vs 骨架代码）

* [ ] 移除不必要的开发文件（测试报告、.pytest\_cache、logs、EnlightenLM\_Quick\_Test）

* [ ] 整理 docs 目录

* [ ] 准备 PyPI 描述（README 摘要）

### 2. 编写 pyproject.toml

**内容规划**:

* [ ] 包名: `enlightenlm` 或 `enlighten-lm`

* [ ] 版本: 从 2.3.0 开始

* [ ] 依赖（从 requirements.txt 转换）

* [ ] 可选依赖（extra）

  * `api`: FastAPI, Uvicorn

  * `local`: Transformers, Torch

  * `full`: 所有依赖

* [ ] console scripts

* [ ] 项目元数据（description, author, license, keywords）

### 3. 更新 __init__.py

* [ ] 导出实际运行的核心类

* [ ] 版本号更新为 2.3.0

### 4. 创建构建脚本

* [ ] `setup.py` 或使用 pyproject.toml 的 setuptools 配置

* [ ] `.gitignore` 更新（dist, build, \*.egg-info）

* [ ] MANIFEST.in（包含 docs 等）

### 5. 测试构建

* [ ] 运行 `pip install .`

* [ ] 测试包导入和基本功能

* [ ] 检查依赖

### 6. PyPI 上传

* [ ] 注册账号 / 登录

* [ ] 使用 twine 上传

* [ ] 验证发布结果

***

## 项目结构最终规划

```
enlightenlm/
├── enlighten/                    # 主包
│   ├── __init__.py
│   ├── hybrid_architecture.py   # 核心功能
│   ├── api_server.py            # API服务器
│   ├── config/                  # 配置
│   │   ├── __init__.py
│   │   └── modes.py
│   └── api/                     # API客户端
│       ├── __init__.py
│       └── deepseek_client.py
├── docs/                        # 文档（打包为 package_data）
├── tests/                       # 测试（不包含在发布包中）
├── pyproject.toml
├── README.md
└── LICENSE
```

**打包内容**:

* ✅ `enlighten/` 包

* ✅ 文档（可选）

* ❌ `tests/` 目录（单独作为 sdist 可选）

* ❌ 开发工具文件

***

## 依赖规划

**核心依赖（必选）**:

* numpy

* loguru

* pydantic

**可选依赖**:

* `api`: fastapi, uvicorn

* `local`: transformers, torch

* `security`: pycryptodome, cryptography

* `full`: 所有

***

## 注意事项

1. 包名需要检查是否可用（pypi.org 查询）
2. 需要作者/维护者信息
3. 确定是否需要上传到 PyPI 或 TestPyPI 先测试
4. 版本号采用语义化（MAJOR.MINOR.PATCH）

