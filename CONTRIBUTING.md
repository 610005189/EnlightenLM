# 贡献指南

> 感谢你对 EnlightenLM 的关注！
> 欢迎提交 Issue 和 Pull Request。

---

## 📋 目录

1. [行为准则](#行为准则)
2. [快速开始](#快速开始)
3. [开发环境](#开发环境)
4. [开发流程](#开发流程)
5. [代码规范](#代码规范)
6. [提交规范](#提交规范)
7. [测试要求](#测试要求)
8. [文档要求](#文档要求)

---

## 行为准则

我们期望所有贡献者遵守以下原则：

- **尊重**：尊重他人的观点和贡献
- **包容**：欢迎不同背景的参与者
- **专业**：保持专业和建设性的沟通
- **责任**：对自己的行为负责

---

## 快速开始

### Fork 项目

```bash
# 1. Fork 本仓库
# 2. 克隆你的 fork
git clone https://github.com/YOUR_USERNAME/EnlightenLM.git
cd EnlightenLM

# 3. 添加上游仓库
git remote add upstream https://github.com/610005189/EnlightenLM.git
```

### 创建开发分支

```bash
# 基于 main 创建功能分支
git checkout main
git pull upstream main
git checkout -b feature/your-feature-name

# 或基于当前 release 分支
git checkout -b hotfix/issue-description v2.1.0
```

---

## 开发环境

### 环境要求

| 工具 | 版本要求 |
|------|---------|
| Python | 3.8+ |
| Git | 2.0+ |
| PyTorch | 2.0+ |
| transformers | 4.30+ |

### 安装开发依赖

```bash
# 克隆后安装
pip install -e ".[dev]"

# 或安装所有依赖
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 验证安装

```bash
python -c "from enlighten import L1Generation, L2WorkingMemory, L3Controller; print('OK')"
```

---

## 开发流程

### 1. 选择任务

- 查看 [Issues](https://github.com/610005189/EnlightenLM/issues)
- 查看 [Project Board](https://github.com/610005189/EnlightenLM/projects)
- 认领任务或创建新 Issue 讨论

### 2. 开发阶段

```bash
# 1. 创建分支
git checkout -b feature/your-feature

# 2. 开发 & 测试
python -m pytest tests/

# 3. 提交 (遵循提交规范)
git add .
git commit -m "feat(module): add new feature"

# 4. 同步上游
git fetch upstream
git rebase upstream/main

# 5. 推送
git push origin feature/your-feature
```

### 3. 提交 Pull Request

1. Push 后在 GitHub 创建 PR
2. 填写 PR 模板
3. 关联相关 Issue
4. 等待 Code Review

### 4. Code Review 反馈

- 及时响应反馈
- 不要在 PR 中 force push
- 保持提交历史整洁

---

## 代码规范

### Python 代码规范

遵循 [PEP 8](https://pep8.org/)：

```bash
# 使用 black 格式化
black enlighten/

# 使用 flake8 检查
flake8 enlighten/

# 类型检查 (建议)
mypy enlighten/
```

### 命名规范

| 类型 | 规范 | 示例 |
|------|------|------|
| 模块 | 小写下划线 | `attention_dan.py` |
| 类 | 大驼峰 | `DANAttention` |
| 函数 | 小写下划线 | `forward()` |
| 常量 | 大写下划线 | `MAX_TOKENS` |
| 私有 | 下划线前缀 | `_private_method()` |

### 文档字符串

```python
class DANAttention:
    """
    DAN (目标驱动注意力网络)

    特点:
    - 根据任务类型强制引导注意力方向
    - 任务偏置 B_DAN 由 L3 下发
    - 目标驱动的主动聚焦

    Args:
        embed_dim: 嵌入维度
        num_heads: 注意力头数

    Example:
        >>> dan = DANAttention(embed_dim=1024, num_heads=12)
        >>> output, weights = dan(query, key, value, task_bias)
    """

    def forward(self, query, key, value, task_bias):
        """
        前向传播

        Args:
            query: [batch, seq_len, embed_dim]
            key: [batch, seq_len, embed_dim]
            value: [batch, seq_len, embed_dim]
            task_bias: [batch, task_bias_dim]

        Returns:
            output: [batch, seq_len, embed_dim]
            attention_weights: [batch, num_heads, seq_len, seq_len]
        """
        pass
```

---

## 提交规范

### 提交信息格式

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Type 类型

| Type | 说明 | 示例 |
|------|------|------|
| `feat` | 新功能 | `feat(l1): add DAN attention` |
| `fix` | Bug 修复 | `fix(l2): correct entropy calculation` |
| `docs` | 文档更新 | `docs: update README` |
| `style` | 代码格式 | `style: format with black` |
| `refactor` | 重构 | `refactor(audit): simplify hash chain` |
| `perf` | 性能优化 | `perf(l1): improve attention speed` |
| `test` | 测试 | `test: add cutoff decision tests` |
| `chore` | 杂项 | `chore: update requirements` |

### 示例

```
feat(l3): add adaptive cutoff decision maker

Add ensemble cutoff decision maker that combines multiple
strategies for more robust cutoff detection.

- Add ensemble voting mechanism
- Add confidence scoring
- Add historical feedback integration

Closes: #123
Related to: #456
```

---

## 测试要求

### 测试覆盖率

| 模块 | 最低覆盖率 |
|------|-----------|
| attention/ | 80% |
| memory/ | 85% |
| cutoff/ | 80% |
| audit/ | 75% |

### 运行测试

```bash
# 运行所有测试
python -m pytest tests/

# 运行指定模块
python -m pytest tests/unit/test_entropy_tracker.py

# 带覆盖率
python -m pytest tests/ --cov=enlighten --cov-report=html

# 性能测试
python -m pytest tests/benchmark/
```

### 单元测试模板

```python
import pytest
import torch
from enlighten.memory.entropy_tracker import EntropyTracker


class TestEntropyTracker:
    """EntropyTracker 单元测试"""

    def setup_method(self):
        """每个测试方法前运行"""
        self.tracker = EntropyTracker(window_size=10)

    def test_initialization(self):
        """测试初始化"""
        assert self.tracker.window_size == 10
        assert len(self.tracker.history) == 0

    def test_update_and_statistics(self):
        """测试更新和统计"""
        attn = torch.softmax(torch.randn(2, 8, 8), dim=-1)
        self.tracker.update(attn)

        stats = self.tracker.get_statistics()
        assert "mean" in stats
        assert "variance" in stats
        assert 0 <= stats["mean"] <= 1  # 熵值范围

    @pytest.mark.parametrize("threshold", [0.3, 0.5, 0.7])
    def test_should_cutoff_threshold(self, threshold):
        """参数化测试"""
        self.tracker.entropy_threshold = threshold
        # ... 测试逻辑
```

---

## 文档要求

### 代码内文档

- 公共类/函数必须有 docstring
- 复杂逻辑添加行内注释
- 类型注解 (type hints) 优先于文档说明

### 更新文档

如果你的更改涉及：

| 更改类型 | 需要更新的文档 |
|---------|--------------|
| 新功能 | README, docs/, CHANGELOG |
| API 变更 | docs/api_reference.md, CHANGELOG |
| 配置变更 | docs/integration_guide.md |
| 数学逻辑 | docs/math_verification.md |
| 性能优化 | README 性能表格 |

### 文档更新检查清单

```markdown
- [ ] README.md 版本号已更新
- [ ] CHANGELOG.md 已添加变更记录
- [ ] docs/ 相关文档已更新
- [ ] 示例代码能正常运行
- [ ] 文档中的链接都有效
```

---

## 问题反馈

### Bug 报告

```markdown
## Bug 描述
清晰描述问题

## 环境信息
- OS:
- Python 版本:
- EnlightenLM 版本:

## 复现步骤
1.
2.
3.

## 预期 vs 实际
预期: ...
实际: ...

## 堆栈跟踪
```
```

### 功能请求

```markdown
## 功能描述
清晰描述你想要的功能

## 使用场景
描述你的使用场景

## 建议的解决方案
如果有的话

## 参考资料
相关的链接、文档
```

---

## 许可证

通过贡献代码，你同意将你的贡献按照 [MIT License](LICENSE) 发布。

---

*最后更新: 2026-04-23*
