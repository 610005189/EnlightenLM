# L3 贝叶斯元控制 - 论文信号映射文档

> 本文档建立论文《元认知动态调节消解智能体偏见与幻觉的猜想》中理论符号到 EnlightenLM L2 工程实现的映射关系。

---

## 1. 论文内部信号定义

论文定义的内部信号四元组：

```
o_int = (C, A, H, Ḟ)
```

| 符号 | 名称 | 数学定义 | 物理含义 |
|------|------|----------|----------|
| C | 复杂度代价 | D_KL[q(x) \|\| p(x)] | 后验与先验的KL散度，衡量模型复杂度 |
| A | 精度代价 | E_q[ln p(o\|x)] | 对数似然的期望，衡量数据拟合度 |
| H | 后验熵 | H[q(x)] | 后验分布的不确定性 |
| Ḟ | 自由能变化率 | dF_β/dt | 自由能的时间导数 |

---

## 2. 工程实现映射表

### 2.1 主要信号映射

| 论文符号 | L2 工程量 | 计算方式 | 说明 |
|----------|-----------|----------|------|
| C | `entropy_stats['diversity']` | unique_tokens / total_tokens | 词汇多样性，近似复杂度 |
| A | `entropy_stats['entropy']` | 组合熵值 | 多维度文本熵，近似精度 |
| H | `μ_H` (均值), `σ_H²` (方差) | 滑动窗口熵统计 | 后验熵的时序统计 |
| Ḟ | `k_H` (趋势斜率) | polyfit 线性拟合 | 熵值变化趋势 |

### 2.2 扩展信号

| L2 工程量 | 计算方式 | 对应论文概念 |
|-----------|----------|--------------|
| `entropy_stats['variance']` | 滑动窗口熵方差 | H 的二阶矩 |
| `entropy_stats['trend']` | 线性拟合斜率 | Ḟ 离散近似 |
| `entropy_stats['current']` | 最新熵值 | H(t) |
| `entropy_stats['stability']` | 1 - min(variance*10, 1) | 系统稳定性指标 |
| `p_harm_raw` | VAN事件风险值 | 注意力异常检测 |

---

## 3. 详细映射实现

### 3.1 C (复杂度) → diversity

**论文定义**:
```
C ≡ D_KL[q(x)∥p(x)]
```

**工程近似**:
```python
# L2 WorkingMemoryManager.compute_entropy_stats()
tokens = text.split()
unique_tokens = len(set(tokens))
diversity = unique_tokens / max(len(tokens), 1)
```

**简化说明**: 论文的C是KL散度，需要计算两个分布。工程上用词汇多样性(diversity)作为近似：
- diversity 高 → 先验被数据更新 → C 低
- diversity 低 → 后验贴近先验 → C 高

### 3.2 A (精度) → entropy

**论文定义**:
```
A ≡ E_q[ln p(o|x)]
```

**工程近似**:
```python
# 组合多维度熵
combined_entropy = (diversity * 0.4 +
                   (1 - repetition_score) * 0.3 +
                   char_entropy * 0.3)
```

**简化说明**: 论文的A是对数似然的期望。工程上用组合熵近似：
- char_entropy 高 → 词汇丰富 → A 高
- repetition 低 → 文本多样 → A 高

### 3.3 H (后验熵) → μ_H, σ_H²

**论文定义**:
```
H ≡ H[q(x)]
```

**工程实现**:
```python
# WorkingMemoryManager 维护 entropy_history 滑动窗口
def compute_entropy_stats(self):
    entropy_data = np.array(list(self.entropy_history))

    mean = np.mean(entropy_data)        # μ_H
    variance = np.var(entropy_data)      # σ_H²
    # ...
```

**映射关系**:
- `μ_H` = mean(entropy_history) → H 的时序均值
- `σ_H²` = variance(entropy_history) → H 的时序方差

### 3.4 Ḟ (自由能变化率) → k_H

**论文定义**:
```
Ḟ ≡ dF_β/dt
```

**工程近似**:
```python
# 使用线性拟合斜率近似导数
if len(entropy_values) >= 5:
    x = np.arange(len(entropy_values))
    result = np.polyfit(x, entropy_values, 1)
    trend = float(result[0])  # k_H ≈ Ḟ
```

**简化说明**: 论文的Ḟ是自由能对时间的连续导数。工程上用离散差分近似：
- k_H > 0 → 熵上升趋势 → Ḟ > 0
- k_H < 0 → 熵下降趋势 → Ḟ < 0

---

## 4. VAN 监控信号映射

### 4.1 p_harm_raw (危险值)

论文中VAN监控输出风险值，对应工程实现：

```python
# VANMonitor.check_output()
# 返回 (blocked, reason, risk_score)
# risk_score 即 p_harm_raw
```

### 4.2 信号关联

| VAN 事件 | 工程检测 | 对应论文概念 |
|----------|----------|--------------|
| 敏感词检测 | 正则匹配 | 高 C 信号 |
| 自指循环 | 词汇模式 | 高 σ_H² 信号 |
| 词汇重复 | bigram 分析 | 高 repetition_score |
| 低熵截断 | 熵值阈值 | H < 阈值 |

---

## 5. 完整 L3 输入信号

基于以上映射，L3贝叶斯控制器的输入信号：

```
o_int_L3 = (μ_H, σ_H², k_H, p_harm_raw)
```

| 信号 | 来源 | 论文对应 | 说明 |
|------|------|----------|------|
| μ_H | entropy_history.mean() | H | 后验熵均值 |
| σ_H² | entropy_history.var() | ΔH | 后验熵方差 |
| k_H | polyfit slope | Ḟ | 熵变化趋势 |
| p_harm_raw | VANMonitor.risk | C+A | 注意力异常 |

---

## 6. 假设判别条件

### 6.1 论文判别条件

论文定义的三个假设：

- **H₁ (正常)**: μ_H 收敛，σ_H² 小，k_H ≈ 0
- **H₂ (噪声)**: σ_H² 大（尖峰），k_H 振荡
- **H₃ (偏见)**: μ_H 长期偏低，k_H 出现阶跃

### 6.2 工程判别规则

基于L2输出的判别：

```python
def classify_hypothesis(entropy_stats, van_risk):
    μ_H = entropy_stats['mean']
    σ_H² = entropy_stats['variance']
    k_H = entropy_stats['trend']
    p_harm = van_risk

    # H₁ 正常
    if abs(k_H) < 0.1 and σ_H² < 0.05:
        return 'H1'

    # H₃ 偏见 (高复杂度 + 高风险)
    if μ_H < 0.3 and p_harm > 0.5:
        return 'H3'

    # H₂ 噪声 (高方差 + 振荡)
    if σ_H² > 0.1:
        return 'H2'

    return 'H1'  # 默认正常
```

---

## 7. 简化近似总结

| 论文概念 | 工程近似 | 简化原因 |
|----------|----------|----------|
| KL散度 C | 词汇多样性 | 无法获取真实后验分布 |
| 对数似然 A | 组合文本熵 | API黑盒无法获取logits |
| 后验熵 H | 滑动熵统计 | 用时序统计代替分布熵 |
| 自由能导数 Ḟ | 线性拟合斜率 | 离散近似连续导数 |
| 贝叶斯后验 | 朴素分类器 | 假设空间只有3个 |

---

## 8. 后续验证

本映射的正确性需通过第二步的Trace可视化实验验证：
- 在三种条件下采集 o_int_L3
- 验证 (μ_H, σ_H², k_H) 在3D空间中确实可分
- 若不可分，需调整映射或近似方式

---

**文档版本**: v1.0
**创建日期**: 2026-04-24
**关联论文**: docs/paper/元认知动态调节消解智能体偏见与幻觉的猜想.md
**关联计划**: docs/L3优化计划.md
