# 多模态VAN架构设计文档

**版本**: v1.0
**日期**: 2026-04-25
**状态**: ✅ 已实现完成

---

## 1. 概述

### 1.1 目标
设计一个支持多模态输入（文本、图像、音频）的VAN（变异性吸引子网络）架构，实现跨模态的有害内容检测。

### 1.2 设计原则
- **模块化**: 各模态编码器独立设计，便于扩展
- **可配置性**: 支持三级漏斗机制（light/medium/full）
- **可扩展性**: 便于添加新的模态类型
- **高效性**: 支持VAN Level light模式下的快速检测

---

## 2. 架构总览

```
┌─────────────────────────────────────────────────────────────────┐
│                      MultimodalVan                               │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │TextVanEncoder│  │ImageVanEncoder│  │AudioVanEncoder│           │
│  │  - 词级注意力 │  │  - Patch注意力 │  │  - 帧级注意力 │           │
│  │  - 句级注意力 │  │  - 空间注意力  │  │  - 语谱注意力 │           │
│  │  - MLP分类器 │  │  - 安全检测头 │  │  - MLP分类器  │           │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘           │
│         │                 │                 │                   │
│         └─────────────────┼─────────────────┘                   │
│                           ▼                                      │
│                  ┌─────────────────┐                            │
│                  │  Fusion Layer   │                            │
│                  │ (cross_attention)│                            │
│                  └────────┬────────┘                            │
│                           ▼                                      │
│                  ┌─────────────────┐                            │
│                  │  Harm Probability│                           │
│                  │     Output       │                            │
│                  └─────────────────┘                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. 图像输入处理流程

### 3.1 流程图

```
图像输入 [B, 3, H, W]
        │
        ▼
┌───────────────────┐
│  图像预处理        │
│  - 调整尺寸        │
│  - 归一化          │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  Patch卷积        │
│  Conv2d(3, 1024,  │
│   kernel=16)      │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  Patch级注意力     │◄──────────┐
│  MultiheadAttn    │           │
└─────────┬─────────┘           │
          │                      │
          ▼                      │
┌───────────────────┐             │
│  空间级注意力      │────────────┘
│  MultiheadAttn    │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  特征池化         │
│  Mean Pooling     │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  安全检测头        │ (可选)
│  - NSFW           │
│  - Violence       │
│  - Fraud          │
│  - Hate           │
│  - Terrorism      │
│  - Spam           │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  Harm Probability │
└───────────────────┘
```

### 3.2 详细处理步骤

#### 3.2.1 图像预处理
```python
# 归一化到 [0, 1]
images = images / 255.0  # 如果是uint8

# ImageNet 归一化
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
images = (images - mean) / std
```

#### 3.2.2 Patch分割
```python
patch_conv = nn.Conv2d(3, embed_dim, kernel_size=16, stride=16)
# 输入: [B, 3, 224, 224] -> 输出: [B, 1024, 14, 14]
patches = patch_conv(images)
# 转换为序列: [B, 1024, 14*14] = [B, 1024, 196]
patches = patches.view(B, embed_dim, -1).transpose(1, 2)
```

#### 3.2.3 双层注意力
```python
# Patch级注意力 - 局部特征交互
patch_attn_out, patch_weights = self.patch_attention(patches, patches, patches)

# 空间级注意力 - 全局场景理解
spatial_input = patch_attn_out.transpose(0, 1)
spatial_attn_out, spatial_weights = self.spatial_attention(
    spatial_input, spatial_input, spatial_input
)
```

---

## 4. 图像安全检测算法

### 4.1 三级漏斗机制

#### Level Light: 规则匹配 + 快速安全检测
- **像素级异常检测**: 极端亮度/对比度
- **纹理性异常**: 重复模式/棋盘格
- **肤色比例检测**: 粗略NSFW筛选
- **边缘密度分析**: 低复杂度图像检测

#### Level Medium: 轻量MLP分类器
- 基于Patch特征的二分类
- 计算效率优先
- 适用于实时场景

#### Level Full: 完整注意力 + 多任务分类
- 6类安全检测头: NSFW, Violence, Fraud, Hate, Terrorism, Spam
- 注意力加权融合
- 最高检测精度

### 4.2 ImageSecurityDetector 算法

```python
class ImageSecurityDetector:
    """快速规则检测器 (Light级别)"""

    def detect(images: Tensor) -> Dict[str, Any]:
        # 1. 像素级异常
        pixel_anomaly = check_brightness_contrast(images)

        # 2. 纹理异常
        texture_anomaly = check_texture_pattern(images)

        # 3. 肤色比例
        skin_ratio = estimate_skin_color_ratio(images)

        # 4. 边缘密度
        edge_density = compute_edge_density(images)

        # 5. 综合风险评分
        risk_score = fuse_scores(
            pixel_anomaly * 0.3 +
            texture_anomaly * 0.25 +
            skin_score * 0.25 +
            edge_score * 0.2
        )

        return {
            "risk_score": risk_score,
            "is_suspicious": risk_score > 0.5,
            "warnings": generate_warnings(...)
        }
```

### 4.3 ImageVanEncoder 安全检测头

```python
class ImageVanEncoder:
    """完整安全检测 (Full级别)"""

    def __init__(self, config):
        self.security_heads = nn.ModuleDict({
            "nsfw": create_security_head(),
            "violence": create_security_head(),
            "fraud": create_security_head(),
            "hate": create_security_head(),
            "terrorism": create_security_head(),
            "spam": create_security_head(),
        })

    def compute_security_scores(self, pooled_features):
        scores = {}
        for category, head in self.security_heads.items():
            scores[category] = head(pooled_features)

        scores["fused_score"] = self.security_fusion(
            torch.cat([scores[c] for c in scores], dim=-1)
        )
        return scores
```

### 4.4 安全检测阈值配置

| 类别 | Light阈值 | Medium阈值 | Full阈值 | 权重 |
|------|-----------|------------|----------|------|
| NSFW | 0.3 | 0.5 | 0.7 | 0.25 |
| Violence | 0.3 | 0.5 | 0.7 | 0.25 |
| Fraud | 0.2 | 0.4 | 0.6 | 0.15 |
| Hate | 0.2 | 0.4 | 0.6 | 0.15 |
| Terrorism | 0.1 | 0.3 | 0.5 | 0.15 |
| Spam | 0.4 | 0.6 | 0.8 | 0.05 |

---

## 5. 数据流与接口

### 5.1 输入格式

```python
# 文本输入
text_input: Dict[str, Tensor] = {
    "input_ids": torch.tensor([[101, 2003, 1045, 4649, 102]]),  # [B, seq_len]
}

# 图像输入
image_input: Dict[str, Tensor] = {
    "images": torch.randn(B, 3, 224, 224),  # [B, C, H, W]
}

# 音频输入
audio_input: Dict[str, Tensor] = {
    "audio": torch.randn(B, 16000),  # [B, samples]
}

# 多模态输入
multimodal_input: Dict[str, Tensor] = {
    "text": text_input["input_ids"],
    "image": image_input["images"],
    "audio": audio_input["audio"],
}
```

### 5.2 输出格式

```python
output: Dict[str, Any] = {
    "harm_prob": 0.85,  # 综合有害概率
    "modality_harm_probs": {
        "text": 0.2,
        "image": 0.9,
        "audio": 0.1,
    },
    "modality_results": {
        "text": {
            "harm_prob": 0.2,
            "word_attention": Tensor,
            "sentence_attention": Tensor,
        },
        "image": {
            "harm_prob": 0.9,
            "patch_attention": Tensor,
            "spatial_attention": Tensor,
            "security_scores": {
                "nsfw": 0.7,
                "violence": 0.2,
                "fraud": 0.1,
                "hate": 0.05,
                "terrorism": 0.01,
                "spam": 0.3,
                "fused_score": 0.85,
            },
        },
        "image_security": {  # Light模式特有
            "risk_score": 0.75,
            "is_suspicious": True,
            "warnings": ["high_skin_ratio_detected"],
        }
    },
    "is_multimodal": True,
    "active_modalities": ["text", "image", "audio"],
}
```

---

## 6. 配置参数

```python
@dataclass
class MultimodalConfig:
    modalities: List[ModalityType] = None
    embed_dim: int = 1024              # 嵌入维度
    num_heads: int = 16                  # 注意力头数
    text_vocab_size: int = 50000        # 词表大小
    image_patch_size: int = 16          # 图像块大小
    image_scales: List[int] = [8, 16, 32]  # 多尺度配置
    audio_sample_rate: int = 16000      # 音频采样率
    fusion_strategy: str = "cross_attention"
    van_level: str = "medium"           # light/medium/full
    enable_image_security: bool = True  # 启用图像安全检测
    security_threshold: float = 0.7     # 安全检测阈值
```

---

## 7. 实现文件结构

```
enlighten/attention/
├── van.py                    # 原始VAN实现
├── multimodal_van.py         # 多模态VAN实现 (已更新)
├── fusion.py                 # 模态融合策略
├── sparse.py                 # 稀疏注意力
├── dan.py                    # 背侧注意力网络
└── __init__.py
```

---

## 8. 使用示例

### 8.1 初始化
```python
config = MultimodalConfig(
    modalities=[ModalityType.TEXT, ModalityType.IMAGE],
    van_level="full",
    enable_image_security=True,
    security_threshold=0.7,
)
van = MultimodalVan(config)
```

### 8.2 单模态图像检测
```python
images = torch.randn(1, 3, 224, 224)
result = van({"image": images})
print(f"Harm probability: {result['harm_prob']}")
print(f"Security scores: {result['modality_results']['image']['security_scores']}")
```

### 8.3 多模态检测
```python
inputs = {
    "text": text_tokens,
    "image": images,
}
result = van(inputs)
print(f"Multimodal harm: {result['harm_prob']}")
```

---

## 9. 下一步计划

### 9.1 已验证项
- [x] ImageVanEncoder 的多尺度特征提取
- [x] ImageSecurityDetector 的规则有效性
- [x] 安全检测头的训练数据需求
- [x] Light/Medium/Full模式的性能对比

### 9.2 待完成项
- [ ] 添加视频模态支持
- [ ] 添加注意力可视化

---

## 10. 参考资料

- 原VAN实现: [van.py](file:///c:/Users/wiggin/Documents/trae_projects/EnlightenLM/enlighten/attention/van.py)
- 多模态VAN: [multimodal_van.py](file:///c:/Users/wiggin/Documents/trae_projects/EnlightenLM/enlighten/attention/multimodal_van.py)
- L3控制器: [hybrid_architecture.py](file:///c:/Users/wiggin/Documents/trae_projects/EnlightenLM/enlighten/hybrid_architecture.py#L944-L1219)
