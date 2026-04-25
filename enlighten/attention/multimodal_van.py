"""
Multimodal VAN - 多模态视觉注意力网络
支持文本、图像、音频等多种输入模式的 VAN 检测

架构设计:
1. 多模态输入处理: TextVanEncoder, ImageVanEncoder, AudioVanEncoder
2. 三级漏斗机制: light(规则匹配) → medium(轻量MLP) → full(完整注意力)
3. 图像安全检测: 多尺度分析 + 注意力机制 + 多任务分类
4. 模态融合: cross_attention 融合多模态特征
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from enum import Enum


class ModalityType(Enum):
    """模态类型"""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    MULTIMODAL = "multimodal"


class HarmCategory(Enum):
    """有害内容类别（图像专用）"""
    NSFW = "nsfw"
    VIOLENCE = "violence"
    FRAUD = "fraud"
    HATE = "hate"
    TERRORISM = "terrorism"
    SPAM = "spam"


@dataclass
class MultimodalConfig:
    """多模态配置"""
    modalities: List[ModalityType] = None
    embed_dim: int = 1024
    num_heads: int = 16
    text_vocab_size: int = 50000
    image_patch_size: int = 16
    image_scales: List[int] = field(default_factory=lambda: [8, 16, 32])
    audio_sample_rate: int = 16000
    fusion_strategy: str = "cross_attention"
    van_level: str = "medium"
    enable_image_security: bool = True
    security_threshold: float = 0.7


class TextVanEncoder(nn.Module):
    """
    文本 VAN 编码器

    使用词级和句级注意力检测有害文本
    """

    def __init__(self, config: MultimodalConfig):
        super().__init__()
        self.config = config

        self.word_embedding = nn.Embedding(config.text_vocab_size, config.embed_dim)

        self.word_attention = nn.MultiheadAttention(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            batch_first=True
        )

        self.sentence_attention = nn.MultiheadAttention(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            batch_first=True
        )

        self.text_classifier = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.embed_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids: torch.Tensor) -> Dict[str, Any]:
        """
        前向传播

        Args:
            input_ids: [batch, seq_len]

        Returns:
            包含分类结果和注意力的字典
        """
        word_embeds = self.word_embedding(input_ids)

        word_attn_output, word_attn_weights = self.word_attention(
            word_embeds, word_embeds, word_embeds
        )

        sentence_attn_output, sentence_attn_weights = self.sentence_attention(
            word_attn_output, word_attn_output, word_attn_output
        )

        pooled_output = sentence_attn_output.mean(dim=1)

        harm_prob = self.text_classifier(pooled_output).squeeze(-1)

        return {
            "harm_prob": harm_prob,
            "word_attention": word_attn_weights,
            "sentence_attention": sentence_attn_weights,
            "embeddings": sentence_attn_output
        }


class ImageVanEncoder(nn.Module):
    """
    图像 VAN 编码器 - 增强版

    图像处理流程:
    1. 多尺度特征提取 (MSFA): 提取不同尺度的图像特征
    2. Patch级注意力: 检测局部异常区域
    3. 空间级注意力: 整体场景理解
    4. 安全检测头: 多任务分类 (NSFW/暴力/欺诈等)

    Args:
        config: MultimodalConfig 配置对象
    """

    def __init__(self, config: MultimodalConfig):
        super().__init__()
        self.config = config
        self.enable_security = config.enable_image_security

        self.patch_conv = nn.Conv2d(
            in_channels=3,
            out_channels=config.embed_dim,
            kernel_size=config.image_patch_size,
            stride=config.image_patch_size
        )

        self.patch_attention = nn.MultiheadAttention(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            batch_first=True
        )

        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            batch_first=True
        )

        self.image_classifier = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.embed_dim // 2, 1),
            nn.Sigmoid()
        )

        if self.enable_security:
            self._init_security_heads(config)

    def _init_security_heads(self, config: MultimodalConfig):
        """初始化安全检测头"""
        self.security_heads = nn.ModuleDict({
            "nsfw": self._create_security_head(config, 1),
            "violence": self._create_security_head(config, 1),
            "fraud": self._create_security_head(config, 1),
            "hate": self._create_security_head(config, 1),
            "terrorism": self._create_security_head(config, 1),
            "spam": self._create_security_head(config, 1),
        })

        self.security_fusion = nn.Sequential(
            nn.Linear(len(self.security_heads) * 1, config.embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.embed_dim // 2, 1),
            nn.Sigmoid()
        )

    def _create_security_head(self, config: MultimodalConfig, output_dim: int) -> nn.Module:
        """创建单个安全检测头"""
        return nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.embed_dim // 2, output_dim),
            nn.Sigmoid()
        )

    def forward(self, images: torch.Tensor) -> Dict[str, Any]:
        """
        前向传播

        Args:
            images: [batch, channels, height, width]

        Returns:
            包含分类结果和注意力的字典
        """
        batch_size = images.shape[0]

        patches = self.patch_conv(images)
        patches = patches.view(batch_size, self.config.embed_dim, -1).transpose(1, 2)

        patch_attn_output, patch_attn_weights = self.patch_attention(
            patches, patches, patches
        )

        spatial_input = patch_attn_output.transpose(0, 1)
        spatial_attn_output, spatial_attn_weights = self.spatial_attention(
            spatial_input, spatial_input, spatial_input
        )

        pooled_output = spatial_attn_output.mean(dim=0)

        harm_prob = self.image_classifier(pooled_output).squeeze(-1)

        result = {
            "harm_prob": harm_prob,
            "patch_attention": patch_attn_weights,
            "spatial_attention": spatial_attn_weights,
            "embeddings": spatial_attn_output,
            "pooled_features": pooled_output
        }

        if self.enable_security:
            security_scores = self._compute_security_scores(pooled_output)
            result["security_scores"] = security_scores
            result["security_harm_prob"] = security_scores["fused_score"]
            batch_size = harm_prob.shape[0] if harm_prob.dim() > 0 else 1
            if harm_prob.dim() == 0:
                harm_prob_val = harm_prob.item()
            else:
                harm_prob_val = harm_prob.mean().item()
            new_harm_prob = max(harm_prob_val, security_scores["fused_score"])
            result["harm_prob"] = torch.full((batch_size,), new_harm_prob)

        return result

    def _compute_security_scores(self, pooled_features: torch.Tensor) -> Dict[str, float]:
        """
        计算各类型安全检测分数

        Args:
            pooled_features: [batch, embed_dim] 池化后的特征

        Returns:
            各类型安全分数和融合分数
        """
        scores = {}
        individual_scores = []

        batch_size = pooled_features.shape[0]

        for category, head in self.security_heads.items():
            score_tensor = head(pooled_features)
            if batch_size == 1:
                score = score_tensor.item()
            else:
                score = score_tensor.mean().item()
            scores[category] = score
            individual_scores.append(score)

        if batch_size == 1:
            scores["fused_score"] = self.security_fusion(
                torch.tensor(individual_scores, device=pooled_features.device).unsqueeze(0)
            ).item()
        else:
            scores["fused_score"] = self.security_fusion(
                torch.tensor(individual_scores, device=pooled_features.device).unsqueeze(0)
            ).mean().item()

        return scores


class ImageSecurityDetector(nn.Module):
    """
    独立图像安全检测器

    用于VAN Level light模式下的快速规则检测

    检测策略:
    1. 像素级异常: 极端亮度/对比度/色彩分布
    2. 纹理性异常: 重复模式/棋盘格检测
    3. 语义级异常: 肤色比例检测 (NSFW筛选)
    4. 边缘密度异常: 低复杂度或高复杂度图像
    5. 色彩空间异常: 不自然的色彩分布

    安全阈值配置 (参考架构文档 4.4):
    | 类别     | Light阈值 | 权重 |
    |----------|-----------|------|
    | NSFW     | 0.3       | 0.25 |
    | Violence | 0.3       | 0.25 |
    | Fraud    | 0.2       | 0.15 |
    | Hate     | 0.2       | 0.15 |
    | Terrorism| 0.1       | 0.15 |
    | Spam     | 0.4       | 0.05 |
    """

    PIXEL_ANOMALY_THRESHOLD = 0.02
    SKIN_COLOR_RATIO_HIGH = 0.45
    SKIN_COLOR_RATIO_LOW = 0.05
    EDGE_DENSITY_LOW = 0.01
    EDGE_DENSITY_HIGH = 0.5
    COLOR_VARIANCE_LOW = 0.005
    COLOR_VARIANCE_HIGH = 0.2

    def __init__(self, config: MultimodalConfig):
        super().__init__()
        self.config = config

        self.edge_detector = nn.Conv2d(3, 1, kernel_size=3, padding=1, bias=False)
        self.texture_conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.color_conv = nn.Conv2d(3, 3, kernel_size=1)

        self._init_ycbcr_weights()

    def _init_ycbcr_weights(self):
        """初始化YCbCr转换权重（带偏移量以确保CbCr在正确范围）"""
        ycbcr_weight = torch.tensor([
            [0.299, 0.587, 0.114],
            [-0.168736, -0.331264, 0.5],
            [0.5, -0.418688, -0.081312]
        ], dtype=torch.float32)
        self.register_buffer('ycbcr_weight', ycbcr_weight)
        self.register_buffer('ycbcr_offset', torch.tensor([0.0, 0.5, 0.5], dtype=torch.float32))

    def _rgb_to_ycbcr(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """RGB转YCbCr色彩空间（归一化到[0,1]范围）"""
        batch_size = images.shape[0]
        images_flat = images.view(batch_size, 3, -1)
        ycbcr_flat = torch.matmul(self.ycbcr_weight, images_flat)
        ycbcr_flat = ycbcr_flat + self.ycbcr_offset.view(3, 1)
        ycbcr = ycbcr_flat.view(batch_size, 3, images.shape[2], images.shape[3])

        y, cb, cr = ycbcr[:, 0], ycbcr[:, 1], ycbcr[:, 2]
        return y, cb, cr

    def forward(self, images: torch.Tensor) -> Dict[str, Any]:
        """
        快速安全检测

        Args:
            images: [batch, 3, H, W] 归一化图像 (值范围 [0, 1])

        Returns:
            安全检测结果
        """
        batch_size = images.shape[0]
        device = images.device

        pixel_anomaly_score = self._check_pixel_anomaly(images)
        texture_score = self._check_texture_anomaly(images)
        skin_ratio = self._estimate_skin_ratio_ycbcr(images)
        edge_density = self._compute_edge_density(images)
        color_score = self._check_color_anomaly(images)
        histogram_score = self._check_histogram_anomaly(images)

        risk_score = self._compute_risk_score(
            pixel_anomaly_score, texture_score, skin_ratio,
            edge_density, color_score, histogram_score
        )

        category_scores = self._compute_category_scores(
            skin_ratio, texture_score, color_score,
            pixel_anomaly_score, edge_density, histogram_score
        )

        return {
            "pixel_anomaly": pixel_anomaly_score,
            "texture_anomaly": texture_score,
            "skin_ratio": skin_ratio,
            "edge_density": edge_density,
            "color_anomaly": color_score,
            "histogram_anomaly": histogram_score,
            "risk_score": risk_score,
            "is_suspicious": risk_score > 0.5,
            "warnings": self._generate_warnings(
                pixel_anomaly_score, texture_score, skin_ratio,
                edge_density, color_score, histogram_score
            ),
            "category_scores": category_scores,
            "harm_prob": risk_score
        }

    def _rgb_to_ycbcr(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """RGB转YCbCr色彩空间"""
        batch_size = images.shape[0]
        images_flat = images.view(batch_size, 3, -1)
        ycbcr_flat = torch.matmul(self.ycbcr_weight, images_flat)
        ycbcr = ycbcr_flat.view(batch_size, 3, images.shape[2], images.shape[3])

        y, cb, cr = ycbcr[:, 0], ycbcr[:, 1], ycbcr[:, 2]
        return y, cb, cr

    def _check_pixel_anomaly(self, images: torch.Tensor) -> float:
        """检测像素级异常"""
        mean = images.mean(dim=[2, 3])
        std = images.std(dim=[2, 3])

        brightness_anomaly = ((mean < 0.02) | (mean > 0.98)).float().mean().item()
        contrast_anomaly = (std < 0.01).float().mean().item()

        return max(brightness_anomaly, contrast_anomaly)

    def _check_texture_anomaly(self, images: torch.Tensor) -> float:
        """检测纹理性异常（重复模式/棋盘格）"""
        texture_features = self.texture_conv(images)

        texture_variance = texture_features.var(dim=[2, 3]).mean().item()

        if texture_variance < 0.001:
            return 0.8
        elif texture_variance < 0.01:
            return 0.4

        return 0.0

    def _estimate_skin_ratio_ycbcr(self, images: torch.Tensor) -> float:
        """
        使用YCbCr色彩空间估计肤色比例（更精确的NSFW检测）

        YCbCr空间中肤色有明显的Cb和Cr范围:
        - Cb: 77-127 (归一化后 0.302-0.498)
        - Cr: 133-173 (归一化后 0.522-0.678)
        """
        y, cb, cr = self._rgb_to_ycbcr(images)

        cb_min, cb_max = 77 / 255.0, 127 / 255.0
        cr_min, cr_max = 133 / 255.0, 173 / 255.0

        cb_skin = (cb >= cb_min) & (cb <= cb_max)
        cr_skin = (cr >= cr_min) & (cr <= cr_max)

        skin_mask = (cb_skin & cr_skin).float()

        skin_ratio = skin_mask.mean(dim=[1, 2]).mean().item()

        return skin_ratio

    def _estimate_skin_ratio(self, images: torch.Tensor) -> float:
        """估计肤色比例（RGB空间粗略检测 - 保留兼容性）"""
        r, g, b = images[:, 0, :, :], images[:, 1, :, :], images[:, 2, :, :]

        skin_mask = (
            (r > 0.4) & (g > 0.2) & (b > 0.1) &
            (r > g) & (r > b) &
            (r - g > 0.05)
        ).float()

        skin_ratio = skin_mask.mean(dim=[1, 2]).mean().item()

        return skin_ratio

    def _compute_edge_density(self, images: torch.Tensor) -> float:
        """计算边缘密度"""
        edges = self.edge_detector(images)
        edge_density = edges.abs().mean(dim=[2, 3]).mean().item()

        return edge_density

    def _check_color_anomaly(self, images: torch.Tensor) -> float:
        """检测色彩异常（不自然的色彩分布）"""
        r, g, b = images[:, 0], images[:, 1], images[:, 2]

        color_variance = torch.stack([r.var(), g.var(), b.var()]).mean().item()

        if color_variance < self.COLOR_VARIANCE_LOW:
            return 0.6
        elif color_variance > self.COLOR_VARIANCE_HIGH:
            return 0.3

        rg_corr = ((r - r.mean()) * (g - g.mean())).mean() / (r.std() * g.std() + 1e-8)
        rb_corr = ((r - r.mean()) * (b - b.mean())).mean() / (r.std() * b.std() + 1e-8)
        gb_corr = ((g - g.mean()) * (b - b.mean())).mean() / (g.std() * b.std() + 1e-8)

        unusual_correlation = 0.0
        if abs(rg_corr) > 0.9:
            unusual_correlation += 0.3
        if abs(rb_corr) > 0.9:
            unusual_correlation += 0.3
        if abs(gb_corr) > 0.9:
            unusual_correlation += 0.3

        return min(unusual_correlation, 1.0)

    def _check_histogram_anomaly(self, images: torch.Tensor) -> float:
        """检测直方图异常（过度处理/伪造迹象）"""
        batch_size = images.shape[0]

        histogram_scores = []
        for i in range(batch_size):
            img = images[i]
            hist_range = (0, 1)

            hist_r = torch.histc(img[0], bins=32, min=hist_range[0], max=hist_range[1])
            hist_g = torch.histc(img[1], bins=32, min=hist_range[0], max=hist_range[1])
            hist_b = torch.histc(img[2], bins=32, min=hist_range[0], max=hist_range[1])

            hist_r_norm = hist_r / (hist_r.sum() + 1e-8)
            hist_g_norm = hist_g / (hist_g.sum() + 1e-8)
            hist_b_norm = hist_b / (hist_b.sum() + 1e-8)

            entropy_r = -(hist_r_norm * torch.log(hist_r_norm + 1e-8)).sum()
            entropy_g = -(hist_g_norm * torch.log(hist_g_norm + 1e-8)).sum()
            entropy_b = -(hist_b_norm * torch.log(hist_b_norm + 1e-8)).sum()

            avg_entropy = (entropy_r + entropy_g + entropy_b) / 3.0

            max_entropy = torch.log(torch.tensor(32.0))
            normalized_entropy = avg_entropy / max_entropy

            if normalized_entropy < 0.3:
                histogram_scores.append(0.7)
            elif normalized_entropy > 0.95:
                histogram_scores.append(0.4)
            else:
                histogram_scores.append(0.0)

        return sum(histogram_scores) / len(histogram_scores)

    def _compute_risk_score(
        self,
        pixel_anomaly: float,
        texture_anomaly: float,
        skin_ratio: float,
        edge_density: float,
        color_anomaly: float,
        histogram_anomaly: float
    ) -> float:
        """
        计算综合风险分数

        根据架构文档4.4的权重配置:
        - NSFW相关 (肤色): 0.25
        - Violence相关 (纹理): 0.25
        - Fraud相关 (色彩): 0.15
        - Hate相关 (直方图): 0.15
        - Terrorism相关 (像素): 0.15
        - Spam相关 (边缘): 0.05
        """
        risk = 0.0

        if pixel_anomaly > 0.5:
            risk += 0.15
        if texture_anomaly > 0.5:
            risk += 0.25 * texture_anomaly
        if skin_ratio > self.SKIN_COLOR_RATIO_HIGH:
            risk += 0.25
        elif skin_ratio < self.SKIN_COLOR_RATIO_LOW:
            risk += 0.05
        if edge_density < self.EDGE_DENSITY_LOW:
            risk += 0.05
        elif edge_density > self.EDGE_DENSITY_HIGH:
            risk += 0.03
        if color_anomaly > 0.3:
            risk += 0.15 * color_anomaly
        if histogram_anomaly > 0.3:
            risk += 0.15 * histogram_anomaly

        return min(risk, 1.0)

    def _compute_category_scores(
        self,
        skin_ratio: float,
        texture_anomaly: float,
        color_anomaly: float,
        pixel_anomaly: float,
        edge_density: float,
        histogram_anomaly: float
    ) -> Dict[str, float]:
        """计算各类别的安全分数"""
        return {
            "nsfw": min(skin_ratio / self.SKIN_COLOR_RATIO_HIGH, 1.0) if skin_ratio > self.SKIN_COLOR_RATIO_HIGH else 0.0,
            "violence": texture_anomaly,
            "fraud": color_anomaly,
            "hate": histogram_anomaly,
            "terrorism": pixel_anomaly,
            "spam": 0.0 if self.EDGE_DENSITY_LOW < edge_density < self.EDGE_DENSITY_HIGH else 0.5
        }

    def _generate_warnings(
        self,
        pixel_anomaly: float,
        texture_anomaly: float,
        skin_ratio: float,
        edge_density: float,
        color_anomaly: float,
        histogram_anomaly: float
    ) -> List[str]:
        """生成警告信息"""
        warnings = []

        if pixel_anomaly > 0.5:
            warnings.append("extreme_brightness_or_contrast_detected")
        if texture_anomaly > 0.5:
            warnings.append("abnormal_texture_pattern_detected")
        if skin_ratio > self.SKIN_COLOR_RATIO_HIGH:
            warnings.append("high_skin_ratio_detected")
        elif skin_ratio < self.SKIN_COLOR_RATIO_LOW:
            warnings.append("unusual_color_distribution_detected")
        if edge_density < self.EDGE_DENSITY_LOW:
            warnings.append("low_edge_density_detected")
        elif edge_density > self.EDGE_DENSITY_HIGH:
            warnings.append("high_edge_density_detected")
        if color_anomaly > 0.3:
            warnings.append("unusual_color_correlation_detected")
        if histogram_anomaly > 0.3:
            warnings.append("histogram_anomaly_detected")

        return warnings


class AudioVanEncoder(nn.Module):
    """
    音频 VAN 编码器

    使用帧级和语谱注意力检测有害音频
    """

    def __init__(self, config: MultimodalConfig):
        super().__init__()
        self.config = config

        self.mel_conv = nn.Conv2d(
            in_channels=1,
            out_channels=config.embed_dim // 4,
            kernel_size=(80, 3),
            stride=(80, 1)
        )

        self.frame_attention = nn.MultiheadAttention(
            embed_dim=config.embed_dim // 4,
            num_heads=config.num_heads // 4,
            batch_first=True
        )

        self.spectrum_attention = nn.MultiheadAttention(
            embed_dim=config.embed_dim // 4,
            num_heads=config.num_heads // 4,
            batch_first=True
        )

        self.audio_classifier = nn.Sequential(
            nn.Linear(config.embed_dim // 4, config.embed_dim // 8),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.embed_dim // 8, 1),
            nn.Sigmoid()
        )

    def forward(self, audio: torch.Tensor) -> Dict[str, Any]:
        """
        前向传播

        Args:
            audio: [batch, samples] 或 [batch, 1, freq, time]

        Returns:
            包含分类结果和注意力的字典
        """
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)

        mel_spec = self.mel_conv(audio)

        batch_size = mel_spec.shape[0]
        # 调整维度：[batch, channels, freq, time] -> [batch, time, channels]
        mel_spec = mel_spec.squeeze(2).transpose(1, 2)

        # 确保输入是 3D 张量 [batch, seq_len, embed_dim]
        frame_attn_output, frame_attn_weights = self.frame_attention(
            mel_spec, mel_spec, mel_spec
        )

        # 空间注意力输入：[seq_len, batch, embed_dim]
        spectrum_input = frame_attn_output.transpose(0, 1)
        spectrum_attn_output, spectrum_attn_weights = self.spectrum_attention(
            spectrum_input, spectrum_input, spectrum_input
        )

        pooled_output = spectrum_attn_output.mean(dim=0)

        harm_prob = self.audio_classifier(pooled_output).squeeze(-1)

        return {
            "harm_prob": harm_prob,
            "frame_attention": frame_attn_weights,
            "spectrum_attention": spectrum_attn_weights,
            "embeddings": spectrum_attn_output
        }


class MultimodalVan(nn.Module):
    """
    多模态 VAN

    统一处理文本、图像、音频等多种输入模式

    支持三级漏斗机制:
    - light: 规则匹配 + 快速安全检测
    - medium: 轻量MLP分类器
    - full: 完整注意力机制

    Args:
        config: MultimodalConfig 配置对象
    """

    def __init__(self, config: MultimodalConfig):
        super().__init__()
        self.config = config
        self.van_level = config.van_level

        self.modalities = config.modalities or [ModalityType.TEXT]

        if ModalityType.TEXT in self.modalities:
            self.text_encoder = TextVanEncoder(config)

        if ModalityType.IMAGE in self.modalities:
            self.image_encoder = ImageVanEncoder(config)

            if self.van_level == "light":
                self.image_security_detector = ImageSecurityDetector(config)

        if ModalityType.AUDIO in self.modalities:
            self.audio_encoder = AudioVanEncoder(config)

        self.fusion_strategy = config.fusion_strategy

        if self.fusion_strategy == "cross_attention":
            self.fusion_layer = nn.MultiheadAttention(
                embed_dim=config.embed_dim,
                num_heads=config.num_heads,
                batch_first=True
            )

        fusion_input_dim = 0
        if ModalityType.TEXT in self.modalities:
            fusion_input_dim += config.embed_dim
        if ModalityType.IMAGE in self.modalities:
            fusion_input_dim += config.embed_dim
        if ModalityType.AUDIO in self.modalities:
            fusion_input_dim += config.embed_dim // 4

        self.fusion_classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, config.embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.embed_dim, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        """
        前向传播

        Args:
            inputs: 包含不同模态输入的字典

        Returns:
            多模态检测结果
        """
        embeddings = []
        harm_probs = []
        modality_results = {}

        if "text" in inputs and hasattr(self, "text_encoder"):
            text_result = self.text_encoder(inputs["text"])
            text_embedding = text_result["embeddings"]
            text_pooled = text_embedding.mean(dim=1)
            embeddings.append(text_pooled)
            harm_probs.append(text_result["harm_prob"])
            modality_results["text"] = text_result

        if "image" in inputs and hasattr(self, "image_encoder"):
            image_result = self.image_encoder(inputs["image"])
            image_embedding = image_result["embeddings"]
            image_pooled = image_embedding.mean(dim=0)
            embeddings.append(image_pooled)
            harm_probs.append(image_result["harm_prob"])
            modality_results["image"] = image_result

            if self.van_level == "light" and hasattr(self, "image_security_detector"):
                security_result = self.image_security_detector(inputs["image"])
                modality_results["image_security"] = security_result

                if security_result["is_suspicious"]:
                    harm_probs[-1] = max(harm_probs[-1], security_result["risk_score"])

        if "audio" in inputs and hasattr(self, "audio_encoder"):
            audio_result = self.audio_encoder(inputs["audio"])
            audio_embedding = audio_result["embeddings"]
            audio_pooled = audio_embedding.mean(dim=0)
            embeddings.append(audio_pooled)
            harm_probs.append(audio_result["harm_prob"])
            modality_results["audio"] = audio_result

        if len(embeddings) == 1:
            final_embedding = embeddings[0]
            fused_prob = harm_probs[0]
        else:
            final_embedding = torch.cat(embeddings, dim=-1)
            fused_prob = self.fusion_classifier(final_embedding).squeeze(-1)

        return {
            "harm_prob": fused_prob,
            "modality_harm_probs": {
                k: v["harm_prob"] if isinstance(v, dict) else v
                for k, v in modality_results.items()
            },
            "modality_results": modality_results,
            "is_multimodal": len(embeddings) > 1,
            "active_modalities": list(modality_results.keys())
        }

    def detect_harm(
        self,
        inputs: Dict[str, torch.Tensor],
        threshold: float = 0.5
    ) -> Tuple[Union[bool, torch.Tensor], Union[float, torch.Tensor], Dict[str, Any]]:
        """
        有害内容检测

        Args:
            inputs: 输入数据
            threshold: 检测阈值

        Returns:
            (是否有害, 有害概率, 详细信息)
        """
        result = self.forward(inputs)

        harm_prob = result["harm_prob"]
        if hasattr(harm_prob, 'item'):
            harm_prob = harm_prob.item()

        is_harmful = harm_prob > threshold

        return is_harmful, harm_prob, result
