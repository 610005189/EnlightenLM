"""
Multimodal VAN - 多模态视觉注意力网络
支持文本、图像、音频等多种输入模式的 VAN 检测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple, Union
from enum import Enum


class ModalityType(Enum):
    """模态类型"""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    MULTIMODAL = "multimodal"


@dataclass
class MultimodalConfig:
    """多模态配置"""
    modalities: List[ModalityType] = None
    embed_dim: int = 1024
    num_heads: int = 16
    text_vocab_size: int = 50000
    image_patch_size: int = 16
    audio_sample_rate: int = 16000
    fusion_strategy: str = "cross_attention"


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
    图像 VAN 编码器

    使用 Patch 级和空间级注意力检测有害图像
    """

    def __init__(self, config: MultimodalConfig):
        super().__init__()
        self.config = config

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

        return {
            "harm_prob": harm_prob,
            "patch_attention": patch_attn_weights,
            "spatial_attention": spatial_attn_weights,
            "embeddings": spatial_attn_output
        }


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
    """

    def __init__(self, config: MultimodalConfig):
        super().__init__()
        self.config = config

        self.modalities = config.modalities or [ModalityType.TEXT]

        if ModalityType.TEXT in self.modalities:
            self.text_encoder = TextVanEncoder(config)

        if ModalityType.IMAGE in self.modalities:
            self.image_encoder = ImageVanEncoder(config)

        if ModalityType.AUDIO in self.modalities:
            self.audio_encoder = AudioVanEncoder(config)

        self.fusion_strategy = config.fusion_strategy

        if self.fusion_strategy == "cross_attention":
            self.fusion_layer = nn.MultiheadAttention(
                embed_dim=config.embed_dim,
                num_heads=config.num_heads,
                batch_first=True
            )

        # 计算融合分类器的输入维度
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
            # 文本输出: [batch, seq_len, embed_dim]
            text_embedding = text_result["embeddings"]
            # 池化为 [batch, embed_dim]
            text_pooled = text_embedding.mean(dim=1)
            embeddings.append(text_pooled)
            harm_probs.append(text_result["harm_prob"])
            modality_results["text"] = text_result

        if "image" in inputs and hasattr(self, "image_encoder"):
            image_result = self.image_encoder(inputs["image"])
            # 图像输出: [seq_len, batch, embed_dim]
            image_embedding = image_result["embeddings"]
            # 池化为 [batch, embed_dim]
            image_pooled = image_embedding.mean(dim=0)
            embeddings.append(image_pooled)
            harm_probs.append(image_result["harm_prob"])
            modality_results["image"] = image_result

        if "audio" in inputs and hasattr(self, "audio_encoder"):
            audio_result = self.audio_encoder(inputs["audio"])
            # 音频输出: [seq_len, batch, embed_dim]
            audio_embedding = audio_result["embeddings"]
            # 池化为 [batch, embed_dim]
            audio_pooled = audio_embedding.mean(dim=0)
            embeddings.append(audio_pooled)
            harm_probs.append(audio_result["harm_prob"])
            modality_results["audio"] = audio_result

        if len(embeddings) == 1:
            final_embedding = embeddings[0]
            fused_prob = harm_probs[0]
        else:
            # 所有嵌入都是 [batch, embed_dim]，可以直接拼接
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

        is_harmful = result["harm_prob"] > threshold

        return is_harmful, result["harm_prob"], result
