"""
多模态 VAN 集成测试
测试多模态视觉注意力网络的功能
"""

import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enlighten.attention.multimodal_van import MultimodalVan, MultimodalConfig, ModalityType


def test_multimodal_van_init():
    """测试多模态 VAN 初始化"""
    print("=== 测试多模态 VAN 初始化 ===")

    # 测试默认配置（仅文本）
    config = MultimodalConfig()
    van = MultimodalVan(config)
    assert van is not None
    assert ModalityType.TEXT in van.modalities
    assert hasattr(van, "text_encoder")
    print("✅ 默认配置初始化成功")

    # 测试多模态配置
    multimodal_config = MultimodalConfig(
        modalities=[ModalityType.TEXT, ModalityType.IMAGE, ModalityType.AUDIO]
    )
    multimodal_van = MultimodalVan(multimodal_config)
    assert ModalityType.TEXT in multimodal_van.modalities
    assert ModalityType.IMAGE in multimodal_van.modalities
    assert ModalityType.AUDIO in multimodal_van.modalities
    assert hasattr(multimodal_van, "text_encoder")
    assert hasattr(multimodal_van, "image_encoder")
    assert hasattr(multimodal_van, "audio_encoder")
    print("✅ 多模态配置初始化成功")


def test_text_van_encoder():
    """测试文本 VAN 编码器"""
    print("\n=== 测试文本 VAN 编码器 ===")

    config = MultimodalConfig(
        modalities=[ModalityType.TEXT],
        text_vocab_size=10000,
        embed_dim=256,
        num_heads=4
    )
    van = MultimodalVan(config)

    # 生成测试数据
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, 10000, (batch_size, seq_len))

    # 测试前向传播
    result = van.text_encoder(input_ids)
    assert "harm_prob" in result
    assert "word_attention" in result
    assert "sentence_attention" in result
    assert "embeddings" in result
    assert result["harm_prob"].shape == (batch_size,)
    assert result["embeddings"].shape == (batch_size, seq_len, 256)
    print("✅ 文本 VAN 编码器测试成功")


def test_image_van_encoder():
    """测试图像 VAN 编码器"""
    print("\n=== 测试图像 VAN 编码器 ===")

    config = MultimodalConfig(
        modalities=[ModalityType.IMAGE],
        embed_dim=256,
        num_heads=4,
        image_patch_size=8
    )
    van = MultimodalVan(config)

    # 生成测试数据
    batch_size = 2
    channels = 3
    height = 32
    width = 32
    images = torch.randn(batch_size, channels, height, width)

    # 测试前向传播
    result = van.image_encoder(images)
    assert "harm_prob" in result
    assert "patch_attention" in result
    assert "spatial_attention" in result
    assert "embeddings" in result
    assert result["harm_prob"].shape == (batch_size,)
    print("✅ 图像 VAN 编码器测试成功")


def test_audio_van_encoder():
    """测试音频 VAN 编码器"""
    print("\n=== 测试音频 VAN 编码器 ===")

    config = MultimodalConfig(
        modalities=[ModalityType.AUDIO],
        embed_dim=256,
        num_heads=4
    )
    van = MultimodalVan(config)

    # 生成测试数据
    batch_size = 2
    freq = 80
    time = 100
    audio = torch.randn(batch_size, 1, freq, time)

    # 测试前向传播
    result = van.audio_encoder(audio)
    assert "harm_prob" in result
    assert "frame_attention" in result
    assert "spectrum_attention" in result
    assert "embeddings" in result
    assert result["harm_prob"].shape == (batch_size,)
    print("✅ 音频 VAN 编码器测试成功")


def test_multimodal_fusion():
    """测试多模态融合"""
    print("\n=== 测试多模态融合 ===")

    config = MultimodalConfig(
        modalities=[ModalityType.TEXT, ModalityType.IMAGE],
        embed_dim=256,
        num_heads=4,
        text_vocab_size=10000,
        image_patch_size=8,
        fusion_strategy="cross_attention"
    )
    van = MultimodalVan(config)

    # 生成测试数据
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, 10000, (batch_size, seq_len))

    channels = 3
    height = 32
    width = 32
    images = torch.randn(batch_size, channels, height, width)

    # 测试多模态输入
    inputs = {
        "text": input_ids,
        "image": images
    }

    result = van(inputs)
    assert "harm_prob" in result
    assert "modality_harm_probs" in result
    assert "modality_results" in result
    assert "is_multimodal" in result
    assert "active_modalities" in result
    assert result["is_multimodal"] is True
    assert len(result["active_modalities"]) == 2
    assert "text" in result["active_modalities"]
    assert "image" in result["active_modalities"]
    print("✅ 多模态融合测试成功")


def test_harm_detection():
    """测试有害内容检测"""
    print("\n=== 测试有害内容检测 ===")

    config = MultimodalConfig(
        modalities=[ModalityType.TEXT],
        text_vocab_size=10000,
        embed_dim=256,
        num_heads=4
    )
    van = MultimodalVan(config)

    # 生成测试数据
    batch_size = 1
    seq_len = 10
    input_ids = torch.randint(0, 10000, (batch_size, seq_len))

    inputs = {"text": input_ids}

    # 测试检测
    is_harmful, harm_prob, details = van.detect_harm(inputs, threshold=0.5)
    assert isinstance(is_harmful, bool) or isinstance(is_harmful, torch.Tensor)
    assert isinstance(harm_prob, float)
    assert 0 <= harm_prob <= 1
    assert isinstance(details, dict)
    print(f"✅ 有害内容检测测试成功: 有害概率 = {harm_prob:.4f}")


def test_multimodal_configs():
    """测试不同的多模态配置"""
    print("\n=== 测试多模态配置组合 ===")

    # 测试仅图像
    image_config = MultimodalConfig(modalities=[ModalityType.IMAGE])
    image_van = MultimodalVan(image_config)
    assert hasattr(image_van, "image_encoder")
    assert not hasattr(image_van, "text_encoder")
    print("✅ 仅图像配置测试成功")

    # 测试仅音频
    audio_config = MultimodalConfig(modalities=[ModalityType.AUDIO])
    audio_van = MultimodalVan(audio_config)
    assert hasattr(audio_van, "audio_encoder")
    assert not hasattr(audio_van, "text_encoder")
    print("✅ 仅音频配置测试成功")

    # 测试三模态
    three_modal_config = MultimodalConfig(
        modalities=[ModalityType.TEXT, ModalityType.IMAGE, ModalityType.AUDIO]
    )
    three_modal_van = MultimodalVan(three_modal_config)
    assert hasattr(three_modal_van, "text_encoder")
    assert hasattr(three_modal_van, "image_encoder")
    assert hasattr(three_modal_van, "audio_encoder")
    print("✅ 三模态配置测试成功")


def test_embedding_dimensions():
    """测试不同嵌入维度的配置"""
    print("\n=== 测试嵌入维度配置 ===")

    # 测试不同嵌入维度
    for embed_dim in [128, 256, 512]:
        config = MultimodalConfig(
            modalities=[ModalityType.TEXT],
            embed_dim=embed_dim,
            text_vocab_size=10000,
            num_heads=4
        )
        van = MultimodalVan(config)
        
        batch_size = 1
        seq_len = 5
        input_ids = torch.randint(0, 10000, (batch_size, seq_len))
        
        result = van.text_encoder(input_ids)
        assert result["embeddings"].shape == (batch_size, seq_len, embed_dim)
        print(f"✅ 嵌入维度 {embed_dim} 测试成功")


def main():
    """主测试函数"""
    print("开始多模态 VAN 集成测试...\n")

    try:
        test_multimodal_van_init()
        test_text_van_encoder()
        test_image_van_encoder()
        test_audio_van_encoder()
        test_multimodal_fusion()
        test_harm_detection()
        test_multimodal_configs()
        test_embedding_dimensions()

        print("\n=== 测试完成 ===")
        print("✅ 所有多模态 VAN 测试通过！")
        return True
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
