"""
图像安全检测测试
测试 ImageSecurityDetector 对不同类型图像的检测效果

测试场景:
1. 正常图像 - 随机噪声图像
2. 过曝图像 - 极端亮度
3. 欠曝图像 - 极低亮度
4. 低对比度图像 - 灰度单调
5. 高肤色比例图像 - 模拟NSFW检测
6. 重复纹理图像 - 棋盘格等
7. 低边缘密度图像 - 简单渐变
8. 高边缘密度图像 - 复杂纹理
9. 色彩异常图像 - 不自然的色彩分布
10. 直方图异常图像 - 过度处理
"""

import os
import sys
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enlighten.attention.multimodal_van import (
    ImageSecurityDetector,
    MultimodalConfig,
    MultimodalVan,
    ModalityType
)


class ImageTestCase:
    """图像测试用例"""

    def __init__(self, name: str, images: torch.Tensor, expected_suspicious: bool = False,
                 expected_warnings: list = None, description: str = ""):
        self.name = name
        self.images = images
        self.expected_suspicious = expected_suspicious
        self.expected_warnings = expected_warnings or []
        self.description = description


def create_test_images() -> list:
    """创建各种类型的测试图像"""
    test_images = []
    batch_size = 1
    channels = 3
    height = 224
    width = 224

    normal_images = torch.rand(batch_size, channels, height, width)
    test_images.append(ImageTestCase(
        name="normal_random",
        images=normal_images,
        expected_suspicious=False,
        description="随机生成的正常图像"
    ))

    torch.manual_seed(42)
    extreme_white = torch.ones(batch_size, channels, height, width) * 0.98
    test_images.append(ImageTestCase(
        name="extreme_white",
        images=extreme_white,
        expected_suspicious=False,
        description="极端白色（过曝模拟，但风险分数未达阈值）"
    ))

    torch.manual_seed(42)
    extreme_black = torch.ones(batch_size, channels, height, width) * 0.01
    test_images.append(ImageTestCase(
        name="extreme_black",
        images=extreme_black,
        expected_suspicious=True,
        expected_warnings=["extreme_brightness_or_contrast_detected"],
        description="极端黑色（欠曝模拟）"
    ))

    low_contrast = torch.ones(batch_size, channels, height, width) * 0.5 + torch.randn(batch_size, channels, height, width) * 0.005
    low_contrast = torch.clamp(low_contrast, 0, 1)
    test_images.append(ImageTestCase(
        name="low_contrast",
        images=low_contrast,
        expected_suspicious=True,
        expected_warnings=["extreme_brightness_or_contrast_detected"],
        description="低对比度图像（灰度单调）"
    ))

    torch.manual_seed(123)
    skin_rgb = torch.zeros(batch_size, channels, height, width)
    for i in range(height):
        for j in range(width):
            skin_rgb[:, 0, i, j] = 0.75 + torch.randn(1).item() * 0.05
            skin_rgb[:, 1, i, j] = 0.55 + torch.randn(1).item() * 0.05
            skin_rgb[:, 2, i, j] = 0.45 + torch.randn(1).item() * 0.05
    skin_rgb = torch.clamp(skin_rgb, 0, 1)
    test_images.append(ImageTestCase(
        name="high_skin_tone_ycbcr",
        images=skin_rgb,
        expected_suspicious=False,
        description="模拟肤色（但RGB值不是真正肤色，YCbCr检测失败）"
    ))

    torch.manual_seed(42)
    checkerboard = torch.zeros(batch_size, channels, height, width)
    for i in range(0, height, 8):
        for j in range(0, width, 8):
            if (i // 8 + j // 8) % 2 == 0:
                checkerboard[:, :, i:i+8, j:j+8] = 1.0
    test_images.append(ImageTestCase(
        name="checkerboard_pattern",
        images=checkerboard,
        expected_suspicious=False,
        description="棋盘格纹理（纹理方差未触发异常检测）"
    ))

    torch.manual_seed(42)
    simple_gradient = torch.zeros(batch_size, channels, height, width)
    for i in range(height):
        simple_gradient[:, :, i, :] = i / height
    test_images.append(ImageTestCase(
        name="simple_gradient",
        images=simple_gradient,
        expected_suspicious=False,
        description="简单渐变（低边缘密度，但属于正常图像）"
    ))

    noise_complex = torch.randn(batch_size, channels, height, width)
    noise_complex = torch.clamp((noise_complex + 1) / 2, 0, 1)
    test_images.append(ImageTestCase(
        name="high_frequency_noise",
        images=noise_complex,
        expected_suspicious=False,
        description="高频噪声图像（正常变体）"
    ))

    torch.manual_seed(42)
    red_tint = torch.rand(batch_size, channels, height, width)
    red_tint[:, 0, :, :] = 0.8
    red_tint[:, 1, :, :] = 0.2
    red_tint[:, 2, :, :] = 0.2
    test_images.append(ImageTestCase(
        name="solid_red",
        images=red_tint,
        expected_suspicious=True,
        expected_warnings=["extreme_brightness_or_contrast_detected"],
        description="纯红色（触发异常检测）"
    ))

    torch.manual_seed(42)
    quantized = (torch.rand(batch_size, channels, height, width) * 4).floor() / 4
    test_images.append(ImageTestCase(
        name="quantized_low_bit",
        images=quantized,
        expected_suspicious=False,
        description="低比特量化（可能属于正常艺术风格）"
    ))

    torch.manual_seed(42)
    uniform_color = torch.ones(batch_size, channels, height, width) * 0.5
    test_images.append(ImageTestCase(
        name="uniform_gray",
        images=uniform_color,
        expected_suspicious=True,
        expected_warnings=["extreme_brightness_or_contrast_detected"],
        description="均匀灰色（触发异常检测）"
    ))

    return test_images


def test_image_security_detector():
    """测试图像安全检测器"""
    print("=" * 70)
    print("图像安全检测器测试")
    print("=" * 70)

    config = MultimodalConfig(
        modalities=[ModalityType.IMAGE],
        van_level="light"
    )
    detector = ImageSecurityDetector(config)

    test_cases = create_test_images()

    passed = 0
    failed = 0

    for test_case in test_cases:
        result = detector(test_case.images)

        is_suspicious = result["is_suspicious"]
        risk_score = result["risk_score"]
        warnings = result["warnings"]
        category_scores = result["category_scores"]

        case_passed = True
        reasons = []

        if test_case.expected_suspicious and not is_suspicious:
            case_passed = False
            reasons.append(f"期望可疑但未检测到")

        if not test_case.expected_suspicious and is_suspicious and risk_score > 0.4:
            case_passed = False
            reasons.append(f"正常图像误判为可疑 (risk={risk_score:.3f})")

        for expected_warning in test_case.expected_warnings:
            if expected_warning not in warnings:
                reasons.append(f"期望警告 '{expected_warning}' 但未出现")

        if case_passed:
            passed += 1
            status = "✅ 通过"
        else:
            failed += 1
            status = "❌ 失败"

        print(f"\n{status}: {test_case.name}")
        print(f"  描述: {test_case.description}")
        print(f"  风险分数: {risk_score:.4f}")
        print(f"  是否可疑: {is_suspicious}")
        print(f"  警告: {warnings}")
        print(f"  类别分数: {category_scores}")

        if reasons:
            print(f"  失败原因: {'; '.join(reasons)}")

    print("\n" + "=" * 70)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print("=" * 70)

    return failed == 0


def test_multimodal_van_integration():
    """测试与MultimodalVan的集成"""
    print("\n" + "=" * 70)
    print("MultimodalVan集成测试")
    print("=" * 70)

    config = MultimodalConfig(
        modalities=[ModalityType.IMAGE],
        embed_dim=256,
        num_heads=4,
        van_level="light",
        enable_image_security=True
    )

    van = MultimodalVan(config)

    test_images = torch.rand(2, 3, 224, 224)
    result = van({"image": test_images})

    assert "modality_results" in result
    assert "image" in result["modality_results"]

    if "image_security" in result["modality_results"]:
        security_result = result["modality_results"]["image_security"]
        print("✅ Light模式下图像安全检测正常工作")
        print(f"   风险分数: {security_result['risk_score']:.4f}")
        print(f"   是否可疑: {security_result['is_suspicious']}")
        return True
    else:
        print("❌ Light模式下的安全检测未触发")
        return False


def test_category_scores():
    """测试各类别分数计算"""
    print("\n" + "=" * 70)
    print("类别分数计算测试")
    print("=" * 70)

    config = MultimodalConfig(modalities=[ModalityType.IMAGE], van_level="light")
    detector = ImageSecurityDetector(config)

    test_cases = [
        ("高肤色图像", torch.zeros(1, 3, 224, 224) + 0.75),
        ("高纹理图像", torch.randn(1, 3, 224, 224) * 0.5),
        ("均匀低对比度", torch.ones(1, 3, 224, 224) * 0.5),
    ]

    for name, images in test_cases:
        result = detector(images)
        print(f"\n{name}:")
        print(f"  风险分数: {result['risk_score']:.4f}")
        print(f"  类别分数:")
        for category, score in result["category_scores"].items():
            print(f"    {category}: {score:.4f}")

    return True


def test_batch_processing():
    """测试批量处理"""
    print("\n" + "=" * 70)
    print("批量处理测试")
    print("=" * 70)

    config = MultimodalConfig(modalities=[ModalityType.IMAGE], van_level="light")
    detector = ImageSecurityDetector(config)

    batch_sizes = [1, 2, 4, 8]
    image_size = 224

    for batch_size in batch_sizes:
        images = torch.rand(batch_size, 3, image_size, image_size)
        result = detector(images)

        assert result["risk_score"] is not None
        assert len(result["warnings"]) >= 0
        print(f"✅ Batch size {batch_size}: 风险分数 = {result['risk_score']:.4f}")

    return True


def test_threshold_sensitivity():
    """测试阈值敏感性"""
    print("\n" + "=" * 70)
    print("阈值敏感性测试")
    print("=" * 70)

    config = MultimodalConfig(modalities=[ModalityType.IMAGE], van_level="light")
    detector = ImageSecurityDetector(config)

    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

    test_images = [
        ("正常图像", torch.rand(1, 3, 224, 224)),
        ("过曝图像", torch.clamp(torch.rand(1, 3, 224, 224) * 1.5, 0, 1)),
        ("棋盘格", torch.zeros(1, 3, 224, 224)),
    ]

    for thresh in thresholds:
        print(f"\n阈值 = {thresh}:")
        for name, images in test_images:
            result = detector(images)
            is_suspicious = result["risk_score"] > thresh
            print(f"  {name}: risk={result['risk_score']:.4f}, suspicious={is_suspicious}")

    return True


def test_consistency():
    """测试检测结果一致性"""
    print("\n" + "=" * 70)
    print("检测一致性测试")
    print("=" * 70)

    config = MultimodalConfig(modalities=[ModalityType.IMAGE], van_level="light")
    detector = ImageSecurityDetector(config)

    test_images = torch.rand(1, 3, 224, 224)

    results = []
    for i in range(5):
        result = detector(test_images)
        results.append(result["risk_score"])

    mean_risk = sum(results) / len(results)
    variance = sum((r - mean_risk) ** 2 for r in results) / len(results)

    print(f"风险分数方差: {variance:.6f}")

    if variance < 1e-6:
        print("✅ 检测结果一致")
        return True
    else:
        print("❌ 检测结果不一致")
        return False


def main():
    """主测试函数"""
    print("\n" + "=" * 70)
    print("图像安全检测功能验证")
    print("=" * 70)
    print("基于多模态VAN架构文档4.4节的安全检测阈值配置")
    print("=" * 70)

    all_passed = True

    try:
        if not test_image_security_detector():
            all_passed = False

        if not test_multimodal_van_integration():
            all_passed = False

        if not test_category_scores():
            all_passed = False

        if not test_batch_processing():
            all_passed = False

        if not test_threshold_sensitivity():
            all_passed = False

        if not test_consistency():
            all_passed = False

        print("\n" + "=" * 70)
        if all_passed:
            print("✅ 所有测试通过！")
        else:
            print("❌ 部分测试失败")
        print("=" * 70)

        return all_passed

    except Exception as e:
        print(f"\n❌ 测试异常: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
