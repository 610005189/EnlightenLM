from enlighten.memory.hallucination_discriminator import HallucinationDiscriminator, HallucinationDiscriminatorConfig, HallucinationFeatures
import numpy as np

# 测试 MLP 幻觉判别器模型
def test_hallucination_discriminator():
    print("测试 MLP 幻觉判别器模型...")
    
    # 初始化配置
    config = HallucinationDiscriminatorConfig()
    
    # 初始化模型
    model = HallucinationDiscriminator(config)
    print("模型初始化成功")
    
    # 测试单个样本前向传播
    print("\n测试单个样本前向传播:")
    features = HallucinationFeatures(
        entropy=0.5,
        confidence=0.5,
        repetition_rate=0.2,
        diversity=0.8,
        char_entropy=0.7,
        variance=0.1,
        trend=0.0,
        intervention_count=0
    )
    risk_prob = model.forward(features)
    print(f"幻觉风险概率: {risk_prob:.4f}")
    assert 0 <= risk_prob <= 1, "风险概率应在 [0, 1] 范围内"
    
    # 测试批量前向传播
    print("\n测试批量前向传播:")
    batch_features = np.random.rand(10, 8)  # 10个样本，每个8维特征
    batch_output = model.forward_batch(batch_features)
    print(f"批量输出形状: {batch_output.shape}")
    print(f"批量输出示例: {batch_output.detach().numpy()[:5].flatten()}")
    assert batch_output.shape == (10, 1), "批量输出形状应为 (10, 1)"
    assert all(0 <= prob <= 1 for prob in batch_output.detach().numpy().flatten()), "所有风险概率应在 [0, 1] 范围内"
    
    # 测试预测功能
    print("\n测试预测功能:")
    prediction = model.predict(features)
    print(f"预测结果: {prediction}")
    assert "risk_probability" in prediction, "预测结果应包含风险概率"
    assert "is_hallucination" in prediction, "预测结果应包含是否为幻觉的判断"
    
    # 测试特征提取功能
    print("\n测试特征提取功能:")
    text = "中国的首都是北京。"
    entropy_stats = {
        "current": 0.5,
        "variance": 0.1,
        "trend": 0.0,
        "intervention_count": 0
    }
    extracted_features = model.extract_features(text, entropy_stats)
    print(f"提取的特征: {extracted_features}")
    assert isinstance(extracted_features, HallucinationFeatures), "提取的特征应是 HallucinationFeatures 类型"
    
    print("\nMLP 幻觉判别器模型测试通过！")

if __name__ == "__main__":
    test_hallucination_discriminator()