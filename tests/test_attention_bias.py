import torch
from src.l1.attention_bias import AttentionBias

# 测试注意力偏置
print("Testing AttentionBias...")

# 初始化注意力偏置
attention_bias = AttentionBias(rank=8)

# 测试创建偏置
seq_len = 100
U, V = attention_bias.create_bias(seq_len, bias_type="layer-wise")
print(f"Created bias with shape: U={U.shape}, V={V.shape}")

# 测试计算偏置
bias = attention_bias.compute_bias(U, V, causal=True)
print(f"Computed bias shape: {bias.shape}")
print(f"Bias min: {bias.min().item()}, max: {bias.max().item()}")

# 测试组合偏置
bias1 = attention_bias.compute_bias(U, V, causal=True)
bias2 = attention_bias.compute_bias(U, V, causal=False)
combined_bias = attention_bias.combine_biases([bias1, bias2])
print(f"Combined bias shape: {combined_bias.shape}")

# 测试应用偏置
attention_scores = torch.randn(1, 8, seq_len, seq_len)
applied_bias = attention_bias.apply_bias(attention_scores, bias)
print(f"Applied bias shape: {applied_bias.shape}")

# 测试模型特定偏置
model_types = ["qwen", "llama", "deepseek", "unknown"]
for model_type in model_types:
    model_bias = attention_bias.get_model_specific_bias(model_type, seq_len, {"scale": 0.5})
    print(f"{model_type} bias shape: {model_bias.shape}")

print("AttentionBias tests passed!")
