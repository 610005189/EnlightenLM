import numpy as np
import torch
from typing import Dict, List, Optional, Tuple


class AttentionBias:
    """
    注意力偏置实现，支持低秩分解表示和不同模型架构
    """
    
    def __init__(self, rank: int = 8):
        """
        初始化注意力偏置
        
        Args:
            rank: 低秩分解的秩
        """
        self.rank = rank
        self.bias_cache = {}
    
    def create_bias(self, seq_len: int, bias_type: str = "layer-wise") -> Tuple[torch.Tensor, torch.Tensor]:
        """
        创建偏置矩阵的低秩分解表示
        
        Args:
            seq_len: 序列长度
            bias_type: 偏置类型，layer-wise 或 head-wise
            
        Returns:
            U 和 V 因子矩阵
        """
        cache_key = (seq_len, bias_type)
        if cache_key in self.bias_cache:
            return self.bias_cache[cache_key]
        
        # 创建低秩因子矩阵
        U = torch.randn(seq_len, self.rank)
        V = torch.randn(seq_len, self.rank)
        
        # 缓存结果
        self.bias_cache[cache_key] = (U, V)
        
        return U, V
    
    def compute_bias(self, U: torch.Tensor, V: torch.Tensor, causal: bool = True) -> torch.Tensor:
        """
        计算偏置矩阵
        
        Args:
            U: 左因子矩阵
            V: 右因子矩阵
            causal: 是否应用因果掩码
            
        Returns:
            偏置矩阵
        """
        # 计算偏置矩阵
        bias = torch.matmul(U, V.transpose(0, 1))
        
        # 应用因果掩码
        if causal:
            seq_len = bias.shape[0]
            mask = torch.tril(torch.ones(seq_len, seq_len))
            bias = bias * mask + (1 - mask) * -1e9
        
        return bias
    
    def combine_biases(self, biases: List[torch.Tensor]) -> torch.Tensor:
        """
        组合多个偏置
        
        Args:
            biases: 偏置列表
            
        Returns:
            组合后的偏置
        """
        if not biases:
            return None
        
        combined = biases[0]
        for bias in biases[1:]:
            combined += bias
        
        return combined
    
    def apply_bias(self, attention_scores: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """
        应用偏置到注意力分数
        
        Args:
            attention_scores: 注意力分数
            bias: 偏置矩阵
            
        Returns:
            应用偏置后的注意力分数
        """
        # 确保偏置形状匹配
        if bias.shape != attention_scores.shape[-2:]:
            # 扩展偏置维度以匹配注意力分数
            bias = bias.unsqueeze(0).unsqueeze(0)  # 添加batch和head维度
        
        return attention_scores + bias
    
    def get_model_specific_bias(self, model_type: str, seq_len: int, bias_params: Dict) -> torch.Tensor:
        """
        获取特定模型的注意力偏置
        
        Args:
            model_type: 模型类型，如 "qwen", "llama", "deepseek"
            seq_len: 序列长度
            bias_params: 偏置参数
            
        Returns:
            注意力偏置矩阵
        """
        # 根据模型类型创建不同的偏置
        if model_type.lower() == "qwen":
            return self._create_qwen_bias(seq_len, bias_params)
        elif model_type.lower() == "llama":
            return self._create_llama_bias(seq_len, bias_params)
        elif model_type.lower() == "deepseek":
            return self._create_deepseek_bias(seq_len, bias_params)
        else:
            # 默认偏置
            U, V = self.create_bias(seq_len)
            return self.compute_bias(U, V)
    
    def _create_qwen_bias(self, seq_len: int, bias_params: Dict) -> torch.Tensor:
        """
        创建Qwen模型的注意力偏置
        
        Args:
            seq_len: 序列长度
            bias_params: 偏置参数
            
        Returns:
            注意力偏置矩阵
        """
        # Qwen模型的偏置实现
        U, V = self.create_bias(seq_len)
        bias = self.compute_bias(U, V)
        
        # 应用Qwen特定的调整
        if "scale" in bias_params:
            bias *= bias_params["scale"]
        
        return bias
    
    def _create_llama_bias(self, seq_len: int, bias_params: Dict) -> torch.Tensor:
        """
        创建LLaMA模型的注意力偏置
        
        Args:
            seq_len: 序列长度
            bias_params: 偏置参数
            
        Returns:
            注意力偏置矩阵
        """
        # LLaMA模型的偏置实现
        U, V = self.create_bias(seq_len)
        bias = self.compute_bias(U, V)
        
        # 应用LLaMA特定的调整
        if "rope_scale" in bias_params:
            # 适配RoPE缩放
            pass
        
        return bias
    
    def _create_deepseek_bias(self, seq_len: int, bias_params: Dict) -> torch.Tensor:
        """
        创建DeepSeek模型的注意力偏置
        
        Args:
            seq_len: 序列长度
            bias_params: 偏置参数
            
        Returns:
            注意力偏置矩阵
        """
        # DeepSeek模型的偏置实现
        U, V = self.create_bias(seq_len)
        bias = self.compute_bias(U, V)
        
        # 应用DeepSeek特定的调整
        if "attention_scale" in bias_params:
            bias *= bias_params["attention_scale"]
        
        return bias
