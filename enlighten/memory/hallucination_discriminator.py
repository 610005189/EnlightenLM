"""
幻觉判别器模块

实现轻量级 MLP 幻觉判别器，用于预测幻觉风险
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Any
from dataclasses import dataclass


@dataclass
class HallucinationFeatures:
    """幻觉检测特征"""
    entropy: float
    confidence: float
    repetition_rate: float
    diversity: float
    char_entropy: float
    variance: float
    trend: float
    intervention_count: int


class HallucinationDiscriminatorConfig:
    """幻觉判别器配置"""
    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 64,
        output_dim: int = 1,
        dropout_rate: float = 0.2,
        threshold: float = 0.7,
        model_path: Optional[str] = None
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.threshold = threshold
        self.model_path = model_path


class HallucinationDiscriminator(nn.Module):
    """
    轻量级 MLP 幻觉判别器
    
    输入: 8维特征向量
    - entropy: 熵值
    - confidence: 置信度
    - repetition_rate: 重复率
    - diversity: 词汇多样性
    - char_entropy: 字符熵
    - variance: 熵方差
    - trend: 熵趋势
    - intervention_count: 干预次数
    
    输出: 幻觉风险概率 [0, 1]
    """
    
    def __init__(self, config: HallucinationDiscriminatorConfig):
        super().__init__()
        self.config = config
        
        # 定义MLP网络结构
        self.model = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim // 2, config.output_dim),
            nn.Sigmoid()
        )
        
        # 加载预训练模型（如果有）
        if config.model_path:
            try:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                self.load_state_dict(torch.load(config.model_path, map_location=device))
                print(f"Loaded hallucination discriminator model from {config.model_path} on {device}")
            except Exception as e:
                print(f"Failed to load model: {e}")
        
    def forward(self, features: HallucinationFeatures) -> float:
        """
        前向传播
        
        Args:
            features: 幻觉检测特征
            
        Returns:
            幻觉风险概率
        """
        # 构建特征向量
        feature_vector = torch.tensor([
            features.entropy,
            features.confidence,
            features.repetition_rate,
            features.diversity,
            features.char_entropy,
            features.variance,
            features.trend,
            features.intervention_count
        ], dtype=torch.float32)
        
        # 添加批次维度
        feature_vector = feature_vector.unsqueeze(0)
        
        # 前向传播
        with torch.no_grad():
            output = self.model(feature_vector)
        
        # 返回概率值
        return output.item()
    
    def predict(self, features: HallucinationFeatures) -> Dict[str, Any]:
        """
        预测幻觉风险
        
        Args:
            features: 幻觉检测特征
            
        Returns:
            包含风险概率和判断结果的字典
        """
        risk_prob = self.forward(features)
        is_hallucination = risk_prob >= self.config.threshold
        
        return {
            "risk_probability": risk_prob,
            "is_hallucination": is_hallucination,
            "threshold": self.config.threshold
        }
    
    def extract_features(self, text: str, entropy_stats: Dict[str, float]) -> HallucinationFeatures:
        """
        从文本和熵统计中提取特征
        
        Args:
            text: 输入文本
            entropy_stats: 熵统计信息
            
        Returns:
            幻觉检测特征
        """
        # 计算词汇重复率
        tokens = text.split()
        if len(tokens) < 2:
            repetition_rate = 0.0
        else:
            unique_tokens = len(set(tokens))
            repetition_rate = 1.0 - (unique_tokens / len(tokens))
        
        # 计算词汇多样性
        diversity = len(set(tokens)) / max(len(tokens), 1)
        
        # 计算字符熵
        char_set = set(text)
        char_entropy = len(char_set) / max(len(text), 1)
        
        # 构建特征对象
        features = HallucinationFeatures(
            entropy=entropy_stats.get("current", 1.0),
            confidence=1.0 - entropy_stats.get("current", 1.0),  # 置信度与熵成反比
            repetition_rate=repetition_rate,
            diversity=diversity,
            char_entropy=char_entropy,
            variance=entropy_stats.get("variance", 0.1),
            trend=entropy_stats.get("trend", 0.0),
            intervention_count=entropy_stats.get("intervention_count", 0)
        )
        
        return features
    
    def save_model(self, path: str):
        """
        保存模型
        
        Args:
            path: 保存路径
        """
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")


class SimpleHallucinationDetector:
    """
    简单的幻觉检测器（基于规则）
    
    当MLP模型不可用时使用
    """
    
    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
    
    def predict(self, features: HallucinationFeatures) -> Dict[str, Any]:
        """
        基于规则预测幻觉风险
        
        Args:
            features: 幻觉检测特征
            
        Returns:
            包含风险概率和判断结果的字典
        """
        # 计算综合风险分数
        risk_score = 0.0
        
        # 低熵（可能表示过度自信）
        if features.entropy < 0.3:
            risk_score += 0.3
        
        # 高重复率
        if features.repetition_rate > 0.5:
            risk_score += 0.2
        
        # 低多样性
        if features.diversity < 0.3:
            risk_score += 0.2
        
        # 高干预次数
        if features.intervention_count > 2:
            risk_score += 0.2
        
        # 熵趋势为负（质量下降）
        if features.trend < -0.05:
            risk_score += 0.1
        
        # 限制在[0, 1]范围内
        risk_score = min(max(risk_score, 0.0), 1.0)
        
        is_hallucination = risk_score >= self.threshold
        
        return {
            "risk_probability": risk_score,
            "is_hallucination": is_hallucination,
            "threshold": self.threshold
        }
    
    def extract_features(self, text: str, entropy_stats: Dict[str, float]) -> HallucinationFeatures:
        """
        从文本和熵统计中提取特征
        
        Args:
            text: 输入文本
            entropy_stats: 熵统计信息
            
        Returns:
            幻觉检测特征
        """
        # 计算词汇重复率
        tokens = text.split()
        if len(tokens) < 2:
            repetition_rate = 0.0
        else:
            unique_tokens = len(set(tokens))
            repetition_rate = 1.0 - (unique_tokens / len(tokens))
        
        # 计算词汇多样性
        diversity = len(set(tokens)) / max(len(tokens), 1)
        
        # 计算字符熵
        char_set = set(text)
        char_entropy = len(char_set) / max(len(text), 1)
        
        # 构建特征对象
        features = HallucinationFeatures(
            entropy=entropy_stats.get("current", 1.0),
            confidence=1.0 - entropy_stats.get("current", 1.0),  # 置信度与熵成反比
            repetition_rate=repetition_rate,
            diversity=diversity,
            char_entropy=char_entropy,
            variance=entropy_stats.get("variance", 0.1),
            trend=entropy_stats.get("trend", 0.0),
            intervention_count=entropy_stats.get("intervention_count", 0)
        )
        
        return features