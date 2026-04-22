import torch
import numpy as np
import logging
import hashlib
import os
from typing import Dict, List, Optional, Tuple
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)


class L3MetaAttentionController:
    """
    L3 元注意力控制器，负责生成任务偏置和管理核心价值观内嵌
    """
    
    def __init__(self, config: Dict):
        """
        初始化L3控制器
        
        Args:
            config: 控制器配置
        """
        self.config = config
        self.model_name = config.get("model_name", "google/bert-mini-uncased")
        self.device = config.get("device", "cuda")
        
        # 核心价值观配置
        self.core_values_config = config.get("core_values", {})
        self.negative_vocab_path = self.core_values_config.get("negative_vocab_path", "config/negative_vocab.txt")
        self.positive_vocab_path = self.core_values_config.get("positive_vocab_path", "config/positive_vocab.txt")
        self.negative_bias = self.core_values_config.get("negative_bias", -1e9)
        self.positive_bias = self.core_values_config.get("positive_bias", 2.0)
        
        # 用户参数配置
        self.user_params_config = config.get("user_params", {})
        self.creativity_range = self.user_params_config.get("creativity_range", [-1, 1])
        self.detail_attention_range = self.user_params_config.get("detail_attention_range", [0, 1])
        self.safety_margin_range = self.user_params_config.get("safety_margin_range", [0.3, 1])
        self.role_presets = self.user_params_config.get("role_presets", ["teacher", "assistant", "analyst"])
        
        # 直接使用模拟模型，避免网络依赖
        logger.info(f"Using mock L3 model: {self.model_name} on {self.device}")
        
        # 模拟分词器
        class MockTokenizer:
            def __call__(self, text, return_tensors=None, truncation=None, padding=None, max_length=None):
                return {"input_ids": torch.tensor([[1, 2, 3, 4, 5]]), "attention_mask": torch.tensor([[1, 1, 1, 1, 1]])}
            
            def encode(self, text):
                return [1, 2, 3, 4, 5]
            
            @property
            def pad_token(self):
                return "[PAD]"
            
            @pad_token.setter
            def pad_token(self, value):
                pass
        
        # 模拟模型
        class MockModel:
            def to(self, device):
                return self
            
            def eval(self):
                return self
            
            def __call__(self, input_ids, attention_mask=None):
                class MockOutput:
                    def __init__(self):
                        self.last_hidden_state = torch.randn(1, 5, 100)
                return MockOutput()
        
        self.tokenizer = MockTokenizer()
        self.model = MockModel()
        
        # 加载词汇表
        self.negative_vocab, self.negative_vocab_hash = self._load_vocab_with_hash(self.negative_vocab_path)
        self.positive_vocab, self.positive_vocab_hash = self._load_vocab_with_hash(self.positive_vocab_path)
        
        # 影响力向量（预定义）
        self.influence_vectors = {
            "creativity": np.array([0.2, -0.1, 0.0, 0.1, 0.0]),  # 对各价值观维度的影响
            "detail": np.array([0.0, 0.3, 0.1, 0.0, 0.2]),
            "safety": np.array([0.0, 0.0, 0.5, 0.3, 0.0]),
            "role": np.array([0.1, 0.1, 0.1, 0.1, 0.1])
        }
        
        # 核心价值观向量（专家预定义）
        self.core_values_vector = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        
        # 价值观维度定义
        self.value_dimensions = {
            "anti_discrimination": 0,
            "factualness": 1,
            "privacy_protection": 2,
            "reject_illegal": 3,
            "ethical_reasoning": 4
        }
    
    def _load_vocab_with_hash(self, vocab_path: str) -> Tuple[List[str], str]:
        """
        加载词汇表并计算哈希
        
        Args:
            vocab_path: 词汇表路径
            
        Returns:
            词汇列表和哈希值
        """
        try:
            # 检查文件是否存在
            if not os.path.exists(vocab_path):
                logger.warning(f"Vocab file not found: {vocab_path}, using empty vocab")
                return [], hashlib.sha256(b"").hexdigest()
            
            vocab = []
            content = []
            with open(vocab_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    content.append(line)
                    if line and not line.startswith('#'):
                        vocab.append(line)
            
            # 计算哈希
            content_str = '\n'.join(content)
            vocab_hash = hashlib.sha256(content_str.encode()).hexdigest()
            
            return vocab, vocab_hash
        except Exception as e:
            logger.error(f"Error loading vocab: {e}")
            return [], hashlib.sha256(b"").hexdigest()
    
    def reload_vocab(self):
        """
        重新加载词汇表
        """
        self.negative_vocab, self.negative_vocab_hash = self._load_vocab_with_hash(self.negative_vocab_path)
        self.positive_vocab, self.positive_vocab_hash = self._load_vocab_with_hash(self.positive_vocab_path)
        logger.info(f"Vocab reloaded. Negative hash: {self.negative_vocab_hash[:8]}, Positive hash: {self.positive_vocab_hash[:8]}")
    
    def generate_task_bias(self, input_text: str, seq_len: int) -> Dict:
        """
        生成任务偏置
        
        Args:
            input_text: 输入文本
            seq_len: 序列长度
            
        Returns:
            任务偏置参数
        """
        # 分词
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        # 获取模型输出
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs.last_hidden_state
        
        # 计算池化输出
        # 使用[CLS] token的表示作为池化输出
        pooled_output = last_hidden_state[:, 0, :].squeeze().numpy()
        
        # 生成层级别的偏置参数
        # 这里使用简单的线性变换，实际应用中可能需要更复杂的模型
        num_layers = 12  # 假设12层
        layers = []
        
        for i in range(num_layers):
            # 基于池化输出生成每层的参数
            # 简单起见，使用不同的线性组合
            alpha = (pooled_output[i % len(pooled_output)] - 0.5) * 2  # 映射到[-1, 1]
            beta = (pooled_output[(i + 1) % len(pooled_output)] - 0.5) * 2
            gamma = (pooled_output[(i + 2) % len(pooled_output)] - 0.5) * 2
            
            # 硬截断到[-5.0, 5.0]
            alpha = max(-5.0, min(5.0, alpha))
            beta = max(-5.0, min(5.0, beta))
            gamma = max(-5.0, min(5.0, gamma))
            
            layers.append({
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma
            })
        
        return {
            "layers": layers,
            "global_scale": 1.0
        }
    
    def generate_core_bias(self, input_tokens: List[str]) -> Dict:
        """
        生成核心价值观偏置
        
        Args:
            input_tokens: 输入token列表
            
        Returns:
            核心价值观偏置
        """
        seq_len = len(input_tokens)
        bias = np.zeros((seq_len, seq_len))
        
        # 窗口大小
        W = 5
        
        for i, token in enumerate(input_tokens):
            # 检查负向词汇
            for neg_word in self.negative_vocab:
                if neg_word in token:
                    # 强制所有查询位置无法注意该token
                    bias[:, i] = self.negative_bias
                    # 邻近衰减窗口
                    for j in range(max(0, i - W), min(seq_len, i + W + 1)):
                        decay = 1e9 * (1 - abs(i - j) / W)
                        bias[:, j] = max(bias[:, j], -decay)
            
            # 检查正向词汇
            for pos_word in self.positive_vocab:
                if pos_word in token:
                    bias[:, i] += self.positive_bias
        
        return {
            "bias": bias.tolist(),
            "negative_vocab_hash": self.negative_vocab_hash,
            "positive_vocab_hash": self.positive_vocab_hash
        }
    
    def safe_project_user_params(self, user_params: Dict) -> Dict:
        """
        安全投影用户参数
        
        Args:
            user_params: 用户参数
            
        Returns:
            安全投影后的用户参数
        """
        # 提取用户参数
        creativity = user_params.get("creativity", 0.0)
        detail = user_params.get("detail", 0.5)
        safety = user_params.get("safety", 0.5)
        role = user_params.get("role", 0)  # 0: teacher, 1: assistant, 2: analyst
        
        # 验证参数范围
        creativity = max(self.creativity_range[0], min(self.creativity_range[1], creativity))
        detail = max(self.detail_attention_range[0], min(self.detail_attention_range[1], detail))
        safety = max(self.safety_margin_range[0], min(self.safety_margin_range[1], safety))
        role = max(0, min(len(self.role_presets) - 1, role))
        
        # 计算用户影响向量
        user_influence = (
            creativity * self.influence_vectors["creativity"] +
            detail * self.influence_vectors["detail"] +
            safety * self.influence_vectors["safety"] +
            (role / len(self.role_presets)) * self.influence_vectors["role"]
        )
        
        # 计算与核心价值观的相似度
        similarity = np.dot(user_influence, self.core_values_vector) / (
            np.linalg.norm(user_influence) * np.linalg.norm(self.core_values_vector) + 1e-10
        )
        
        # 如果方向相反，投影到正交补空间
        if similarity < 0:
            # 计算正交补向量
            projection = user_influence - similarity * self.core_values_vector
            user_influence = projection
        
        # 应用幅值截断
        norm = np.linalg.norm(user_influence)
        if norm > 0.2:
            user_influence = (user_influence / norm) * 0.2
        
        # 逆映射回用户参数
        # 这里使用简单的线性映射，实际应用中可能需要更复杂的模型
        safe_creativity = float(np.dot(user_influence, self.influence_vectors["creativity"]))
        safe_detail = float(np.dot(user_influence, self.influence_vectors["detail"]))
        safe_safety = float(np.dot(user_influence, self.influence_vectors["safety"]))
        safe_role = role  # 角色预设保持不变
        
        # 确保参数在有效范围内
        safe_creativity = max(self.creativity_range[0], min(self.creativity_range[1], safe_creativity))
        safe_detail = max(self.detail_attention_range[0], min(self.detail_attention_range[1], safe_detail))
        safe_safety = max(self.safety_margin_range[0], min(self.safety_margin_range[1], safe_safety))
        
        return {
            "creativity": safe_creativity,
            "detail": safe_detail,
            "safety": safe_safety,
            "role": safe_role
        }
    
    def generate_user_bias(self, user_params: Dict, seq_len: int) -> Dict:
        """
        生成用户偏置
        
        Args:
            user_params: 安全投影后的用户参数
            seq_len: 序列长度
            
        Returns:
            用户偏置
        """
        # 基于用户参数生成偏置
        # 这里使用简单的实现，实际应用中可能需要更复杂的模型
        creativity = user_params.get("creativity", 0.0)
        detail = user_params.get("detail", 0.5)
        
        # 创建偏置矩阵
        bias = np.zeros((seq_len, seq_len))
        
        # 创造力影响：增加远距离token的注意力
        for i in range(seq_len):
            for j in range(seq_len):
                distance = abs(i - j)
                if distance > 0:
                    bias[i, j] += creativity * (1.0 / (distance + 1))
        
        # 细节关注度：增加局部token的注意力
        for i in range(seq_len):
            for j in range(max(0, i - 2), min(seq_len, i + 3)):
                bias[i, j] += detail * 0.5
        
        return {
            "bias": bias.tolist()
        }
    
    def validate_core_values_integrity(self) -> bool:
        """
        验证核心价值观完整性
        
        Returns:
            核心价值观是否完整
        """
        # 检查词汇表是否加载成功
        if not self.negative_vocab or not self.positive_vocab:
            logger.error("Vocabulary not loaded properly")
            return False
        
        # 检查词汇表哈希是否有效
        if not self.negative_vocab_hash or not self.positive_vocab_hash:
            logger.error("Vocabulary hash not generated properly")
            return False
        
        # 检查核心价值观向量是否有效
        if self.core_values_vector is None or len(self.core_values_vector) != 5:
            logger.error("Core values vector not valid")
            return False
        
        return True
    
    def get_vocab_info(self) -> Dict:
        """
        获取词汇表信息
        
        Returns:
            词汇表信息
        """
        return {
            "negative_vocab_size": len(self.negative_vocab),
            "positive_vocab_size": len(self.positive_vocab),
            "negative_vocab_hash": self.negative_vocab_hash,
            "positive_vocab_hash": self.positive_vocab_hash
        }
