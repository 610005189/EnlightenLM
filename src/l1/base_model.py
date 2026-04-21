import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


class L1BaseModel:
    """
    L1 基座模型，负责生成回答并记录内部状态
    """
    
    def __init__(self, config: Dict):
        """
        初始化L1模型
        
        Args:
            config: 模型配置
        """
        self.config = config
        self.model_name = config.get("model_name", "Qwen/Qwen2-7B")
        self.device = config.get("device", "cuda")
        self.inference_config = config.get("inference", {})
        self.attention_bias_config = config.get("attention_bias", {})
        self.optimization_config = config.get("optimization", {})
        
        # 优化配置
        self.quantization = self.optimization_config.get("quantization", False)
        self.quantization_bits = self.optimization_config.get("quantization_bits", 4)
        self.use_cache = self.optimization_config.get("use_cache", True)
        self.batch_size = self.optimization_config.get("batch_size", 1)
        
        # 加载模型和分词器
        logger.info(f"Loading L1 model: {self.model_name}")
        
        # 配置量化
        model_kwargs = {}
        if self.quantization:
            logger.info(f"Using {self.quantization_bits}-bit quantization")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=self.quantization_bits == 4,
                load_in_8bit=self.quantization_bits == 8,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            model_kwargs["quantization_config"] = quantization_config
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            **model_kwargs
        ).to(self.device)
        self.model.eval()
        
        # 注意力偏置配置
        self.bias_rank = self.attention_bias_config.get("rank", 8)
        self.bias_strategy = self.attention_bias_config.get("strategy", "layer-wise")
        
        # 初始化状态记录
        self.hidden_states = []
        self.attention_maps = []
        self.generated_tokens = []
        
        # 内存优化
        self._optimize_memory()
    
    def _optimize_memory(self):
        """
        优化内存使用
        """
        # 启用内存优化
        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("Memory optimization enabled")
    
    def generate(self, input_text: str, bias: Optional[Dict] = None) -> Tuple[str, Dict]:
        """
        生成回答并记录内部状态
        
        Args:
            input_text: 输入文本
            bias: 注意力偏置
            
        Returns:
            生成的回答和内部状态快照
        """
        # 重置状态记录
        self.hidden_states = []
        self.attention_maps = []
        self.generated_tokens = []
        
        # 分词
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        seq_len = input_ids.shape[1]
        
        # 生成配置
        max_new_tokens = self.inference_config.get("max_new_tokens", 512)
        temperature = self.inference_config.get("temperature", 0.7)
        top_p = self.inference_config.get("top_p", 0.9)
        
        # 注册钩子以记录隐藏状态和注意力图
        hooks = []
        # 只记录最后一层的隐藏状态和注意力图，减少内存使用
        last_layer_idx = len(self.model.model.layers) - 1
        for i, layer in enumerate(self.model.model.layers):
            if i == last_layer_idx:
                # 记录隐藏状态
                def hook_fn_hidden(module, input, output, layer_idx=i):
                    # 分段平均池化，每16个token一个向量
                    hidden_state = output[0].detach().cpu().numpy()
                    batch_size, seq_len, hidden_dim = hidden_state.shape
                    
                    # 分段平均池化
                    pooled_hidden = []
                    for b in range(batch_size):
                        for j in range(0, seq_len, 16):
                            end = min(j + 16, seq_len)
                            segment = hidden_state[b, j:end]
                            if segment.shape[0] > 0:
                                pooled = np.mean(segment, axis=0)
                                # 投影到256维
                                if hidden_dim > 256:
                                    # 简单线性投影
                                    pooled = np.dot(pooled, np.random.randn(hidden_dim, 256))
                                pooled_hidden.append(pooled)
                    self.hidden_states.append(np.array(pooled_hidden))
                
                # 记录注意力图
                def hook_fn_attention(module, input, output, layer_idx=i):
                    attention = output[1].detach().cpu().numpy()  # (batch, heads, seq_len, seq_len)
                    batch_size, num_heads, seq_len, _ = attention.shape
                    
                    # 计算平均注意力熵
                    avg_entropy = 0
                    for b in range(batch_size):
                        for h in range(num_heads):
                            # 计算每个位置的熵
                            for pos in range(seq_len):
                                probs = attention[b, h, pos]
                                entropy = -np.sum(probs * np.log(probs + 1e-10))
                                avg_entropy += entropy
                    avg_entropy /= (batch_size * num_heads * seq_len)
                    
                    # 记录每步top-5关注token
                    top_tokens = []
                    for b in range(batch_size):
                        for pos in range(seq_len):
                            # 对所有头取平均
                            avg_attention = np.mean(attention[b, :, pos], axis=0)
                            # 获取top-5
                            top_indices = np.argsort(avg_attention)[-5:][::-1]
                            top_weights = avg_attention[top_indices]
                            top_tokens.append({
                                "position": pos,
                                "tokens": top_indices.tolist(),
                                "weights": top_weights.tolist()
                            })
                    
                    self.attention_maps.append({
                        "layer": layer_idx,
                        "avg_entropy": avg_entropy,
                        "top_tokens": top_tokens
                    })
                
                # 注册钩子
                hooks.append(layer.register_forward_hook(hook_fn_hidden))
                if hasattr(layer.self_attention, "attention"):
                    hooks.append(layer.self_attention.attention.register_forward_hook(hook_fn_attention))
        
        # 生成回答
        with torch.no_grad():
            # 如果提供了偏置，注入到注意力层
            if bias is not None:
                # 根据模型类型应用偏置
                model_type = self.model_name.lower()
                if "qwen" in model_type:
                    self._inject_bias_qwen(bias)
                elif "llama" in model_type:
                    self._inject_bias_llama(bias)
                elif "deepseek" in model_type:
                    self._inject_bias_deepseek(bias)
                else:
                    # 通用偏置注入
                    self._inject_bias_generic(bias)
            
            # 生成
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                use_cache=self.use_cache
            )
        
        # 移除钩子
        for hook in hooks:
            hook.remove()
        
        # 解码生成的文本
        generated_text = self.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        
        # 生成快照
        snapshot = self._create_snapshot(input_text, generated_text)
        
        # 释放显存
        torch.cuda.empty_cache()
        
        return generated_text, snapshot
    
    def generate_batch(self, input_texts: List[str], biases: Optional[List[Dict]] = None) -> List[Tuple[str, Dict]]:
        """
        批量生成回答
        
        Args:
            input_texts: 输入文本列表
            biases: 注意力偏置列表
            
        Returns:
            生成的回答和内部状态快照列表
        """
        results = []
        
        # 分批处理
        for i in range(0, len(input_texts), self.batch_size):
            batch_texts = input_texts[i:i+self.batch_size]
            batch_biases = biases[i:i+self.batch_size] if biases else [None]*len(batch_texts)
            
            for text, bias in zip(batch_texts, batch_biases):
                result = self.generate(text, bias)
                results.append(result)
        
        return results
    
    def _inject_bias_qwen(self, bias: Dict):
        """
        注入偏置到Qwen模型
        
        Args:
            bias: 注意力偏置
        """
        # Qwen模型的偏置注入实现
        # 这里需要根据Qwen模型的具体架构进行调整
        pass
    
    def _inject_bias_llama(self, bias: Dict):
        """
        注入偏置到LLaMA模型
        
        Args:
            bias: 注意力偏置
        """
        # LLaMA模型的偏置注入实现
        # 这里需要根据LLaMA模型的具体架构进行调整
        pass
    
    def _inject_bias_deepseek(self, bias: Dict):
        """
        注入偏置到DeepSeek模型
        
        Args:
            bias: 注意力偏置
        """
        # DeepSeek模型的偏置注入实现
        # 这里需要根据DeepSeek模型的具体架构进行调整
        pass
    
    def _inject_bias_generic(self, bias: Dict):
        """
        通用偏置注入实现
        
        Args:
            bias: 注意力偏置
        """
        # 通用偏置注入实现
        # 这里提供一个通用的实现思路
        pass
    
    def _create_snapshot(self, input_text: str, generated_text: str) -> Dict:
        """
        创建内部状态快照
        
        Args:
            input_text: 输入文本
            generated_text: 生成的文本
            
        Returns:
            快照字典
        """
        # 压缩隐藏状态
        compressed_hidden = []
        for hidden in self.hidden_states:
            if len(hidden) > 0:
                # 进一步压缩，只保留最后10个向量
                compressed = hidden[-10:] if len(hidden) > 10 else hidden
                compressed_hidden.append(compressed.tolist())
        
        # 压缩注意力图
        compressed_attention = []
        for attention in self.attention_maps:
            # 只保留平均熵和最后5个位置的top-5
            if attention.get("top_tokens"):
                top_tokens = attention["top_tokens"][-5:] if len(attention["top_tokens"]) > 5 else attention["top_tokens"]
                compressed_attention.append({
                    "layer": attention["layer"],
                    "avg_entropy": attention["avg_entropy"],
                    "top_tokens": top_tokens
                })
        
        snapshot = {
            "input_text": input_text,
            "output_text": generated_text,
            "attention_summaries": compressed_attention,
            "hidden_state_pooled": compressed_hidden,
            "generation_temperature": self.inference_config.get("temperature", 0.7),
            "generation_steps": len(self.generated_tokens) if self.generated_tokens else 0
        }
        
        return snapshot
    
    def inject_attention_bias(self, bias: Dict):
        """
        注入注意力偏置
        
        Args:
            bias: 注意力偏置
        """
        # 实现注意力偏置注入
        # 这需要根据具体模型架构进行调整
        pass
    
    def get_memory_usage(self) -> Dict:
        """
        获取内存使用情况
        
        Returns:
            内存使用情况
        """
        if self.device == "cuda":
            allocated = torch.cuda.memory_allocated() / 1024**3
            cached = torch.cuda.memory_reserved() / 1024**3
            return {
                "allocated": allocated,
                "cached": cached,
                "device": self.device
            }
        else:
            return {
                "device": self.device,
                "message": "Memory usage not available for non-CUDA devices"
            }
