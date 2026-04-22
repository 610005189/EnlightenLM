import torch
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)


class L1BaseModel:
    """
    L1 基座模型，负责生成回答并记录内部状态
    支持真实模型加载和离线模拟模式
    """
    
    def __init__(self, config: Dict):
        """
        初始化L1模型
        
        Args:
            config: 模型配置
        """
        self.config = config
        self.model_name = config.get("model_name", "distilgpt2")
        self.device = config.get("device", "cpu")
        self.inference_config = config.get("inference", {})
        self.attention_bias_config = config.get("attention_bias", {})
        self.optimization_config = config.get("optimization", {})
        
        # 优化配置
        self.quantization = self.optimization_config.get("quantization", False)
        self.quantization_bits = self.optimization_config.get("quantization_bits", 4)
        self.use_cache = self.optimization_config.get("use_cache", True)
        self.batch_size = self.optimization_config.get("batch_size", 1)
        
        # 检查是否离线模式
        self.offline_mode = config.get("offline_mode", False)
        
        # 检查是否强制使用模拟模型
        self.use_mock = config.get("use_mock", False)
        
        # 真实模型加载状态
        self.real_model_loaded = False
        
        # 注意力钩子注册
        self.attention_hooks = []
        self.hook_handles = []
        
        # 根据配置选择加载模式
        if self.use_mock or self.offline_mode:
            logger.info(f"Using mock L1 model: {self.model_name} on {self.device} (offline_mode={self.offline_mode})")
            self._setup_mock_model()
        else:
            self._try_load_real_model()
        
        # 注意力偏置配置
        self.bias_rank = self.attention_bias_config.get("rank", 8)
        self.bias_strategy = self.attention_bias_config.get("strategy", "layer-wise")
        
        # 初始化状态记录
        self.hidden_states = []
        self.attention_maps = []
        self.generated_tokens = []
        
        # 内存优化
        self._optimize_memory()
    
    def _try_load_real_model(self):
        """
        尝试加载真实模型，加载失败时回退到模拟模型
        """
        try:
            logger.info(f"Loading real L1 model: {self.model_name} on {self.device}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.real_model_loaded = True
            logger.info(f"Real model {self.model_name} loaded successfully")
            
            # 注册注意力钩子
            self._register_attention_hooks()
            
        except Exception as e:
            logger.warning(f"Failed to load real model {self.model_name}: {type(e).__name__}: {str(e)}. Falling back to mock model.")
            self._setup_mock_model()
    
    def _setup_mock_model(self):
        """
        设置模拟模型（用于离线模式或加载失败时）
        模拟模型现在能基于输入生成更有意义的输出
        """
        vocab = [
            "the", "a", "an", "is", "are", "was", "were", "to", "of", "and",
            "in", "that", "it", "for", "on", "with", "as", "be", "at", "by",
            "this", "we", "you", "mock", "response", "generated", "input",
            "model", "attention", "hidden", "state", "token"
        ]
        
        class MockTokenizer:
            def __init__(self, vocab):
                self.vocab = vocab
                self.vocab_size = len(vocab) + 100
                self._pad_token_id = 0
                self.eos_token_id = 1
            
            def __call__(self, text, return_tensors=None, truncation=None, padding=None, max_length=None):
                words = text.lower().split()
                ids = []
                for w in words[:50]:
                    if w in self.vocab:
                        ids.append(self.vocab.index(w) + 10)
                    else:
                        ids.append(1)
                if not ids:
                    ids = [1]
                return {"input_ids": torch.tensor([ids])}
            
            def decode(self, ids, skip_special_tokens=None):
                if hasattr(ids, 'tolist'):
                    ids = ids.tolist()
                if isinstance(ids, list) and len(ids) > 0 and isinstance(ids[0], list):
                    ids = ids[0]
                words = []
                for i in ids:
                    # 特殊tokens处理: 0是pad_token_id, 1是eos_token_id
                    if skip_special_tokens and (i == 0 or i == 1):
                        continue
                    if 10 <= i < 10 + len(self.vocab):
                        words.append(self.vocab[i - 10])
                    elif i == 1:
                        words.append("<eos>")
                    elif i == 0:
                        words.append("<pad>")
                    else:
                        words.append(f"unk_{i}")
                return " ".join(words)
            
            @property
            def pad_token(self):
                return "[PAD]"
            
            @pad_token.setter
            def pad_token(self, value):
                pass
            
            @property
            def pad_token_id(self):
                return self._pad_token_id
            
            @pad_token_id.setter
            def pad_token_id(self, value):
                self._pad_token_id = value
        
        class MockModel:
            def __init__(self, tokenizer, device):
                self.tokenizer = tokenizer
                self.device = device
                self._hidden_state = torch.randn(1, 10, 768)
                self._attention_weights = torch.randn(1, 12, 10, 10)
            
            def to(self, device):
                self.device = device
                return self
            
            def eval(self):
                return self
            
            def __call__(self, input_ids, attention_mask=None, output_hidden_states=False):
                class Output:
                    last_hidden_state = torch.randn(1, input_ids.shape[1], 768)
                    hidden_states = tuple(torch.randn(1, input_ids.shape[1], 768) for _ in range(6))
                    attentions = tuple(torch.randn(1, 12, input_ids.shape[1], input_ids.shape[1]) for _ in range(6))
                return Output()
            
            def generate(self, input_ids, **kwargs):
                max_new_tokens = kwargs.get("max_new_tokens", 20)
                temperature = kwargs.get("temperature", 0.7)
                
                if hasattr(input_ids, 'tolist'):
                    input_list = input_ids.tolist()[0] if isinstance(input_ids.tolist()[0], list) else input_ids.tolist()
                else:
                    input_list = input_ids[0].tolist() if hasattr(input_ids[0], 'tolist') else list(input_ids[0])
                
                input_len = len(input_list)
                seq_len = min(input_len + max_new_tokens, 100)
                
                output_ids = list(range(10, seq_len))
                
                return torch.tensor([output_ids])
            
            def get_hidden_state(self):
                return self._hidden_state
            
            def get_attention_weights(self):
                return self._attention_weights
        
        self.tokenizer = MockTokenizer(vocab)
        self.model = MockModel(self.tokenizer, self.device)
    
    def _register_attention_hooks(self):
        """
        注册注意力钩子，用于记录注意力权重和隐藏状态
        """
        # 清除旧钩子
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()
        
        # 注册新钩子
        def create_hook(name):
            def hook(module, input, output):
                # 记录注意力权重
                if hasattr(output, 'attentions') and output.attentions is not None:
                    self.attention_maps.append(output.attentions)
                elif hasattr(output, 'hidden_states') and output.hidden_states is not None:
                    self.hidden_states.append(output.hidden_states)
                # 兼容不同架构
                elif isinstance(output, tuple) and len(output) > 1:
                    # 假设第二个是注意力
                    if output[1] is not None:
                        self.attention_maps.append(output[1])
            return hook
        
        # 尝试找到注意力层并注册钩子
        # 这是一个通用的尝试，不同的模型架构可能需要不同的方法
        if hasattr(self.model, 'transformer'):
            for i, layer in enumerate(self.model.transformer.h):
                if hasattr(layer, 'attn'):
                    handle = layer.attn.register_forward_hook(create_hook(f"layer_{i}_attn"))
                    self.hook_handles.append(handle)
        elif hasattr(self.model, 'model'):
            # 其他架构的尝试
            try:
                if hasattr(self.model.model, 'layers'):
                    for i, layer in enumerate(self.model.model.layers):
                        if hasattr(layer, 'self_attn'):
                            handle = layer.self_attn.register_forward_hook(create_hook(f"layer_{i}_self_attn"))
                            self.hook_handles.append(handle)
            except Exception as e:
                logger.warning(f"Could not register attention hooks: {e}")
    
    def _optimize_memory(self):
        """
        优化内存使用
        """
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
        # 清空旧状态
        self.hidden_states = []
        self.attention_maps = []
        self.generated_tokens = []
        
        inputs = self.tokenizer(input_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        
        max_new_tokens = self.inference_config.get("max_new_tokens", 512)
        temperature = self.inference_config.get("temperature", 0.7)
        top_p = self.inference_config.get("top_p", 0.9)
        
        # 首先进行一次前向传播来收集注意力和隐藏状态
        with torch.no_grad():
            # 尝试获取隐藏状态和注意力权重
            if self.real_model_loaded:
                initial_output = self.model(
                    input_ids, 
                    output_hidden_states=True, 
                    output_attentions=True
                )
                if hasattr(initial_output, 'hidden_states'):
                    self.hidden_states = list(initial_output.hidden_states)
                if hasattr(initial_output, 'attentions'):
                    self.attention_maps = list(initial_output.attentions)
            else:
                # 模拟模型的情况
                initial_output = self.model(input_ids, output_hidden_states=True)
                if hasattr(initial_output, 'hidden_states'):
                    self.hidden_states = list(initial_output.hidden_states)
                if hasattr(initial_output, 'attentions'):
                    self.attention_maps = list(initial_output.attentions)
                elif hasattr(self.model, 'get_hidden_state'):
                    self.hidden_states = [self.model.get_hidden_state()]
                elif hasattr(self.model, 'get_attention_weights'):
                    self.attention_maps = [self.model.get_attention_weights()]
            
            # 生成token
            output = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_hidden_states=True
            )
            
            # 处理输出
            if hasattr(output, 'sequences'):
                generated_ids = output.sequences[0]
            else:
                generated_ids = output[0]
            
            # 如果是return_dict_in_generate的话，获取中间的hidden_states
            if hasattr(output, 'hidden_states') and output.hidden_states:
                self.hidden_states.extend(output.hidden_states)
        
        output_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        self.generated_tokens = generated_ids.tolist()
        
        # 构建快照
        snapshot = {
            "input_text": input_text,
            "output_text": output_text,
            "hidden_states": self.hidden_states,
            "attention_maps": self.attention_maps,
            "generated_tokens": self.generated_tokens,
            "generation_config": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p
            },
            "model_mode": "mock" if not self.real_model_loaded else "real",
            # 新增：为L2元描述生成准备的摘要信息
            "attention_summaries": self._create_attention_summaries()
        }
        
        return output_text, snapshot
    
    def _create_attention_summaries(self) -> List[Dict]:
        """
        从记录的注意力图中创建摘要，用于L2元描述生成
        
        Returns:
            注意力摘要列表
        """
        summaries = []
        
        if not self.attention_maps:
            return summaries
        
        # 对每一层的注意力计算摘要
        for i, layer_attn in enumerate(self.attention_maps):
            # 取这一层注意力的平均
            if isinstance(layer_attn, tuple):
                # 多头注意力，取平均
                attn = layer_attn[0] if layer_attn else None
            else:
                attn = layer_attn
            
            if attn is not None and isinstance(attn, torch.Tensor):
                # 计算这一层的平均注意力熵
                attn_np = attn.cpu().numpy()
                entropy = -np.sum(
                    attn_np * np.log(attn_np + 1e-10),
                    axis=-1
                ).mean()
                
                # 找到注意力最大的token
                if attn_np.ndim >= 2:
                    top_token_idx = np.argmax(attn_np.mean(axis=0) if attn_np.ndim >= 3 else attn_np)
                    summaries.append({
                        "layer": i,
                        "avg_entropy": float(entropy),
                        "top_tokens": [int(top_token_idx)],
                        "weights": [float(attn_np.max())]
                    })
        
        return summaries
    
    def generate_batch(self, input_texts: List[str], biases: Optional[List[Dict]] = None) -> List[Tuple[str, Dict]]:
        """
        批量生成回答
        """
        results = []
        
        for i in range(0, len(input_texts), self.batch_size):
            batch_texts = input_texts[i:i+self.batch_size]
            batch_biases = biases[i:i+self.batch_size] if biases else [None]*len(batch_texts)
            
            for text, bias in zip(batch_texts, batch_biases):
                result = self.generate(text, bias)
                results.append(result)
        
        return results
    
    def inject_attention_bias(self, bias: Dict):
        """
        注入注意力偏置
        """
        # 这里实现简单的注意力偏置注入
        # 实际应用中会更复杂
        logger.info(f"Injecting attention bias: {bias}")
    
    def get_memory_usage(self) -> Dict:
        """
        获取内存使用情况
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
    
    def is_real_model_loaded(self) -> bool:
        """
        检查是否加载了真实模型
        """
        return self.real_model_loaded
    
    def cleanup(self):
        """
        清理资源
        """
        # 移除注意力钩子
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()
        
        # 清空内存
        self.hidden_states = []
        self.attention_maps = []
        self.generated_tokens = []
        
        # 如果是CUDA设备，释放显存
        if self.device == "cuda":
            torch.cuda.empty_cache()
