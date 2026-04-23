#!/usr/bin/env python3
"""
Phase 1 概念验证测试脚本
验证 L1/L2/L3 核心组件的基本功能
使用千问 API 进行真实测试
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from enlighten.l1_generation import L1Generation, SimplifiedL1
from enlighten.l2_working_memory import L2WorkingMemory, SimplifiedL2
from enlighten.l3_controller import L3Controller, SimplifiedL3

# 测试配置
CONFIG = {
    "use_api": True,  # 使用 API 模式
    "api_provider": "Qianwen",
    "api_key": os.environ.get("DASHSCOPE_API_KEY"),
    "test_questions": [
        "中国的首都是哪里？",
        "《红楼梦》的作者是谁？",
        "水在标准大气压下的沸点是多少摄氏度？",
        "太阳系中最大的行星是什么？",
        "电话是谁发明的？"
    ],
    "ground_truths": [
        "北京",
        "曹雪芹",
        "100",
        "木星",
        "贝尔"
    ]
}

class APIClient:
    """千问 API 客户端"""
    def __init__(self, api_key):
        self.api_key = api_key
    
    def generate(self, prompt, max_tokens=30):
        """调用千问 API 生成回答"""
        try:
            import dashscope
            from dashscope import Generation
            
            dashscope.api_key = self.api_key
            
            response = Generation.call(
                model="qwen-turbo",
                prompt=prompt,
                max_tokens=max_tokens,
                result_format='message'
            )
            
            if response.status_code == 200:
                content = response.output.choices[0].message.content
                return content
            else:
                print(f"API调用失败: {response.message}")
                return f"API错误: {response.message}"
        except ImportError:
            print("警告: dashscope未安装，使用模拟模式")
            return f"模拟回答: {prompt}"
        except Exception as e:
            print(f"API调用异常: {e}")
            return f"异常: {str(e)}"

def test_l1_l2_l3_integration():
    """测试 L1/L2/L3 集成"""
    print("=== Phase 1 概念验证测试 ===")
    print("测试 L1/L2/L3 核心组件集成...")
    
    # 检查 API 密钥
    api_key = CONFIG["api_key"]
    if not api_key:
        raise ValueError("API 密钥未设置，请设置环境变量 DASHSCOPE_API_KEY")
    
    # 初始化 API 客户端
    api_client = APIClient(api_key)
    
    # 初始化简化版组件
    simplified_l1 = SimplifiedL1(embed_dim=512, task_bias_dim=64)
    simplified_l2 = SimplifiedL2(memory_size=512, embedding_dim=512)
    simplified_l3 = SimplifiedL3()
    
    # 模拟输入
    batch_size = 1
    seq_len = 10
    embed_dim = 512
    
    # 测试问题
    for i, (question, ground_truth) in enumerate(zip(CONFIG["test_questions"], CONFIG["ground_truths"])):
        print(f"\n=== 测试问题 {i+1}: {question} ===")
        
        # 步骤 1: 调用 API 生成回答
        response = api_client.generate(question)
        print(f"API 回答: {response}")
        
        # 步骤 2: 模拟 L1 处理
        print("L1 处理...")
        query = torch.randn(batch_size, seq_len, embed_dim)
        key = torch.randn(batch_size, seq_len, embed_dim)
        value = torch.randn(batch_size, seq_len, embed_dim)
        task_bias = torch.randn(batch_size, 64)
        
        l1_output, van_event = simplified_l1(query, key, value, task_bias)
        print(f"L1 输出形状: {l1_output.shape}")
        print(f"VAN 事件: {van_event}")
        
        # 步骤 3: 模拟 L2 处理
        print("L2 处理...")
        attention_weights = torch.randn(batch_size, seq_len, seq_len)
        sparse_kv, entropy_stats = simplified_l2(l1_output, attention_weights)
        print(f"L2 稀疏 KV 形状: {sparse_kv.shape}")
        print(f"L2 熵统计: {entropy_stats}")
        
        # 步骤 4: 模拟 L3 处理
        print("L3 处理...")
        control_signals = simplified_l3.forward(entropy_stats, van_event)
        print(f"L3 调控信号: {control_signals}")
        
        # 步骤 5: 验证回答
        correct = ground_truth in response
        print(f"回答正确: {correct}")
        print(f"预期答案: {ground_truth}")

def test_core_components():
    """测试核心组件"""
    print("\n=== 核心组件测试 ===")
    
    # 测试 L1 生成层
    print("\n测试 L1 生成层...")
    try:
        # 加载一个小模型进行测试
        tokenizer = AutoTokenizer.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
        model = AutoModelForCausalLM.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
        
        l1 = L1Generation(model, tokenizer, {
            "embed_dim": 768,
            "num_heads": 12,
            "task_bias_dim": 128
        })
        
        # 测试输入
        input_ids = tokenizer("你好", return_tensors="pt").input_ids
        attention_mask = torch.ones_like(input_ids)
        
        output = l1.forward(input_ids, attention_mask)
        print(f"L1 输出: {output}")
        print("L1 生成层测试成功!")
    except Exception as e:
        print(f"L1 测试失败: {e}")
    
    # 测试 L2 工作记忆层
    print("\n测试 L2 工作记忆层...")
    try:
        l2 = L2WorkingMemory(memory_size=512, embedding_dim=768)
        
        # 测试输入
        hidden_states = torch.randn(1, 10, 768)
        attention_weights = torch.randn(1, 10, 10)
        
        output = l2.forward(hidden_states, attention_weights)
        print(f"L2 输出: {output}")
        print("L2 工作记忆层测试成功!")
    except Exception as e:
        print(f"L2 测试失败: {e}")
    
    # 测试 L3 元控制器
    print("\n测试 L3 元控制器...")
    try:
        l3 = L3Controller()
        
        # 测试输入
        entropy_stats = {
            "mean": 0.3,
            "variance": 0.02,
            "trend": -0.1,
            "current": 0.2
        }
        
        control_signals = l3.forward(entropy_stats, van_event=False)
        print(f"L3 调控信号: {control_signals}")
        print("L3 元控制器测试成功!")
    except Exception as e:
        print(f"L3 测试失败: {e}")

if __name__ == "__main__":
    print("开始 Phase 1 概念验证测试...")
    # 测试集成
    test_l1_l2_l3_integration()
    
    # 测试核心组件
    test_core_components()
    
    print("\n=== Phase 1 概念验证测试完成 ===")