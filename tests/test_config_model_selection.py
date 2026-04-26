#!/usr/bin/env python3
"""
配置测试脚本
测试从配置文件和环境变量中读取模型提供者设置的功能
"""

import os
import tempfile
from pathlib import Path
from enlighten.config.loader import load_config
from enlighten.hybrid_architecture import HybridEnlightenLM


def test_config_based_model_selection():
    """测试基于配置的模型选择"""
    print("=== 测试基于配置的模型选择 ===")
    
    # 测试1: 使用默认配置（API模式）
    print("\n1. 测试默认配置（API模式）:")
    config = load_config("balanced")
    model = HybridEnlightenLM(config=config)
    print(f"   use_local_model: {model.use_local_model}")
    print(f"   local_model_name: {model.local_model_name}")
    print(f"   api_client: {model.api_client is not None}")
    
    # 测试2: 使用配置文件指定本地模型
    print("\n2. 测试配置文件指定本地模型:")
    config_content = """
enlighten:
  mode: balanced
  components:
    model_provider:
      use_local_model: true
      local_model_name: "qwen2.5:7b"
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_content)
        temp_config_path = f.name
    
    try:
        config_local = load_config(config_path=temp_config_path)
        model_local = HybridEnlightenLM(config=config_local)
        print(f"   use_local_model: {model_local.use_local_model}")
        print(f"   local_model_name: {model_local.local_model_name}")
    finally:
        os.unlink(temp_config_path)
    
    # 测试3: 使用环境变量覆盖配置
    print("\n3. 测试环境变量覆盖配置:")
    os.environ['ENLIGHTEN_MODE'] = 'lightweight'
    
    # 创建一个临时配置文件，然后用环境变量覆盖
    config_content2 = """
enlighten:
  mode: full
  components:
    model_provider:
      use_local_model: false
      api_model: "llama3:70b"
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_content2)
        temp_config_path2 = f.name
    
    try:
        config_env = load_config(config_path=temp_config_path2)
        model_env = HybridEnlightenLM(config=config_env)
        print(f"   Config mode: {config_env.mode.value}")
        print(f"   use_local_model: {model_env.use_local_model}")
        print(f"   api_client: {model_env.api_client is not None}")
    finally:
        os.unlink(temp_config_path2)
        if 'ENLIGHTEN_MODE' in os.environ:
            del os.environ['ENLIGHTEN_MODE']
    
    # 测试4: 显式参数覆盖配置
    print("\n4. 测试显式参数覆盖配置:")
    config_base = load_config("balanced")
    model_override = HybridEnlightenLM(
        config=config_base,
        use_local_model=True,
        local_model_name="gpt2"
    )
    print(f"   use_local_model: {model_override.use_local_model}")
    print(f"   local_model_name: {model_override.local_model_name}")
    
    print("\n=== 测试完成 ===")


def test_api_provider_config():
    """测试API提供者配置"""
    print("\n=== 测试API提供者配置 ===")
    
    # 测试不同API提供者的配置
    config_content = """
enlighten:
  mode: balanced
  components:
    model_provider:
      use_local_model: false
      api_provider: "ollama"
      api_model: "qwen2.5:14b"
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_content)
        temp_config_path = f.name
    
    try:
        config = load_config(config_path=temp_config_path)
        model = HybridEnlightenLM(config=config)
        print(f"API Provider: ollama")
        print(f"API Model: qwen2.5:14b")
        print(f"API Client: {model.api_client is not None}")
    finally:
        os.unlink(temp_config_path)
    
    print("\n=== API提供者测试完成 ===")


if __name__ == "__main__":
    test_config_based_model_selection()
    test_api_provider_config()
    print("\n所有测试完成！")