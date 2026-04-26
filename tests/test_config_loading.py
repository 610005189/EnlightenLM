#!/usr/bin/env python3
"""
配置加载测试脚本
直接测试配置加载功能，验证model_provider配置是否正确加载
"""

import tempfile
from enlighten.config.loader import load_config


def test_config_loading():
    """测试配置加载功能"""
    print("=== 测试配置加载功能 ===")
    
    # 测试1: 默认配置
    print("\n1. 测试默认配置:")
    config = load_config("balanced")
    print(f"   Mode: {config.mode.value}")
    print(f"   Model Provider: {config.model_provider}")
    print(f"   Use Local Model: {config.model_provider.use_local_model}")
    print(f"   Local Model Name: {config.model_provider.local_model_name}")
    print(f"   API Provider: {config.model_provider.api_provider}")
    print(f"   API Model: {config.model_provider.api_model}")
    
    # 测试2: 配置文件指定本地模型
    print("\n2. 测试配置文件指定本地模型:")
    config_content = """
enlighten:
  mode: balanced
  components:
    model_provider:
      use_local_model: true
      local_model_name: "qwen2.5:7b"
      api_provider: "ollama"
      api_model: "llama3:70b"
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_content)
        temp_config_path = f.name
    
    try:
        config_local = load_config(config_path=temp_config_path)
        print(f"   Mode: {config_local.mode.value}")
        print(f"   Use Local Model: {config_local.model_provider.use_local_model}")
        print(f"   Local Model Name: {config_local.model_provider.local_model_name}")
        print(f"   API Provider: {config_local.model_provider.api_provider}")
        print(f"   API Model: {config_local.model_provider.api_model}")
    finally:
        import os
        os.unlink(temp_config_path)
    
    # 测试3: 配置文件指定API模式
    print("\n3. 测试配置文件指定API模式:")
    config_content2 = """
enlighten:
  mode: full
  components:
    model_provider:
      use_local_model: false
      api_provider: "ollama"
      api_model: "qwen2.5:14b"
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_content2)
        temp_config_path2 = f.name
    
    try:
        config_api = load_config(config_path=temp_config_path2)
        print(f"   Mode: {config_api.mode.value}")
        print(f"   Use Local Model: {config_api.model_provider.use_local_model}")
        print(f"   API Provider: {config_api.model_provider.api_provider}")
        print(f"   API Model: {config_api.model_provider.api_model}")
    finally:
        import os
        os.unlink(temp_config_path2)
    
    print("\n=== 测试完成 ===")


if __name__ == "__main__":
    test_config_loading()
    print("\n所有测试完成！")