#!/usr/bin/env python3
"""
配置文件加载测试脚本
直接测试_load_from_yaml函数的返回值
"""

import tempfile
from enlighten.config.loader import _load_from_yaml


def test_yaml_loading():
    """测试从YAML文件加载配置"""
    print("=== 测试YAML文件加载 ===")
    
    # 测试配置文件
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
        # 直接调用_load_from_yaml函数
        file_config = _load_from_yaml(temp_config_path)
        print(f"YAML文件加载结果: {file_config}")
        
        # 检查model_provider配置
        if 'model_provider' in file_config:
            print(f"Model Provider配置: {file_config['model_provider']}")
            print(f"Use Local Model: {file_config['model_provider'].get('use_local_model')}")
            print(f"Local Model Name: {file_config['model_provider'].get('local_model_name')}")
            print(f"API Provider: {file_config['model_provider'].get('api_provider')}")
            print(f"API Model: {file_config['model_provider'].get('api_model')}")
        else:
            print("ERROR: model_provider配置未加载！")
    finally:
        import os
        os.unlink(temp_config_path)
    
    print("\n=== 测试完成 ===")


if __name__ == "__main__":
    test_yaml_loading()
    print("\n所有测试完成！")