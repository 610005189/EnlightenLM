"""模型接口异常定义"""


class ModelLoadError(Exception):
    """模型加载错误"""
    pass


class ModelInferenceError(Exception):
    """模型推理错误"""
    pass


class ModelConfigurationError(Exception):
    """模型配置错误"""
    pass