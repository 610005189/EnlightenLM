import yaml
import logging
from src.l1.base_model import L1BaseModel
from src.l2.self_description_model import L2SelfDescriptionModel
from src.l3.meta_attention_controller import L3MetaAttentionController

# 配置日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 加载配置
def load_config():
    config_path = "config/config.yaml"
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise

config = load_config()

logger.info("Testing model loading...")

# 测试L1模型
logger.info("Loading L1 model...")
try:
    l1_model = L1BaseModel(config.get("l1", {}))
    logger.info("L1 model loaded successfully!")
    # 测试生成
    output, snapshot = l1_model.generate("Hello, how are you?")
    logger.info(f"L1 generated: {output}")
except Exception as e:
    logger.error(f"L1 model error: {e}")

# 测试L2模型
logger.info("Loading L2 model...")
try:
    l2_model = L2SelfDescriptionModel(config.get("l2", {}))
    logger.info("L2 model loaded successfully!")
    # 测试生成元描述
    meta_description = l2_model.generate_meta_description({"input_text": "Hello", "output_text": "Hi"})
    logger.info(f"L2 generated meta description: {meta_description}")
except Exception as e:
    logger.error(f"L2 model error: {e}")

# 测试L3模型
logger.info("Loading L3 model...")
try:
    l3_model = L3MetaAttentionController(config.get("l3", {}))
    logger.info("L3 model loaded successfully!")
    # 测试生成任务偏置
    task_bias = l3_model.generate_task_bias("Hello, how are you?", 100)
    logger.info(f"L3 generated task bias: {task_bias}")
except Exception as e:
    logger.error(f"L3 model error: {e}")

logger.info("Model loading test completed!")
