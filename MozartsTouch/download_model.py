import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from pathlib import Path
from transformers import AutoProcessor, MusicgenForConditionalGeneration, BlipForConditionalGeneration, AutoModelForCausalLM
from loguru import logger

# 设置环境变量和路径
# Set environment variables and paths
cwd = Path(__file__).resolve().parent
model_path = cwd / "model"
model_path.mkdir(parents=True, exist_ok=True)

# 通用加载和保存函数
# General load and save functions
def download_and_save_model(model_class, processor_class, model_name, save_dir):
    """
    下载模型和处理器并保存到指定路径
    :param model_class: 模型类
    :param processor_class: 处理器类
    :param model_name: 预训练模型名称
    :param save_dir: 保存目录

    # Download the model and processor, and save them to the specified path
    :param model_class: Model class
    :param processor_class: Processor class
    :param model_name: Name of the pre-trained model
    :param save_dir: Directory to save the files
    """
    
    model_save_path = save_dir / f"{model_name.split('/')[-1]}_model"
    processor_save_path =  save_dir / f"{model_name.split('/')[-1]}_processor"

    try:
        # logger.info(f"正在尝试加载模型和处理器: {model_name}...")
        logger.info(f"Attempting to load the model and processor: {model_name}...")
        model = model_class.from_pretrained(model_save_path, trust_remote_code=True)
        processor = processor_class.from_pretrained(processor_save_path,trust_remote_code=True)

        # logger.info(f"{model_name} 加载成功！")
        logger.info(f"{model_name} Loaded successfully!")
    except Exception as e:
        # logger.info(f"加载 {model_name} 时出错: {e}")
        logger.info(f"Error occurred while loading {model_name}: {e}")
        model = model_class.from_pretrained(model_name, trust_remote_code=True)
        processor = processor_class.from_pretrained(model_name, trust_remote_code=True)

        model.save_pretrained(model_save_path)
        processor.save_pretrained(processor_save_path)

# 下载模型
# Download the model
# download_and_save_model(MusicgenForConditionalGeneration, AutoProcessor, "facebook/musicgen-small", model_path)
# download_and_save_model(BlipForConditionalGeneration, AutoProcessor, "Salesforce/blip-image-captioning-base", model_path)
# download_and_save_model(MusicgenForConditionalGeneration, AutoProcessor, "facebook/musicgen-medium", model_path)
download_and_save_model(AutoModelForCausalLM, AutoProcessor, "microsoft/Florence-2-large", model_path)
