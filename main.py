import json
import os

import tensorflow as tf

from model.esrgan import ESRGAN
from model.ha_esrgan import HA_ESRGAN
from model.real_esrgan import RealESRGAN
from model.srcnn import SRCNN
from model.srgan import SRGAN
from model.vdsr import VDSR
from util.toml import parse_toml

# 日志级别
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

# 不使用 GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def setup_gpu():
    """
    设置 GPU
    """
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        # 设置 GPU 内存自动增长
        tf.config.experimental.set_memory_growth(gpu, True)
        # tf.config.set_logical_device_configuration(
        #     gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=1024)]
        # )


def build_sr_model(config):
    """
    构建超分模型
    """
    # 获取模型名称
    model_name = config["model"]["model_name"].lower()
    model_list = [
        "srcnn",
        "vdsr",
        "srgan",
        "esrgan",
        "real-esrgan",
        "ha-esrgan",
    ]

    # 校验模型名称
    if model_name not in model_list:
        raise ValueError(
            "The model name is not corrected, please enter 'srcnn', 'vdsr', 'srgan', 'esrgan', 'real-esrgan' or 'ha-esrgan'."
        )

    # 获取数据集配置
    dataset_config = config["dataset"]
    # 获取模型配置
    model_config = config["model"][model_name]
    model_config["model_name"] = model_name

    # 合并字典
    model_config = {**dataset_config, **model_config}

    # 格式化输出配置信息
    print(
        json.dumps(
            model_config,
            indent=4,
            ensure_ascii=False,
            sort_keys=False,
            separators=(",", ":"),
        ))

    # 构建模型
    sr_model = None
    if model_name == "srcnn":
        sr_model = SRCNN(**model_config)
    elif model_name == "vdsr":
        sr_model = VDSR(**model_config)
    elif model_name == "srgan":
        sr_model = SRGAN(**model_config)
    elif model_name == "esrgan":
        sr_model = ESRGAN(**model_config)
    elif model_name == "ha-esrgan":
        sr_model = HA_ESRGAN(**model_config)
    elif model_name == "real-esrgan":
        sr_model = RealESRGAN(**model_config)

    return sr_model


if __name__ == "__main__":
    setup_gpu()  # 设置 GPU
    config = parse_toml("./config/config.toml")  # 读取 toml 配置文件
    sr_model = build_sr_model(config)  # 构建超分模型

    mode = config["model"]["mode"]  # 获取训练模式
    if mode == "pretrain":
        sr_model.pretrain()  # 开始预训练
    elif mode == "train":
        sr_model.train()  # 开始训练
    else:
        raise ValueError(
            "Unsupported mode, please set the mode to 'pretrain' or 'train'.")
