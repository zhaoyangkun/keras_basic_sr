import json
from model.real_esrgan import RealESRGAN
from model.srgan import SRGAN
from model.esrgan import ESRGAN
from model.rs_esrgan import RS_ESRGAN
from util.toml import parse_toml

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def build_sr_model(config):
    """
    构建超分模型
    """
    # 获取模型名称
    model_name = config["model"]["model_name"].lower()

    # 校验模型名称
    if model_name not in ["srgan", "esrgan", "real-esrgan", "rs-esrgan"]:
        raise ValueError(
            "The model name is not corrected, Please Enter srgan, esrgan, real-esrgan or rs-esrgan"
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
        )
    )

    # 构建模型
    sr_model = None
    if model_name == "srgan":
        sr_model = SRGAN(**model_config)
    elif model_name == "esrgan":
        sr_model = ESRGAN(**model_config)
    elif model_name == "rs-esrgan":
        sr_model = RS_ESRGAN(**model_config)
    elif model_name == "real-esrgan":
        sr_model = RealESRGAN(**model_config)

    return sr_model


if __name__ == "__main__":
    config = parse_toml("./config/config.toml")  # 读取 toml 配置文件
    sr_model = build_sr_model(config)  # 构建超分模型

    # sr_model.pretrain()  # 开始预训练
    sr_model.train() # 开始训练
