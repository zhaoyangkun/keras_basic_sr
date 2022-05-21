from model.srgan import SRGAN
from model.esrgan import ESRGAN
from util.yaml import parse_yaml


def build_sr_model(config):
    """
    构建超分模型
    """
    train_config = config["train"]
    model_name = train_config["model_name"].lower()
    print("\n", "train_config:", train_config)

    sr_model = None
    if model_name == "srgan":
        sr_model = SRGAN(**train_config)
    elif model_name == "esrgan":
        train_config.pop("rdb_num")
        sr_model = ESRGAN(**train_config)
    elif model_name == "real-esrgan":
        pass
    elif model_name == "adm-esrgan":
        pass
    else:
        raise ValueError(
            "The model name is not corrected, Please Enter srgan, esrgan, real-esrgan or adm-esrgan"
        )

    return sr_model


if __name__ == "__main__":
    config = parse_yaml("./config/config.yaml")  # 读取 yaml 配置文件
    sr_model = build_sr_model(config)  # 构建超分模型
    sr_model.train()  # 开始训练
