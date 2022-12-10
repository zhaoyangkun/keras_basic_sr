import sys

sys.path.append("./")
from util.toml import parse_toml

config = parse_toml("./config/config.toml")  # 读取 toml 配置文件
evaluate_config = config["evaluate_dataset"]
for key, config_item in evaluate_config.items():
    if config_item["is_count"]:
        print(key, config_item["is_count"], type(config_item["is_count"]))
