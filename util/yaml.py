import yaml


def parse_yaml(file_path):
    """解析 yaml 文件

    Args:
        file_path (str): yaml 文件路径

    Returns:
        dict: 解析结果
    """
    config = None
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(e)
    return config
