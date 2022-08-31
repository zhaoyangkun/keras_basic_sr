import logging
import os


def create_logger(log_dir_path, log_file_name, logging_name):
    """
    创建日志
    """
    # 若日志目录不存在，则创建
    if not os.path.isdir(log_dir_path):
        os.makedirs(log_dir_path)
    # 若日志文件不存在，则创建
    log_file_path = os.path.join(log_dir_path, log_file_name)
    if not os.path.exists(log_file_path):
        os.mknod(log_file_path)

    # 获取 logger 对象
    logger = logging.getLogger(logging_name)
    # 输出 Debug 及以上级别的信息
    logger.setLevel(level=logging.DEBUG)

    if not logger.handlers:
        # 获取日志文件句柄，并设置日志级别
        file_handler = logging.FileHandler(log_file_path, encoding="UTF-8")
        file_handler.setLevel(logging.INFO)

        # 获取流句柄，并设置日志级别
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.DEBUG)

        # 设置日志文件格式
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        # stream_handler.setFormatter(formatter)

        # 为 logger 对象添加句柄
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger
