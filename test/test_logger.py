import sys

sys.path.append("./")
from util.logger import create_logger

# 构建日志对象
logger = create_logger("./result/srgan/log", "srgan.log", "srgan")

# 输出不同级别的信息
logger.debug("debug")
logger.info("info")
logger.warning("warning")
logger.error("error")
print("The output of print and logger is different!")

