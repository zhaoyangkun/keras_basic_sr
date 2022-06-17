import sys

sys.path.append("./")
from util.logger import create_logger

# 构建日志对象
logger = create_logger("./result/test/log", "test.log", "test")

# 输出不同级别的信息
logger.debug("debug1")
logger.debug("debug2")
logger.debug("debug3")
logger.info("info")
logger.warning("warning")
logger.error("error")
# print("The output of print and logger is different!")

