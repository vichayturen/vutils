import logging
from logging import Logger


formatter = logging.Formatter(
    fmt="%(asctime)s %(levelname)8s %(process)d --- [%(threadName)s] %(filename)s-%(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
handler = logging.StreamHandler()
handler.setFormatter(formatter)
handler.setLevel(logging.INFO)

logger = Logger('vlogger')
logger.addHandler(handler)

if __name__ == "__main__":
    logger.info('你好')
