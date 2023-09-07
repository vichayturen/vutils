import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(process)d --- [%(threadName)s] %(filename)s-%(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def info(msg: str):
    logging.info(msg)

def warning(msg: str):
    logging.warning(msg)

def error(msg: str):
    logging.error(msg)

if __name__=="__main__":
    info('nihao')
    warning('azhe')
    error('hehe')
