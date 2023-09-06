import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s-%(funcName)s",
    datefmt=""
)

class Logger:
    def __init__(self):
        pass

    def info(self, msg: str):
        logging.info(msg)
    
    def warning(self, msg: str):
        logging.warning(msg)

    def error(self, msg: str):
        logging.error(msg)
