import os
import logging
from logging.handlers import TimedRotatingFileHandler


def get_logger(name, log_dir='log'):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    file_name = os.path.join(log_dir, '{}.info.log'.format(name))
    file_handler = TimedRotatingFileHandler(file_name, when='D', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger
