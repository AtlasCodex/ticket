'''
Author: AtlasCodex wenlin.xie@outlook.com
Date: 2024-07-01 16:49:49
LastEditors: AtlasCodex wenlin.xie@outlook.com
LastEditTime: 2024-07-02 10:42:52
FilePath: /ticket/logger.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import logging
from logging.handlers import RotatingFileHandler
import yaml

class Logger:
    def __init__(self, config_file):
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)

        log_config = config['logging']
        log_file = log_config['log_file']
        log_level = getattr(logging, log_config['log_level'])
        max_bytes = log_config['max_bytes']
        backup_count = log_config['backup_count']

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

        # Create a file handler
        file_handler = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
        file_handler.setLevel(log_level)

        # Create a formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # Add handlers to the logger
        self.logger.addHandler(file_handler)

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)
