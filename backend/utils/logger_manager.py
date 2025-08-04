# utils/logger_manager.py

import logging
import os
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime

class LoggerManager:
    LOG_DIR = "logs"
    LOG_LEVELS = {
        "info": logging.INFO,
        "error": logging.ERROR,
        "debug": logging.DEBUG,
        "warning": logging.WARNING
    }

    def __init__(self, module_name, log_type="general"):
        self.module_name = module_name
        self.log_type = log_type.lower()
        self.logger = logging.getLogger(f"{module_name}-{log_type}")
        self.logger.setLevel(self.LOG_LEVELS.get(log_type, logging.INFO))

        if not self.logger.handlers:
            self._setup_handler()

    def _setup_handler(self):
        os.makedirs(self.LOG_DIR, exist_ok=True)

        file_path = os.path.join(self.LOG_DIR, f"{self.log_type}_{self.module_name}.log")

        handler = TimedRotatingFileHandler(
            file_path, when="midnight", backupCount=7, encoding='utf-8'
        )

        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def log(self, level, message):
        log_func = getattr(self.logger, level.lower(), self.logger.info)
        log_func(message)

    def info(self, message): self.log("info", message)
    def error(self, message): self.log("error", message)
    def debug(self, message): self.log("debug", message)
    def warning(self, message): self.log("warning", message)
