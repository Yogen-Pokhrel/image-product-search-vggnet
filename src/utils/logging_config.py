import logging
from .env_config import environment_variables

class LoggerFacade:
    def __init__(self, name:str, log_file:str = None):
        self.logger = logging.getLogger(name)

        #set log level
        log_level = environment_variables.log_level
        self.logger.setLevel(getattr(logging, log_level, logging.DEBUG))

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)


    
    def debug(self, message: str):
        self.logger.debug(message)

    def info(self, message: str):
        self.logger.info(message)

    def warning(self, message: str):
        self.logger.warning(message)

    def error(self, message: str):
        self.logger.error(message)

    def critical(self, message: str):
        self.logger.critical(message)

# create default logger 
logger = LoggerFacade(name='Image Retrieval')