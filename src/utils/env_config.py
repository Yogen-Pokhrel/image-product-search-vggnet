import os
from dotenv import load_dotenv

# load environment variables from .env file
load_dotenv()

class EnvConfig:
    def __init__(self, env_file='.env') -> None:
        # load environment variables from the specified file
        load_dotenv(env_file)

    @property
    def log_level(self):
        return os.getenv('LOG_LEVEL', 'DEBUG').upper()

environment_variables = EnvConfig()