import logging
from datetime import datetime
import os

LOG_DIR = 'logs'

CURRENT_TIMESTAMP = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

LOG_FILE_NAME = f'log_{CURRENT_TIMESTAMP}.log'

os.makedir(LOG_DIR, exist_ok = True)

LOG_FILE_PATH = os.join(LOG_DIR, LOG_FILE_NAME)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    filemode='w',
    format = '[%(asctime)s] %(name)s - %(levelname)s %(message)s',
    level= logging.INFO
)