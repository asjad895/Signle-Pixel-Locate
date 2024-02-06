from Data_loading import *
from Data_preprocessing import *
import logging
import logging.config
import sys
# *******************Data Pipeline*********************************************
#  tracking for debug
# Load the logging configuration from the file
logging.config.fileConfig('logging_config.ini')
# Get a logger for your module
logger = logging.getLogger("log/pipeline")
logger.info("Starting Data Pipeline")
def setup_custom_logger(module_name):
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.DEBUG)

    log_file = os.path.join('log', f'{module_name}.log')

    handler = logging.FileHandler(log_file, 'w')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger

# logger = setup_custom_logger("pipeline")

# ***************Model Pipeline*******************************************
# train
# Test
#Tune model
try:
    data_loading()
    data_preprocess()
    logger.info('Loading and preprocessing data completed successfully(Data Pipeline).')
    logger.info('***********Running Model pipeline*************\n')
except Exception as e:
    logger.error(f"Error running Data pipeline{e}",exc_info=True)
