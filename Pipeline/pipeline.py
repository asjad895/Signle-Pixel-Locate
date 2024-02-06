from Data_loading import *
from Data_preprocessing import *
from test import test_model
from train import *
from test import *
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

# pipeline
try:
    data_loading()
    data_preprocess()
    logger.info('Loading and preprocessing data completed successfully(Data Pipeline).')
    logger.info('***********Running Model pipeline*************\n')
    try:
        history=train()
        history_save()
        logger.info("Training completed successfully(Model Pipeline). with this hyperparametres: " + hyper_p) if history else logger.info("Training completed successfully(Model Pipeline).")
    except Exception as e:
        logger.error("Error in calling training",str(e))
    if history or len(os.listdir("Trained_model"))>0:
        try:
            test_model()
            logger.info("Testing completed successfully")
        except Exception as e:
            logger.error("Error in calling test_model",str(e))
except Exception as e:
    logger.error(f"Error running Data pipeline{e}",exc_info=True)
