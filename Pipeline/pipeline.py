if __name__ == "__main__" and __package__ is None:
    __package__ = "Pipeline"

from Pipeline.Data_loading import *
from Pipeline.Data_preprocessing import *
from Pipeline.test import test_model
from Pipeline.train import *
from Pipeline.test import *
import logging
import logging.config
import sys
# *******************Data Pipeline*********************************************
#  tracking for debug
# Load the logging configuration from the file
logging.config.fileConfig('logging_config.ini')
# Get a logger for your module
logger = logging.getLogger("log/pipeline")
logger.info("**********************Starting Pipeline********************")
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
        history_save(history=history)
        logger.info(f"Training completed successfully(Model Pipeline). with this hyperparametres\n: {hyper_p}")
    except Exception as e:
        logger.error("Error in calling training",str(e))
    if len(os.listdir("Trained_model"))>0:
        try:
            test_model()
            logger.info("Testing completed successfully")
        except Exception as e:
            logger.error("Error in calling test_model",str(e))
except Exception as e:
    logger.error(f"Error running  pipeline{e}",exc_info=True)

artifact = wandb.Artifact('model', type='model')

# Add files to the artifact
artifact.add_file('Trained_model/pixel_localization_model_1.h5')
# Save the artifact
wandb.log_artifact(artifact)

wandb.finish()