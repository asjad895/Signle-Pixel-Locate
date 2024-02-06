import os
import tensorflow as tf
import numpy as np
import pandas as pd
# from single_pixel_locate_train import *
import matplotlib.pyplot as plt
from Pipeline.Data_preprocessing import *
import cv2
import logging
import logging.config
import sys
# *******************Data Pipeline*********************************************
#  tracking for debug
# Load the logging configuration from the file
logging.config.fileConfig('logging_config.ini')
# Get a logger for your module
logger = logging.getLogger("log/Tesing")
logger.info("...........Running Testing............")
# Set up logger
log_file_path = "log/evaluation_log.txt"
# logger.basicConfig(filename=log_file_path, level=logger.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load the model
try:
    model_path = "/Trained_model/Face_single_pixel_1.h5"
    loaded_model = tf.keras.models.load_model(model_path)
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error("Error loading model:", str(e))

# Evaluate
try:
    test_images, test_coor = annotate('Data/Face_data/test')
    test_loss = loaded_model.evaluate(test_images, test_coor)
    print("Test Loss on unseen data:", test_loss)
    logger.info("Evaluation completed successfully.")
except Exception as e:
    logger.error(f"Error during evaluation: {str(e)}")

# Predict on test images
try:
    y_pred = loaded_model.predict(test_images)
    # Randomly show results for 5 images
    show_test_result('Data/dataset_lfw_single_coor/test', n_samples=5)
    logger.info("Random test results displayed.")
except Exception as e:
    logger.error(f"Error during result display: {str(e)}")

# Save results to CSV
try:
    y_pred_list = list(zip(y_pred[:, 0], y_pred[:, 1]))
    y_test_list = list(zip(test_coor[:, 0], test_coor[:, 1]))

    image_paths = [os.path.join('Data/Face_data/test', image_file) for image_file in os.listdir('Data/Face_data/test')]

    save_df_to_csv(image_paths, y_test_list, y_pred_list, 'test_result.csv')
    logger.info("Results saved to CSV.")
except Exception as e:
    logger.error(f"Error during saving results to CSV: {str(e)}")

# Save the evaluation log
# logger.shutdown()
