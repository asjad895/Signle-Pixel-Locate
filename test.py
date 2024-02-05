import os
import logging
import tensorflow as tf
import numpy as np
import pandas as pd
from single_pixel_locate_train import *
import matplotlib.pyplot as plt
import cv2

# Set up logging
log_file_path = "log/evaluation_log.txt"
logging.basicConfig(filename=log_file_path, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load the model
model_path = "/Trained_model/Face_single_pixel_1.h5"
loaded_model = tf.keras.models.load_model(model_path)

# Evaluate
try:
    test_loss = loaded_model.evaluate(test_images, test_coor)
    print("Test Loss on unseen data:", test_loss)
    logging.info("Evaluation completed successfully.")
except Exception as e:
    logging.error(f"Error during evaluation: {str(e)}")

# Predict on test images
try:
    y_pred = loaded_model.predict(test_images)
    # Randomly show results for 5 images
    show_test_result('Data/dataset_lfw_single_coor/test', n_samples=5)
    logging.info("Random test results displayed.")
except Exception as e:
    logging.error(f"Error during result display: {str(e)}")

# Save results to CSV
try:
    y_pred_list = list(zip(y_pred[:, 0], y_pred[:, 1]))
    y_test_list = list(zip(test_coor[:, 0], test_coor[:, 1]))

    image_paths = [os.path.join('Data/Face_data/test', image_file) for image_file in os.listdir('Data/Face_data/test')]

    save_df_to_csv(image_paths, y_test_list, y_pred_list, 'test_result.csv')
    logging.info("Results saved to CSV.")
except Exception as e:
    logging.error(f"Error during saving results to CSV: {str(e)}")

# Save the evaluation log
logging.shutdown()
