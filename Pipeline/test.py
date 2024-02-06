import os
import tensorflow as tf
import numpy as np
import pandas as pd
# from single_pixel_locate_train import *
import matplotlib.pyplot as plt
from Data_preprocessing import *
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

def show_test_result(loaded_model, folder_path, num_images=5, target_size=(256, 256)):
    """Shows test results with annotated points on images."""
    try:
        # Get a list of all image files in the folder
        all_images = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        # Randomly select num_images from the list
        selected_images = random.sample(all_images, min(num_images, len(all_images)))
        # Display and process each selected image
        fig, axes = plt.subplots(1, len(selected_images), figsize=(20, 5))
        for i, img_name in enumerate(selected_images):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path,)
            img = cv2.resize(img, (256, 256))
            height, width, _ = img.shape
            img = np.array(img) / 255.0
            points = loaded_model.predict(img.reshape(-1, 256, 256, 3))
            img = img * 255
            img = img.astype(np.uint8)
            for point in points:
                x, y = point
                cv2.circle(img, (int(x), int(y)), 5, (10, 33, 255), -1)
            cv2.circle(img, (130, 160), 3, (100, 200, 100), -1)
            axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axes[i].set_title(f"red-Pred\nGreen-Actual")
            axes[i].axis('off')
        plt.savefig('Test_Data_Result/Result_test_data_1.png')
        plt.show()
        logging.info("Test results displayed successfully.")
    except Exception as e:
        logging.error(f"Error during showing test results: {str(e)}")

# Function to save DataFrame to CSV
def save_df_to_csv(img_array, pred_coor_array, actual_coor_array, filename):
    """Saves a DataFrame with columns for image paths, predicted coordinates, and actual coordinates to a CSV file."""
    try:
        data = {
            'img': img_array,
            'pred_coor': pred_coor_array,
            'actual_coor': actual_coor_array
        }
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        logging.info(f"CSV file '{filename}' saved successfully.")
    except Exception as e:
        logging.error(f"Error during saving DataFrame to CSV: {str(e)}")


def test_model():
    # Load the model
    try:
        model_path = os.path.join("Trained_model","Face_single_pixel_1.h5")
        loaded_model = tf.keras.models.load_model(model_path)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error("Error loading model: {}".format(str(e)))

    # Evaluate
    try:
        test_images, test_coor = annotate('Data/Face_data/test')
        test_loss = loaded_model.evaluate(test_images, test_coor)
        print("Test Loss on unseen data:", test_loss)
        logger.info("Evaluation completed successfully.")
    except Exception as e:
        logger.error("Error during evaluation: {}".format(str(e)))

    # Predict on test images
    try:
        y_pred = loaded_model.predict(test_images)
        # Randomly show results for 5 images
        show_test_result('Data/face_data/test', n_samples=5)
        logger.info("Random test results displayed.")
    except Exception as e:
        logger.error("Error during result display: {}".format(str(e)))

    # Save results to CSV
    try:
        y_pred_list = list(zip(y_pred[:, 0], y_pred[:, 1]))
        y_test_list = list(zip(test_coor[:, 0], test_coor[:, 1]))

        image_paths = [os.path.join('Data/Face_data/test', image_file) for image_file in os.listdir('Data/Face_data/test')]

        save_df_to_csv(image_paths, y_test_list, y_pred_list, 'Test_Data_Result/test_result.csv')
        logger.info("Results saved to CSV.")
    except Exception as e:
        logger.error("Error during saving results to CSV: {}".format(str(e)))

    # Save the evaluation log
    # logger.shutdown()
