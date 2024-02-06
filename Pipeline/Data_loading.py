import os
import logging
import zipfile
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import random
import shutil
from Help import *
# Set up logging
log_file_path = os.path.join("log", "data_loading_log.txt")
logging.basicConfig(filename=log_file_path, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
ColoredOutput.log_message("Data Loading Started....",'BLUE',True)
# Function to unzip file
def unzip_file(zip_path, extract_path):
    """A function for unzipping the zip data and saving it in a new directory."""
    os.makedirs(extract_path, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
    except Exception as e:
        logging.error(f"Error during unzipping: {str(e)}")
        ColoredOutput.log_message(f"Error during unzipping: {str(e)}", 'RED')

# Unzip the file
zip_path = os.path.join("Data", "face_dataset.zip")
print(zip_path)
extract_path = os.path.join("Data", "Face_data")

# Function to check image size and resize if needed
def check_size(file):
    """Checks the size of an image and resizes it if it's smaller than a target size."""
    try:
        img = cv2.imread(file)
        height, width, _ = img.shape
        target_size = (256, 256)
        if height < target_size[0] or width < target_size[1]:
            img = cv2.resize(img, target_size)
            cv2.imwrite(file, img)
    except Exception as e:
        logging.error(f"Error during image size checking and resizing: {str(e)}")
        ColoredOutput.log_message(f"Error during image size checking and resizing: {str(e)}", color='RED')

# Function to copy and check image size
def copy_check_size(images, num_train, num_val, num_test, train_folder_path, test_folder_path, val_folder_path):
    """Copies and checks the size of images, distributing them into train, validation, and test sets."""
    try:
        for img_name in images[:num_train]:
            src_path = img_name[0]
            check_size(src_path)
            img = img_name[1]
            dst_path = os.path.join(train_folder_path, img)
            shutil.copy(src_path, dst_path)

        for img_name in images[num_train:(num_train + num_val)]:
            src_path = img_name[0]
            check_size(src_path)
            dst_path = os.path.join(val_folder_path, img_name[1])
            shutil.copy(src_path, dst_path)

        for img_name in images[(num_train + num_val):]:
            src_path = img_name[0]
            check_size(src_path)
            dst_path = os.path.join(test_folder_path, img_name[1])
            shutil.copy(src_path, dst_path)

        logging.info("Image copying and size checking completed successfully.")
        ColoredOutput.log_message("Image copying and size checking completed successfully","BLUE")
    except Exception as e:
        logging.error(f"Error during image copying and size checking: {str(e)}")
        ColoredOutput.log_message("Image copying and size checking failed", "RED")

def data_loading():
    try:
        unzip_file(zip_path,extract_path)
        logging.info("Data unzipped successfully.")
        ColoredOutput.log_message("Data unzipped successfully.", color='GREEN')
        ColoredOutput.log_message("Data Loaded successfully","GREEN",True)
    except Exception as e:
        ColoredOutput.log_message("Data unzipped failed.", 'RED')

# data_loading()
logging.shutdown()