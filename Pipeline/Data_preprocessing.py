if __name__ == "__main__" and __package__ is None:
    __package__ = "Pipeline"

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
from utils.Help import *
from Pipeline.Data_loading import *

# Set up logging
log_file_path = os.path.join("log", "data_preprocessing_log.txt")
logging.basicConfig(filename=log_file_path, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
ColoredOutput.log_message("Data preprocessing Started....",'BLUE',True)
# Function to create train, validation, and test sets
def train_test(folders):
    """Creates train, validation, and test sets from a given image dataset."""
    images = []
    try:
        for root, dirs, files in os.walk(folders):
            for dir in dirs:
                p = os.path.join(folders, dir)
                for root, dir, files in os.walk(p):
                    for f in files:
                        i = os.path.join(p, f)
                        images.append((i, f))
        images = images[0:-1]
        random.shuffle(images)
        train_ratio = 0.7
        val_ratio = 0.10
        num_train = int(train_ratio * len(images))
        num_val = int(val_ratio * len(images))
        num_test = len(images) - num_val - num_train

        dataset = 'Data/Face_data'
        train_folder_path = os.path.join(dataset, 'train')
        test_folder_path = os.path.join(dataset, 'test')
        val_folder_path = os.path.join(dataset, 'val')
        os.makedirs(train_folder_path, exist_ok=True)
        os.makedirs(test_folder_path, exist_ok=True)
        os.makedirs(val_folder_path, exist_ok=True)

        copy_check_size(images, num_train, num_val, num_test, train_folder_path, test_folder_path, val_folder_path)
        logging.info(f"{dataset} created for new data folder.")
        logging.info(f"Dataset created with {num_train} images for training, {num_test} images for testing, {num_val} for validation.")
    except Exception as e:
        logging.error(f"Error during train-test set creation: {str(e)}")
        ColoredOutput.log_message(f"Error during train-test set creation: {str(e)}","RED")
# Run the data preprocessing
try:
    # train_test('')
    print("later")
except Exception as e:
    logging.error(f"Error during data preprocessing: {str(e)}")
    


"""**Data visualization**"""

import matplotlib.pyplot as plt
def show_random_images(folder_path, num_images=5, target_size=(256, 256)):
    """Displays a random sample of images from a given folder.
    Args:
       folder_path (str): The path to the folder containing images.
       num_images (int, optional): The number of images to display. Defaults to 5.
       target_size (tuple, optional): The desired size to resize images to. Defaults to (256, 256).
   Raises:
       FileNotFoundError: If the specified folder path is not found.
   """
    # Get a list of all image files in the folder
    all_images = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    # print(len(all_images))
    # Randomly select num_images from the list
    selected_images = random.sample(all_images, min(num_images, len(all_images)))
    # Display and process each selected image
    fig, axes = plt.subplots(1, len(selected_images), figsize=(20, 5))
    for i,img_name in enumerate(selected_images):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        # Check the shape of the image
        height, width, _ = img.shape
        # If the shape is less than target_size, resize the image
        if height < target_size[0] or width < target_size[1]:
            img = cv2.resize(img, target_size)
        # Display the image
        axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[i].set_title(f"Shape: {img.shape}")
        axes[i].axis('off')
    plt.savefig("Analysis/Training data.png")
# show_random_images('Data/Face_data/train',)

"""**Data Annotation**


1.   we can randomely annotate a single coordinate on images anywhere like (100,134).
2.   we can draw a bounding boxes around single pixel with 1 pixel hieght an width.
3.  draw a coordinate on specific location like eye,nose etc on images using tools.


"""

# Function to annotate points on an image
def annotate_points(image_path, points):
    """Annotates an image with circles at specified points."""
    try:
        img = cv2.imread(image_path)
        for point in points:
            x, y = point
            cv2.circle(img, (x, y), 5, (10, 100, 10), -1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("Annotated Points")
        plt.axis('off')
        plt.show()
        logging.info("Image annotated successfully.")
    except cv2.error as e:
        logging.error(f"Error during image annotation: {str(e)}")

# Annotate points on a sample image
try:
    annotate_points('', [(130, 160)])
except Exception as e:
    logging.error(f"Error during sample image annotation: {str(e)}")

# Function to create a zip archive of a folder
def create_zip_archive(src_folder, dst_path):
    """Creates a zip archive of the specified folder."""
    try:
        shutil.make_archive(dst_path, 'zip', src_folder)
        logging.info(f"Zip archive created successfully at {dst_path}.zip")
    except Exception as e:
        logging.error(f"Error during zip archive creation: {str(e)}")

# Create a zip archive of the folder
try:
    # create_zip_archive('/content/bottle_dataset', '/content/dataset_lfw_single_coor')
    print("......................later")
except Exception as e:
    logging.error(f"Error during zip archive creation for the dataset: {str(e)}")

# Function to collect images and annotations from a directory
def annotate(dir):
    """Collects images and placeholders for annotations from a directory."""
    images = []
    annotations = []
    dataset_path = dir
    if os.path.isdir(dataset_path):
        try:
            for image_file in os.listdir(dataset_path):
                image_path = os.path.join(dataset_path, image_file)
                if os.path.isfile(image_path) and image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image = cv2.imread(image_path)
                    image = cv2.resize(image, (256, 256))
                    images.append(image)
                    annotations.append((130, 160))
        except Exception as e:
            logging.error(f"Error during image collection and annotation: {str(e)}")
    else:
        logging.error(f"{dataset_path} is not a directory.")
    
    images = np.array(images) / 255.0
    annotations = np.array(annotations)
    return images, annotations

def data_preprocess():
    # train_test('')
    # Collect images and annotations
    show_random_images('Data/Face_data/train',)
    try:
        train_images, train_coor = annotate('Data/Face_data/train')
        test_images, test_coor = annotate('Data/Face_data/test')
        val_images, val_coor = annotate('Data/Face_data/val')
        logging.info(f"Data Statistic\n train X {len(train_images)}, y {len(train_coor)}\n test X {len(test_images)}, y {len(test_coor)}\n val X {len(val_images)},y {len(val_coor)}")
        ColoredOutput.log_message(f"Data Statistic\n train X {len(train_images)}, y {len(train_coor)}\n test X {len(test_images)}, y {len(test_coor)}\n val X {len(val_images)},y {len(val_coor)}","CYAN",True)
        ColoredOutput.log_message("Data preprocess completed","GREEN",True)
    except Exception as e:
        logging.error(f"Error during image collection and annotation: {str(e)}")
        ColoredOutput.log_message(f"Error during image collection and annotation: {str(e)}","RED",True)

# data_preprocess()
logging.shutdown()