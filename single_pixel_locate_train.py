import os
import logging
import zipfile
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import random
from single_pixel_locate_train import *
import shutil
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Set up logging
log_file_path = os.path.join("log","data_preprocessing_log.txt")
logging.basicConfig(filename=log_file_path, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Function to unzip file
def unzip_file(zip_path, extract_path):
    """A function for unzipping the zip data and saving it in a new directory."""
    os.makedirs(extract_path, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
    except Exception as e:
        logging.error(f"Error during unzipping: {str(e)}")

# Unzip the file
zip_path = os.path.join('Data','face_dataset.zip')
extract_path = 'Data/Face_data'
unzip_file(zip_path, extract_path)
logging.info("Data unzipped successfully.")

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
    except Exception as e:
        logging.error(f"Error during image copying and size checking: {str(e)}")

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
    print(len(all_images))
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
    plt.show()
show_random_images('Data/Face_data/train',)

"""**Data Annotation**


1.   we can randomely annotate a single coordinate on images anywhere like (100,134).
2.   we can draw a bounding boxes around single pixel with 1 pixel hieght an width.
3.  draw a coordinate on specific location like eye,nose etc on images using tools.


"""
# Save the logs to a file
logging.shutdown()
# Commented out IPython magic to ensure Python compatibility.

# Set up logging
log_file_path = os.path.join("log","model_building_log.txt")
logging.basicConfig(filename=log_file_path, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

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

# Collect images and annotations
try:
    train_images, train_coor = annotate('Data/Face_data/train')
    test_images, test_coor = annotate('Data/Face_data/test')
    val_images, val_coor = annotate('Data/Face_data/val')
    logging.info(f"Data Statistic\n train X {len(train_images)}, y {len(train_coor)}\n test X {len(test_images)}, y {len(test_coor)}\n val X {len(val_images)}, y {len(val_coor)}")
except Exception as e:
    logging.error(f"Error during image collection and annotation: {str(e)}")

# Function to build the model
def Pixel_localization_model(input_shape=(256, 256, 3)):
    """Creates a convolutional neural network for regression tasks."""
    try:
        model = models.Sequential()
        model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Conv2D(256, (3, 3), activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Flatten())

        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(2, activation='linear'))

        model.compile(optimizer=Adam(lr=1e-4), loss='mean_squared_error')
        logging.info("Model built successfully.")
        return model
    except Exception as e:
        logging.error(f"Error during model building: {str(e)}")

# Build the model
try:
    model = Pixel_localization_model()
    model.summary()
    plot_model(model, to_file='model_architecture.png', show_shapes=True)
except Exception as e:
    logging.error(f"Error during model building: {str(e)}")

# learning rate scheduler for dynamic optimization when performance will not improve after 3 epochs
lr_scheduler = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-6)
# early stopping
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Training the model with early stopping and learning rate scheduler
history = model.fit(x=train_images,y=train_coor,batch_size=32,steps_per_epoch=len(train_images) // 32,
                    epochs=1,validation_split=0.2,callbacks=[lr_scheduler, early_stopping])

# history.__dict__

# Save the model
model.save("Face_single_pixel.h5", save_format="tf")
# load
loaded_model = tf.keras.models.load_model("Face_single_pixel.h5")

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Loss comparison plot
axes[0].plot(history.history['loss'], label='Training Loss')
axes[0].plot(history.history['val_loss'], label='Validation Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Loss Curve - Training vs. Validation')
axes[0].grid(True)
axes[0].legend()

# Epoch vs LR plot
axes[1].plot(history.history['lr'], label='Learning Rate')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Learning Rate')
axes[1].set_title('Learning Rate over Epochs')
axes[1].grid(True)
axes[1].legend()
plt.tight_layout()
plt.savefig('training_analysis.png')
plt.show()

"""**Gradient visualzation**"""

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# !pip install landscapeviz

# import landscapeviz
# # build mesh and plot
# landscapeviz.build_mesh(model, (train_Images, train_coor), grid_length=40, verbose=0)
# landscapeviz.plot_contour(key="mean_squared_error")
# landscapeviz.plot_3d(key="mean_squared_error")
# Save the logs to a file
logging.shutdown()
# Set up logging
log_file_path = "log/test_model_log.txt"
logging.basicConfig(filename=log_file_path, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Function to evaluate the loaded model
def evaluate_loaded_model(loaded_model, test_images, test_coor):
    """Evaluates the loaded model on test data."""
    try:
        test_loss = loaded_model.evaluate(test_images, test_coor)
        logging.info(f"Test Loss on unseen data: {test_loss}")
    except Exception as e:
        logging.error(f"Error during model evaluation: {str(e)}")

# Function to show test results with annotated points
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
        plt.savefig('Result_test_data.png')
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

# Evaluate the loaded model
try:
    evaluate_loaded_model(loaded_model, test_images, test_coor)
except Exception as e:
    logging.error(f"Error during model evaluation: {str(e)}")

# Show test results with annotated points
try:
    show_test_result(loaded_model, '/Data/Face_data/test')
except Exception as e:
    logging.error(f"Error during showing test results: {str(e)}")

# Save DataFrame to CSV
y_pred=loaded_model.predict(test_images)
try:
    y_pred = list(zip(y_pred[:, 0], y_pred[:, 1]))
    y_test = list(zip(test_coor[:, 0], test_coor[:, 1]))
    l = []
    for image_file in os.listdir('/Data/Face_data/test'):
        image_path = os.path.join('/Data/Face_data/test', image_file)
        l.append(image_path)
    save_df_to_csv(l, y_test, y_pred, 'test_result.csv')
except Exception as e:
    logging.error(f"Error during saving DataFrame to CSV: {str(e)}")

# Save the logs to a file
logging.shutdown()
