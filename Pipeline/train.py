if __name__ == "__main__" and __package__ is None:
    __package__ = "Pipeline"

import datetime
import os
import logging
import wandb
from dotenv import load_dotenv
import tensorflow as tf
from keras import models, layers, callbacks
from keras.utils import plot_model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
# from single_pixel_locate_train import *
from Pipeline.Data_loading import *
from Pipeline.Data_preprocessing import *
import logging.config
import sys

#  tracking for debug
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path)
# Load the logging configuration from the file
logging.config.fileConfig('logging_config.ini')
# Get a logger for your module
logger = logging.getLogger("log/train")
logger.info("...........starting training....................")
# Set up logger
log_file_path = "log/training_log.txt"
# logger.basicConfig(filename=log_file_path, level=logger.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize WandB for experiment tracking
# project=f"Experiment_Date_%s" %(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
project="CV_Track"
wandb_api_key=os.getenv('WANDB_API_KEY')
print(wandb_api_key)
os.environ["WANDB_API_KEY"]=wandb_api_key
try:
    wandb.init(project=project, sync_tensorboard=True)
except RuntimeError as e:
    logger.error("Failed to initialize WandB: %s" %(e))

# Data augmentation if needed (currently not used)
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Model
def create_pixel_localization_model(input_shape=(256, 256, 3)):
    """Creates a convolutional neural network for regression tasks."""
    model = models.Sequential()

    # Convolutional layers for feature extraction
    model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    # Flatten layer
    model.add(layers.Flatten())

    # Dense layers for regression output
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(2, activation='linear'))

    # Compile the model
    model.compile(optimizer=Adam(lr=1e-4), loss='mean_squared_error')
    return model
try:
    model = create_pixel_localization_model()
    model.summary()
    # plot_model(model, to_file='model_architecture.png', show_shapes=True)
    logger.info('Creating model... successfully')
    # learning rate scheduler for dynamic optimization when performance will not improve after 3 epochs
except Exception as e:
    logger.error('Error creating model:', e)

try:
    train_images, train_coor = annotate('Data/Face_data/train')
    test_images, test_coor = annotate('Data/Face_data/test')
    val_images, val_coor = annotate('Data/Face_data/val')
    logger.info(f"Data Statistic\n train X {len(train_images)}, y {len(train_coor)}\n test X {len(test_images)}, y {len(test_coor)}\n val X {len(val_images)},y {len(val_coor)}")
    ColoredOutput.log_message(f"Data Statistic\n train X {len(train_images)}, y {len(train_coor)}\n test X {len(test_images)}, y {len(test_coor)}\n val X {len(val_images)},y {len(val_coor)}","CYAN",True)
    ColoredOutput.log_message("Data splitting completed","GREEN",True)
except Exception as e:
    logger.error(f"Error during image collection and annotation: {str(e)}")
    ColoredOutput.log_message(f"Error during image collection and annotation: {str(e)}","RED",True)

global history
def train():
    # Training
    try:
        lr_scheduler = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-6)
    # early stopping
        early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        # Training the model with early stopping and learning rate scheduler
        history = model.fit(x=train_images, y=train_coor, batch_size=32, steps_per_epoch=len(train_images) // 32,
                        epochs=1, validation_split=0.2, callbacks=[lr_scheduler, early_stopping])

        # Log training information
        logger.info("Training completed successfully.")
    # wandb.log({"Training Completed": True})

    except Exception as e:
        # Log error information
        logger.error(f"Error during training: {str(e)}")
    # wandb.log({"Training Error": str(e)})

# Save the trained model
    try:
        p=os.path.join("Trained_model","pixel_localization_model_1.h5")
        model.save(p,save_format='h5')
        logger.info(f"Model saved successfully to path: {p}\n")
    except Exception as e:
        logger.error(f"Error during model saving: {str(e)}\n")

    return history
    
# Save the training log to WandB
# wandb.save('log/my_app.log')

def history_save(history):
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
    for i in range(len(history.history['lr'])):
        wandb.log({"Learning Rate": history.history['lr'][i]})
        wandb.log({"Epoch":i})
        wandb.log({"val_loss":history.history['val_loss'][i]})
        wandb.log({"train_loss":history.history['loss'][i]})
    
hyper_p={
    "batch_size": 32,
    "epochs": 10,
    "optimmizer":"Adam",
    "loss":"mean_squared_loss",
    "validation_split": 0.2,
    "lr_scheduler_factor": 0.1,
    "lr_scheduler_patience": 3,
    "lr_scheduler_min_lr": 1e-6,
    "early_stopping_patience": 5
}
wandb.config.update(hyper_p)
# Create an artifact

# train()