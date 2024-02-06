import os
import logging
import wandb
from tensorflow.keras import models, layers, callbacks
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from single_pixel_locate_train import *
from Pipeline.Data_loading import *
from Pipeline.Data_preprocessing import *
import logging.config
import sys

#  tracking for debug
# Load the logging configuration from the file
logging.config.fileConfig('logging_config.ini')
# Get a logger for your module
logger = logging.getLogger("log/train")
logger.info("...........starting training....................")
# Set up logger
log_file_path = "log/training_log.txt"
# logger.basicConfig(filename=log_file_path, level=logger.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize WandB for experiment tracking
wandb.init(project="pixel_localization_experiment", sync_tensorboard=True)

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
    logger.info('Creating model... successfully')
    # learning rate scheduler for dynamic optimization when performance will not improve after 3 epochs
    lr_scheduler = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-6)
    # early stopping
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

except Exception as e:
    logger.error('Error creating model:', e)

try:
    train_images, train_coor = annotate('Data/Face_data/train')
    test_images, test_coor = annotate('Data/Face_data/test')
    val_images, val_coor = annotate('Data/Face_data/val')
    logger.info(f"Data Statistic\n train X {len(train_images)}, y {len(train_coor)}\n test X {len(test_images)}, y {len(test_coor)}\n val X {len(val_images)},y {len(val_coor)}")
    ColoredOutput.log_message(f"Data Statistic\n train X {len(train_images)}, y {len(train_coor)}\n test X {len(test_images)}, y {len(test_coor)}\n val X {len(val_images)},y {len(val_coor)}","CYAN",True)
    ColoredOutput.log_message("Data preprocess completed","GREEN",True)
except Exception as e:
    logger.error(f"Error during image collection and annotation: {str(e)}")
    ColoredOutput.log_message(f"Error during image collection and annotation: {str(e)}","RED",True)


# Training
try:
    # Training the model with early stopping and learning rate scheduler
    history = model.fit(x=train_images, y=train_coor, batch_size=32, steps_per_epoch=len(train_images) // 32,
                        epochs=10, validation_split=0.2, callbacks=[lr_scheduler, early_stopping])

    # Log training information
    logger.info("Training completed successfully.")
    wandb.log({"Training Completed": True})

except Exception as e:
    # Log error information
    logger.error(f"Error during training: {str(e)}")
    wandb.log({"Training Error": str(e)})

# Save the trained model
try:
    p="Trained_model/pixel_localization_model_1.h5"
    model.save(p)
    logger.info(f"Model saved successfully to path: {p}\n")
except Exception as e:
    logger.error(f"Error during model saving: {str(e)}\n")

# Save the training log to WandB
wandb.save('log/my_app.lo')
