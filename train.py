import os
import logging
import wandb
from tensorflow.keras import models, layers, callbacks
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from single_pixel_locate_train import *
# Set up logging
log_file_path = "log/training_log.txt"
logging.basicConfig(filename=log_file_path, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

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

model = create_pixel_localization_model()

# Training
try:
    # Training the model with early stopping and learning rate scheduler
    history = model.fit(x=train_images, y=train_coor, batch_size=32, steps_per_epoch=len(train_images) // 32,
                        epochs=10, validation_split=0.2, callbacks=[lr_scheduler, early_stopping])

    # Log training information
    logging.info("Training completed successfully.")
    wandb.log({"Training Completed": True})

except Exception as e:
    # Log error information
    logging.error(f"Error during training: {str(e)}")
    wandb.log({"Training Error": str(e)})

# Save the trained model
model.save("Trained_model/pixel_localization_model_1.h5")

# Save the training log to WandB
wandb.save(log_file_path)
