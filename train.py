from tensorflow.keras import models, layers, callbacks
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#define data or import from other scripts as i did in notebook already i will not do much experiment
from Notebook import *

train_images,train_coor=annotate('/kaggle/working/dataset_lfw_single_coor/train')
test_images,test_coor=annotate('/kaggle/working/dataset_lfw_single_coor/test')
val_images,val_coor=annotate('/kaggle/working/dataset_lfw_single_coor/val')
# learning rate scheduler for dynamic optimization when performance will not improve after 3 epochs
lr_scheduler = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-6)
# early stopping
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Data augmentation if needed but i will not use it now
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

def Pixel_localization_model(input_shape=(256, 256, 3)):
   """Creates a convolutional neural network for regression tasks.
   Args:
       input_shape (tuple, optional): The shape of the input images.
           Defaults to (256, 256, 3).
   Returns:
       keras.models.Model: The compiled model.
   """
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
model=Pixel_localization_model()

#Train
# Training the model with early stopping and learning rate scheduler
history = model.fit(x=train_images,y=train_coor,batch_size=32,steps_per_epoch=len(train_images) // 32,
                    epochs=100,validation_data=(val_images, val_coor),callbacks=[lr_scheduler, early_stopping])