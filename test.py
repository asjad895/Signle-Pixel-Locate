import tensorflow as tf
from Notebook import *
import matplotlib.pyplot as plt
import opencv as cv2
import random,os
import numpy as np
import pandas as pd
# load
loaded_model = tf.keras.models.load_model("/kaggle/working/Face_single_pixel.h5")

# Evaluation
test_loss = loaded_model.evaluate(test_images, test_coor)
print("Test Loss on unseen data:", test_loss)
y_pred = loaded_model.predict(test_images)

def show_test_result(folder_path, num_images=5, target_size=(256, 256)):
    # Get a list of all image files in the folder
    all_images = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    # Randomly select num_images from the list
    selected_images = random.sample(all_images, min(num_images, len(all_images)))
    # Display and process each selected image
    fig, axes = plt.subplots(1, len(selected_images), figsize=(20, 5))
    for i,img_name in enumerate(selected_images):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path,)
        img=cv2.resize(img,(256,256))
        # Check the shape of the image
        height, width, _ = img.shape
        # Draw circles around the specified points
        img=np.array(img)
        img=img/255.0
        points=loaded_model.predict(img.reshape(-1,256,256,3))
        points=points
        img=img*255
        img = img.astype(np.uint8)
        for point in points:
            x, y = point
            cv2.circle(img, (int(x), int(y)), 5, (10, 33, 255),-1)  
#         original_coord
        cv2.circle(img,(130,160),3,(100,200,100),-1)
        axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[i].set_title(f"red-Pred\nGreen-Actual")
        axes[i].axis('off')
    plt.savefig('Result_test_data.png')
    plt.show()
show_test_result('/kaggle/working/dataset_lfw_single_coor/test',)

def save_df_to_csv(img_array, pred_coor_array, actual_coor_array, filename):
    """
    Saves a DataFrame with columns for image paths, predicted coordinates, and actual coordinates to a CSV file.

    Args:
        img_array (list): A list of image paths or image data.
        pred_coor_array (list): A list of predicted coordinates, where each element is a list or tuple of (x, y) coordinates.
        actual_coor_array (list): A list of actual coordinates, where each element is a list or tuple of (x, y) coordinates.
        filename (str): The name of the CSV file to save.
    """

    data = {
        'img': img_array,
        'pred_coor': pred_coor_array,  
        'actual_coor': actual_coor_array
    }
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    
y_pred=list(zip(y_pred[:, 0], y_pred[:, 1]))
y_test=list(zip(test_coor[:, 0], test_coor[:, 1]))

l=[]
for image_file in os.listdir('/kaggle/working/dataset_lfw_single_coor/test'):
    image_path = os.path.join('/kaggle/working/dataset_lfw_single_coor/test', image_file)
    l.append(image_path)
    
save_df_to_csv(l,y_test,y_pred,'test_result.csv')
