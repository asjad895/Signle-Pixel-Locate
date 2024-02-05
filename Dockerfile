# Use an official TensorFlow runtime as a parent image
FROM tensorflow/tensorflow:latest

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Copy data files into the container
COPY Data/Face_data/ /app/data

# Install any dependencies needed for your script
# For example, if you have a requirements.txt file:
RUN pip install --no-cache-dir -r requirements.txt

# Define the command to run your scripts
CMD ["python", "single_pixel_locate_train.py"]
