# HappySad Image Classifier

This repository contains a deep learning model built with TensorFlow and Keras for binary image classification, specifically classifying images as either "happy" or "sad." The model was trained on custom images and employs a simple yet effective Convolutional Neural Network (CNN) architecture.

## Model Architecture
The model is implemented as a Sequential CNN with the following layers:

1. **Convolutional Layers**: Three Conv2D layers with ReLU activations and varying filter sizes (16, 32, 16), followed by MaxPooling layers to capture spatial features and reduce dimensionality.
2. **Flatten Layer**: This layer flattens the feature maps into a 1D array, preparing the data for fully connected layers.
3. **Dense Layers**: A fully connected Dense layer with 256 units and ReLU activation, followed by a Dense layer with a sigmoid activation for binary classification.

## Data Preprocessing
- **Data Loading**: The images are loaded from a structured directory, where each subdirectory corresponds to a class label.
- **Dodgy Image Removal**: Before training, the script removes any images with unsupported formats or corrupted files.
- **Scaling**: Images are scaled to [0, 1] by dividing pixel values by 255.
- **Data Splitting**: The dataset is split into training, validation, and test sets with a 70-15-15 distribution.

## Model Training
The model is compiled using the Adam optimizer and trained with Binary Cross-Entropy as the loss function. During training:
- A TensorBoard callback logs the metrics for easy visualization.
- The model was trained for 20 epochs, and both training and validation loss were tracked.

## Performance Metrics
The model's performance is evaluated on the test set using Precision, Recall, and Accuracy metrics. These metrics are essential for understanding the model’s ability to correctly classify happy and sad images.

## Deployment with Streamlit
A Streamlit application was created for easy deployment and interactive use of the model. This web app allows users to upload their own images, and the model instantly predicts whether the image is "happy" or "sad."

- The Streamlit app provides a user-friendly interface for uploading images.
- **Real-time Prediction**: The app uses the trained model to output a prediction for each uploaded image.
- **Example Images**: Sample screenshots of the deployed Streamlit app are available in the repository, showcasing how predictions are displayed in the interface.

## Usage
- **Prediction**: The model can predict the class of a new image using the saved `.h5` model file. Simply load the image, resize it to 256x256, and normalize the pixel values before passing it to the model for prediction.
- **Output**: The model outputs a probability between 0 and 1. A threshold of 0.5 is applied to classify the image as either "happy" or "sad."

- ![image](https://github.com/user-attachments/assets/88712679-deb5-4435-ace5-84fab073942e)
- ![image](https://github.com/user-attachments/assets/819d1152-218d-4576-aeb8-8f7a5f2180ac)



## Example Code for Prediction
```python
import cv2
import numpy as np
import tensorflow as tf

# Load model
model = tf.keras.models.load_model('models/happysadmodel.h5')

# Load and preprocess the image
img = cv2.imread('path_to_image.jpg')
resize = tf.image.resize(img, (256, 256))
y_pred = model.predict(np.expand_dims(resize/255, axis=0))

# Display result
if y_pred[0][0] > 0.5:
    print("Prediction: Sad")
else:
    print("Prediction: Happy")
```

## Visualization
Sad ==> 1
Happy ==> 0
![image](https://github.com/user-attachments/assets/ec3d68e6-f260-4a35-b256-4075e16db1c8)
The Learning Curve
![image](https://github.com/user-attachments/assets/22a9b5a9-c0a4-4ed7-b86a-8d16ea2cbb83)


## Model Saving
The trained model is saved as `happysadmodel.h5` and can be loaded for future predictions.
