# Rock-Paper-Scissors Hand Gesture Classifier

This project is a machine learning model for classifying hand gestures of Rock, Paper, and Scissors from images. The model is trained on a dataset of hand gesture images and is capable of recognizing these three classes using computer vision techniques.

## Project Overview

The aim of this project is to classify hand gestures from the Rock-Paper-Scissors game. We utilize a Convolutional Neural Network (CNN) model built using TensorFlow and Keras to identify the gesture in an image. The dataset contains 2188 labeled images of 'Rock', 'Paper', and 'Scissors' gestures, captured with consistent lighting on a green background.

## Dataset

The dataset used in this project contains a total of **2188 images** from [Dicoding](https://github.com/dicodingacademy/assets/releases/download/release/rockpaperscissors.zip), divided into three categories:

- **Rock**: 726 images
- **Paper**: 710 images
- **Scissors**: 752 images

All images are in `.png` format with a resolution of **300x200 pixels**. The dataset is split into **60% for training** and **40% for validation**.

## Requirements

To run this project, you will need to install the following dependencies:

- Python 3.x
- TensorFlow
- Keras
- NumPy
- Matplotlib
- OpenCV (for optional image processing)

Install the dependencies using pip:

```bash
pip install tensorflow keras numpy matplotlib opencv-python
```

## Model Architecture

The model is a Convolutional Neural Network (CNN) with the following architecture:

- Conv2D Layer: 64 filters, kernel size 3x3, activation ReLU
- MaxPooling2D Layer
- Conv2D Layer: 128 filters, kernel size 3x3, activation ReLU
- MaxPooling2D Layer
- Conv2D Layer: 256 filters, kernel size 3x3, activation ReLU
- MaxPooling2D Layer
- Flatten Layer
- Dense Layer: 128 units, activation ReLU
- Output Layer: 3 units (softmax for Rock, Paper, Scissors)
- The model uses Adam optimizer and the sparse categorical crossentropy loss function.

## Data Augmentation

To improve the robustness of the model, we apply data augmentation techniques during training, such as:

- Random rotations (up to 40 degrees)
- Width and height shifts (up to 20%)
- Zoom (up to 20%)
- Shear transformations
- Horizontal flipping

## Training

The model is trained for 10 epochs, and the accuracy achieved on the validation set is 82%. Additional steps can be taken to improve this accuracy, such as increasing the number of epochs, fine-tuning the model architecture, or using transfer learning.

## Inference

To make a prediction on a new image, you can use the following code:

```python
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the image you want to predict
img_path = 'path_to_image.png'
img = image.load_img(img_path, target_size=(200, 300))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
img_array = img_array / 255.0  # Rescale the image

# Make prediction
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions[0])

# Output the prediction
class_names = ['Rock', 'Paper', 'Scissors']
print(f"Predicted class: {class_names[predicted_class]}")
```

## Results

After training the model for 10 epochs, the following results were obtained:

- Final Training Accuracy: ~82%
- Final Validation Accuracy: ~82%
- To improve accuracy, you can experiment with more complex architectures, additional data augmentation techniques, or use transfer learning with pre-trained models.

## Future Improvements

- Increase the dataset size or diversify it with more backgrounds and lighting conditions.
- Apply advanced data augmentation techniques or preprocessing.
- Use a pre-trained model (e.g., MobileNetV2 or ResNet) for transfer learning to improve classification performance.
- Implement a real-time camera-based classification using OpenCV.

## Thankyou
