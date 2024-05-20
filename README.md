# Image-Classification-with-CIFAR-10

## Project Description
This project implements a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify images from the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The classes are:

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

With this model, we obtained a test accuracy of 0.80 on the CIFAR-10 dataset.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Data Processing](#data-processing)
- [Model Architecture](#model-architecture)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Model Saving](#model-saving)
- [Prediction](#prediction)
- [License](#license)

## Installation
To run this project, you need to have Python installed along with the following libraries:
- TensorFlow
- NumPy
- Matplotlib

You can install the required libraries using the following command:
```bash
pip install tensorflow numpy matplotlib
```

## Usage 
1) Clone the repository:
```bash
git clone https://github.com/your_username/Image-Classification-with-CIFAR-10.git
cd Image-Classification-with-CIFAR-10
```

2) Run the Jupyter notebook:
```bash
jupyter notebook
```

## Project Structure
```bash
Image-Classification-with-CIFAR-10/
├── image_classification_cifar10.ipynb  # Jupyter notebook with the code
├── cifar10_model.h5                    # Saved model (after training)
├── README.md                           # Project README file
```

## Data Processing
The CIFAR-10 dataset is loaded using the cifar10.load_data() function from TensorFlow's Keras datasets. The images are normalized, and the labels are one-hot encoded.

```bash
from tensorflow.keras.datasets import cifar10
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
num_classes = 10
y_train = np.eye(num_classes)[y_train.reshape(-1)]
y_test = np.eye(num_classes)[y_test.reshape(-1)]
```

## Model Architecture
The CNN model is defined using TensorFlow's Keras Sequential API. The model consists of the following layers:
Convolutional Layer (32 filters, 3x3 kernel)
Convolutional Layer (32 filters, 3x3 kernel)
Max Pooling Layer (2x2 pool size)
Dropout Layer (0.25)
Convolutional Layer (64 filters, 3x3 kernel)
Convolutional Layer (64 filters, 3x3 kernel)
Max Pooling Layer (2x2 pool size)
Dropout Layer (0.25)
Flatten Layer
Dense Layer (512 units, ReLU activation)
Dropout Layer (0.5)
Dense Layer (10 units, Softmax activation)
```bash
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
```

##Model Training
The model is compiled with the Adam optimizer and categorical cross-entropy loss function. The model is trained for 30 epochs with a batch size of 128.
```bash
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

epochs = 30
batch_size = 128

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          shuffle=True)
```

## Model Evaluation
```bash
scores = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', scores)
print('Test accuracy:', scores[1])
```

## Model Saving
```bash
model.save('cifar10_model.h5')
```

## Prediction
The trained model can be loaded and used to make predictions on new images. The predict() function is used to obtain the predicted class probabilities for a set of images.
```bash
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

your_model = load_model('cifar10_model.h5')
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

indices = np.random.choice(len(x_test), 5)
images = x_test[indices]

predictions = your_model.predict(images)

for i, image in enumerate(images):
    predicted_class = class_names[np.argmax(predictions[i])]
    print(f"Image {i+1}: {predicted_class}")
    plt.imshow(image)
    plt.show()

```
## License
This README file provides an overview of the project, including its description, installation instructions, usage, project structure, data processing, model architecture, training, evaluation, saving, and prediction. It also includes sections for contributing and licensing, which are important for open-source projects.
