# Keras-MNIST
MNIST is a famous dataset of handwritten digits. In this dataset there are 60,000 images of digits for training and 10,000 for test. Also we have used adam optimizer and cross-entropy function for measuring loss.

We have used convolution layers with ReLU activation, dropout, and finally linear layer with softmax. With this configuration we are able to achieve 97.17% accuracy. 

1. Import libraries
```Python
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
```

2. Pre-process the data
```Python
num_classes = 10
input_shape = (28, 28, 1)
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)

print('X_train shape', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
```
3. Define your model
```Python
model = keras.Sequential(
        [
            keras.Input(shape = input_shape),
            layers.Conv2D(32, kernel_size=(3,3), activation='relu'),
            layers.MaxPooling2D(pool_size=(2,2)),
            layers.Conv2D(64, kernel_size=(3,3), activation='relu'),
            layers.MaxPooling2D(pool_size=(2,2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax'),
        ]
)
model.summary()
```

4. Train the model
```Python
batch_size = 128
epochs = 15

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)
```

5. Evalute the model
```Python
score = model.evaluate(X_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
```
