from abc import ABC, abstractmethod
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.utils import to_categorical


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# define an Interface
class MnistClassifierInterface(ABC):

    @abstractmethod
    def train(self, x_train, y_train, **kwargs):
        pass

    @abstractmethod
    def predict(self, x_test):
        pass


# define Random Forest model
class MnistRandomForestClassifier(MnistClassifierInterface):
    def __init__(self):
        self.model = None

    def train(self, x_train, y_train, **kwargs):
        x_train_flat = x_train.reshape(x_train.shape[0], -1)

        self.model = RandomForestClassifier(**kwargs)
        self.model.fit(x_train_flat, y_train)

    def predict(self, x_test):
        x_test_flat = x_test.reshape(x_test.shape[0], -1)
        return self.model.predict(x_test_flat)
    

# define Feed-Forward Neural Network
class MnistNeuralNetworkClassifier(MnistClassifierInterface):
    def __init__(self, input_shape=(28, 28), num_classes=10):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = Sequential([
            Dense(512, activation='relu'),
            Dense(256, activation='relu'),
            Dense(self.num_classes, activation='softmax')
        ])

    def train(self, x_train, y_train, epochs=5, batch_size=128, **kwargs):
        x_train_flat = x_train.reshape(x_train.shape[0], -1).astype('float32') / 255.0
        # one-hot encoding
        y_train_cat = to_categorical(y_train, self.num_classes)

        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
        
        self.model.fit(x_train_flat, y_train_cat, 
                       epochs=epochs, batch_size=batch_size, 
                       verbose=2, **kwargs)

    def predict(self, x_test):
        x_test_flat = x_test.reshape(x_test.shape[0], -1).astype('float32') / 255.0
        y_pred = self.model.predict(x_test_flat)
        return np.argmax(y_pred, axis=1)


class MnistCNNClassifier(MnistClassifierInterface):
    def __init__(self, input_shape=(28, 28, 1), num_classes=10):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = Sequential([
            Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(self.input_shape)),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(self.num_classes, activation='softmax')
        ])
    
    def train(self, x_train, y_train, epochs=5, batch_size=128, **kwargs):
        y_train_cat = to_categorical(y_train, self.num_classes)
        
        self.model.compile(optimizer='adam', 
                           loss='categorical_crossentropy', 
                           metrics=['accuracy'])
        
        self.model.fit(x_train, y_train_cat,
                       epochs=epochs, batch_size=batch_size,
                       verbose=2, **kwargs)
        
    def predict(self, x_test):
        pred = self.model.predict(x_test)
        return np.argmax(pred, axis=1)


class MnistClassifier:
    def __init__(self, algorithm='nn'):
        self.algorithm = algorithm.lower()
        if self.algorithm == 'rf':
            self.model = MnistRandomForestClassifier()
        elif self.algorithm == 'nn':
            self.model = MnistNeuralNetworkClassifier()
        elif self.algorithm == 'cnn':
            self.model = MnistCNNClassifier()
        else:
            raise ValueError("Unknown algorithm. Choose 'rf', 'nn' or 'cnn'")
    
    def train(self, x_train, y_train, **kwargs):
        self.model.train(x_train, y_train, **kwargs)

    def predict(self, x_test):
        return self.model.predict(x_test)


