import os
import cv2
import numpy as np
from tensorflow import keras

class CustomDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, dir, batch_size=16, n_classes=5):
        'Initialization'
        self.dir = dir
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.on_epoch_end()
        np.random.shuffle(self.indexes)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        if index >= self.__len__():
          raise IndexError("Index out of range")
        'Generate one batch of data'
        # Generate indexes of the batch
        start = index*self.batch_size
        end = min((index+1)*self.batch_size, len(self.list_IDs))  # handle the case when the batch doesn't have batch_size samples
        indexes = self.indexes[start:end]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.list_IDs = []
        self.labels = {}
        for i in range(self.n_classes):
            path = os.path.join(self.dir, str(i))
            for file in os.listdir(path):
                if file.endswith(".jpg"):
                    self.list_IDs.append(os.path.join(path, file))
                    self.labels[os.path.join(path, file)] = i
        self.indexes = np.arange(len(self.list_IDs))
        np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # Initialization
        X = []
        y = np.empty((len(list_IDs_temp)), dtype=int)  # ensure y has shape (batch_size)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Load image
            img = cv2.imread(ID)
            X.append(img)

            # Store class
            y[i] = self.labels[ID]  # labels are integers

        # Pad images to max height and width
        max_height = max(img.shape[0] for img in X)
        max_width = max(img.shape[1] for img in X)
        X = [np.pad(img, ((0, max_height - img.shape[0]), (0, max_width - img.shape[1]), (0, 0)), 'constant') for img in X]
        # Convert to numpy array
        X = np.array(X)

        # Normalize images
        X = X / 255.0

        # X = [data_augmentation(img) for img in X]

        return X, y  # y is an array of integer labels