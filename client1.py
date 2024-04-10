from typing import Dict, Tuple
from flwr.common import NDArrays
import tensorflow as tf
from tensorflow import keras
import flwr as fl
import numpy as np
import pickle
from io import BytesIO
from PIL import Image
import os
import requests

model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10)  # Assuming 10 output classes for classification
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

train_benign_path = 'ISIC_resized/train/benign'
train_malignant_path = 'ISIC_resized/train/malignant'

test_benign_path = 'ISIC_resized/test/benign'
test_malignant_path = 'ISIC_resized/test/malignant'

SIZE=32

def import_images(benign_path, malignant_path):
    X = []
    Y = []
    # Iterate through all files in the folder
    for filename in os.listdir(benign_path):
        # Construct the full file path
        filepath = os.path.join(benign_path, filename)
        img = Image.open(filepath)
        img = img.resize((SIZE, SIZE))
        X.append(np.asarray(img))
        Y.append(0)
            
    for filename in os.listdir(malignant_path):
        # Construct the full file path
        filepath = os.path.join(malignant_path, filename)
        img = Image.open(filepath)
        img = img.resize((SIZE, SIZE))
        X.append(np.asarray(img))
        Y.append(1)
    X = np.asarray(X)
    Y = np.asarray(Y)
    return X, Y

x_train, y_train = import_images(train_benign_path, train_malignant_path)
print(x_train.shape)
print(y_train.shape)

x_test, y_test = import_images(test_benign_path, test_malignant_path)
print(x_test.shape)
print(y_test.shape)

x_train = x_train/255.
x_test = x_test/255.

#FL ML model
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()
    
    def fit(self, parameters, config):
        #update local model with global parameters
        model.set_weights(parameters)
        #train local model with local data
        model.fit(x_train, y_train, epochs=1, batch_size=32)
        return model.get_weights(), len(x_train), {}
    
    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        return loss, len(x_test), {"accuracy": accuracy}


# Send a GET request to backend to start FL_server
# response = requests.get('http://127.0.0.1:5000/start_server')

# if response.status_code == 200:
#     data = response.json()
#     print(data)
# else:
#     print(f'Request failed with status code: {response.status_code}')


#start Flower client
fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=FlowerClient()
)
