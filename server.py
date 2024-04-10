from flask import Flask, request, jsonify
from flask_cors import CORS
import flwr as fl
import tensorflow as tf
from tensorflow import keras
import numpy as np
from io import BytesIO
from PIL import Image
from threading import Thread
import json

app = Flask(__name__)
CORS(app)

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
   
@app.route('/')
def home():
    return 'Hello, This is home page. Use /predict or /get_weights or /start_server'

@app.route('/predict', methods=['POST'])
def predict():
    # Check if the request contains an image
    if 'image' not in request.files:
        return 'No image found', 400
    
    # Get the image file from the request
    image_file = request.files['image']
    
    # Read the image data
    image_data = image_file.read()
    
    # Create an image object from the binary data
    img = Image.open(BytesIO(image_data))
    img = img.resize((224,224))
    # img.show()
    arr = np.array(img)
    arr = np.expand_dims(arr, axis=0) 
    # prediction = model.predict(arr)
    # print(prediction[0][0])
    # if prediction == 1:
    #     return "CANCER DETECTED"
    return "NO CANCER"

weights = None
class AggregateWeightsCallBack(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        aggregated_weights = super().aggregate_fit(rnd, results, failures)

        global weights
        weights = fl.common.parameters_to_ndarrays(aggregated_weights[0])

        # Update the server model's weights with the aggregated weights
        model.set_weights(weights)
        return aggregated_weights

def start_server():
    fl.server.start_server(
    server_address="skin-cancer-backend.onrender.com:8080",
    config=fl.server.ServerConfig(num_rounds=5),
    strategy=AggregateWeightsCallBack()
    )
    print("FL training done")

# fl_server_running = False

@app.route('/start_server', methods=['GET'])
def train():
    # if(fl_server_running == True):
        # return jsonify({"message": "Server already running"})
    # fl_server_running = True
    start_server()
    # fl_server_running = False
    print(type(weights))
    return jsonify({"message": "Federated learning server started on port 8080"})

@app.route('/get_weights', methods=['GET'])
def get_weights():
    # json_weights = json.dumps(weights)
    return jsonify({"message": "Weights"})
    
if __name__ == '__main__':
    app.run(debug=True)
