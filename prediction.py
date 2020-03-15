import numpy as np
import matplotlib.pyplot as plt
import cv2
import keras
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import imutils

# loading model
model_architecture = 'CNN_Model.json'
model_weights = 'model.h5'
global graph, model
graph = tf.get_default_graph()
model = model_from_json(open(model_architecture).read())
model.load_weights(model_weights)

# model.summary() # uncomment if you want to see the model structure

# setting image params
height = 224
width = 224
channels = 3

def prediction(img):
    image = cv2.imread(img)
    image = cv2.resize(image, (height, width))
    data = np.expand_dims(image, axis=0)
    with graph.as_default():
        results = model.predict(data)
    print(results)
    print('D-Retinopathy : {}%'.format(round(results[1]*100,2)))
    print('Normal : {}% '.format(round(results[0]*100,2)))

    proba = round(results[1]*100,2)
    output = imutils.resize(image, width=400)
    if 50 < proba <= 65:
        level = "Mild"
    elif 65 < proba <= 80:
        level = "Moderate"
    elif 80 < proba <= 90:
        level = "Severe"
    elif proba > 90:
        level = "Proliferative"
    else:
        level = None

    if proba > 50:
        out = "{}: {:.2f}%".format('D-Retinopathy', proba)
        color = (0,255,0)
    else:
        out = "{}".format("Normal")
        color = (255,0,0)
    text = cv2.putText(output, out, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(output, "Level: {}".format(level), (10, 65),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    plt.figure(num='Result')
    plt.title("Predicted results")
    plt.axis('off')
    plt.imshow(output)
    plt.show()
    return out, level

# prediction("/home/sharma/Downloads/mild.jpg")
