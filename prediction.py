import numpy as np
# import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import imutils


class DR:
    def __init__(self):
        # loading model
        model_architecture = 'CNN_Model.json'
        model_weights = 'model.h5'
        self.graph = tf.get_default_graph()
        self.model = model_from_json(open(model_architecture).read())
        self.model.load_weights(model_weights)

        # self.model.summary() # uncomment if you want to see the model structure

        # setting image params
        self.height = 224
        self.width = 224
        self.channels = 3
        self.level = None

    def prediction(self,img):
        self.img = img
        self.image = cv2.imread(self.img)
        self.image = cv2.resize(self.image, (self.height, self.width))
        self.data = np.expand_dims(self.image, axis=0)

        with self.graph.as_default():
            self.results = self.model.predict(self.data)[0]

        print('D-Retinopathy : {}%'.format(round(self.results[1]*100,2)))
        print('Normal : {}% '.format(round(self.results[0]*100,2)))

        self.proba = round(self.results[1]*100,2)
        if 50 < self.proba <= 65:
            self.level = "Mild"
        elif 65 < self.proba <= 80:
            self.level = "Moderate"
        elif 80 < self.proba <= 90:
            self.level = "Severe"
        elif self.proba > 90:
            self.level = "Proliferative"

        if self.proba > 50:
            self.out = "{}: {:.2f}%".format('D-Retinopathy', self.proba)
            self.color = (0,255,0)
        else:
            self.out = "{}".format("Normal")
            self.color = (255,0,0)
        self._display()
        return self.out, self.level

    def _display(self):
        output = imutils.resize(self.image, width=400)
        cv2.putText(output, self.out, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.color, 2)
        cv2.putText(output, "Level: {}".format(self.level), (10, 65),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imwrite("./templates/output.jpeg", output)
        # plt.figure(num='Result')
        # plt.title("Predicted results")
        # plt.axis('off')
        # plt.imshow(output)
        # plt.show()




