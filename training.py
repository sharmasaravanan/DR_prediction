# Importing libraries

import pandas as pd
import numpy as np
import os
import cv2


import tensorflow
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy

from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from IPython.display import display
import matplotlib.pyplot as plt

from models.CNN_model import model_DR
from models.utility import create_filename, binary_target, plot_confusion_matrix

import warnings
warnings.filterwarnings('ignore')

# setting image params
height = 224
width = 224
channels = 3

# loading CSV data
df_train = pd.read_csv('./input/train.csv')
print("Shape of the dataset : ",df_train.shape)


# Create a new column called file_name
df_train['file_name'] = df_train['id_code'].apply(create_filename)
df_train.head()

print()
print("Displaying top-5 rows of data")
display(df_train.head())

# Check the target distribution
display(df_train['diagnosis'].value_counts())

# Create Binary Targets
df_train['binary_target'] = df_train['diagnosis'].apply(binary_target)
print()
display(df_train.head())

# Balance the target distribution
df_0 = df_train[df_train['binary_target'] == 0]
df_1 = df_train[df_train['binary_target'] == 1].sample(len(df_0), random_state=101)


df_data = pd.concat([df_0, df_1], axis=0).reset_index(drop=True)
df_data = shuffle(df_data)

print()
print("Shape of the dataset after balancing : ",df_data.shape)

print()
print("Displaying top-5 rows of dataset after balancing")
display(df_data.head())

# Train Test Split
df_train, df_val = train_test_split(df_data, test_size=0.1, random_state=101)

print()
print("Shape of the training data :", df_train.shape)
print("Shape of the validation data :", df_val.shape)

# Create the directory structure
base_dir = 'dataDir'
if not os.path.isdir(base_dir):
    print("Creating base_dir")
    os.mkdir(base_dir)


# train_dir
train_dir = os.path.join(base_dir, 'train_dir')
if not os.path.isdir(train_dir):
    print("Creating train_dir")
    os.mkdir(train_dir)

a_0 = os.path.join(train_dir, 'a_0')
if not os.path.isdir(a_0):
    print("Creating a_0")
    os.mkdir(a_0)
b_1 = os.path.join(train_dir, 'b_1')
if not os.path.isdir(b_1):
    print("Creating b_1")
    os.mkdir(b_1)

# val_dir
val_dir = os.path.join(base_dir, 'val_dir')
if not os.path.isdir(val_dir):
    print("Creating val_dir")
    os.mkdir(val_dir)
a_0 = os.path.join(val_dir, 'a_0')
if not os.path.isdir(a_0):
    print("Creating a_0")
    os.mkdir(a_0)
b_1 = os.path.join(val_dir, 'b_1')
if not os.path.isdir(b_1):
    print("Creating b_1")
    os.mkdir(b_1)

# Transfer the Images into the Folders
df_data.set_index('file_name', inplace=True)

train_list = list(df_train['file_name'])
for fname in train_list:
    label = df_data.loc[fname, 'binary_target']

    if label == 0:
        sub_folder = 'a_0'
        src = os.path.join('./input/train_images', fname)
        dst = os.path.join(train_dir, sub_folder, fname)

        image = cv2.imread(src)
        image = cv2.resize(image, (height, width))
        cv2.imwrite(dst, image)

    if label == 1:
        sub_folder = 'b_1'
        src = os.path.join('./input/train_images', fname)
        dst = os.path.join(train_dir, sub_folder, fname)

        image = cv2.imread(src)
        image = cv2.resize(image, (height, width))
        cv2.imwrite(dst, image)

print()
print("Total number of 0 label training images :", len(os.listdir('dataDir/train_dir/a_0')))
print("Total number of 1 label training images :", len(os.listdir('dataDir/train_dir/b_1')))

val_list = list(df_val['file_name'])
for fname in val_list:
    label = df_data.loc[fname, 'binary_target']

    if label == 0:
        sub_folder = 'a_0'
        src = os.path.join('./input/train_images', fname)
        dst = os.path.join(val_dir, sub_folder, fname)

        image = cv2.imread(src)
        image = cv2.resize(image, (height, width))
        cv2.imwrite(dst, image)

    if label == 1:
        sub_folder = 'b_1'
        src = os.path.join('./input/train_images', fname)
        dst = os.path.join(val_dir, sub_folder, fname)

        image = cv2.imread(src)
        image = cv2.resize(image, (height, width))
        cv2.imwrite(dst, image)

print()
print("Total number of 0 label validation images :", len(os.listdir('dataDir/val_dir/a_0')))
print("Total number of 1 label validation images :", len(os.listdir('dataDir/val_dir/b_1')))
print()

# Set Up the Generators
train_path = 'dataDir/train_dir'
val_path = 'dataDir/val_dir'

num_train_samples = len(df_train)
num_val_samples = len(df_val)
train_batch_size = 5
val_batch_size = 5

train_steps = np.ceil(num_train_samples / train_batch_size)
val_steps = np.ceil(num_val_samples / val_batch_size)

datagen = ImageDataGenerator(preprocessing_function= tensorflow.keras.applications.mobilenet.preprocess_input)

train_gen = datagen.flow_from_directory(train_path, target_size=(height,width), batch_size=train_batch_size)

val_gen = datagen.flow_from_directory(val_path, target_size=(height,width), batch_size=val_batch_size)

test_gen = datagen.flow_from_directory(val_path, target_size=(height,width), batch_size=1, shuffle=False)

# defining model
model = model_DR()
model.summary()

print()
print("Total number of layers in Model :", len(model.layers))

x = model.layers[-6].output
x = Dropout(0.25)(x)
predictions = Dense(2, activation='softmax')(x)
models = Model(inputs=model.input, outputs=predictions)

models.summary()

for layer in models.layers[:-23]:
    layer.trainable = False

class_weights={
    0: 1.0, # Class 0
    1: 1.0, # Class 1
}

# Train the Model
models.compile(Adam(lr=0.01), loss='categorical_crossentropy', metrics=[categorical_accuracy])

checkpoint = ModelCheckpoint("model.h5", monitor='val_categorical_accuracy', verbose=1, save_best_only=True, mode='max')

reduce_lr = ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.5, patience=2, verbose=1, mode='max',
                              min_lr=0.00001)

early_stopper = EarlyStopping(monitor="val_categorical_accuracy", mode="max", patience=7)

csv_logger = CSVLogger(filename='training_log.csv', separator=',', append=False)

callbacks_list = [checkpoint, reduce_lr, early_stopper, csv_logger]

history = models.fit_generator(train_gen, steps_per_epoch=train_steps, class_weight=class_weights,
                              validation_data=val_gen, validation_steps=val_steps, epochs=100, verbose=1,
                              callbacks=callbacks_list)

# Get the best epoch from the training log
df = pd.read_csv('training_log.csv')

best_acc = df['val_categorical_accuracy'].max()

# display the row with the best accuracy
print()
display(df[df['val_categorical_accuracy'] == best_acc])

# Evaluate the model
models.load_weights('model.h5')
_, accuracy = models.evaluate_generator(test_gen, steps=len(df_val))
print('Accuracy :', accuracy)

# Plot the Training Curves
acc = history.history['categorical_accuracy']
val_acc = history.history['val_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'g', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig("loss.png")
plt.figure()

plt.plot(epochs, acc, 'g', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.xlabel("Epochs")
plt.ylabel("Accuracy %")
plt.title('Training and validation cat accuracy')
plt.legend()
plt.figure()
plt.savefig("accuracy.png")
plt.show()

# Confusion Matrix
predictions = models.predict_generator(test_gen, steps=len(df_val), verbose=1)
cm = confusion_matrix(test_gen.classes, predictions.argmax(axis=1))
plot_confusion_matrix(cm, ['0', '1'])

# Classification Report
y_pred = np.argmax(predictions, axis=1)
y_true = test_gen.classes

report = classification_report(y_true, y_pred, target_names=['0', '1'])
print(report)

#save model
model_json = models.to_json()
open('CNN_Model.json', 'w').write(model_json)