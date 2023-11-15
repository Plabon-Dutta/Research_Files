import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG19
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Precision, Recall, Accuracy

import numpy as np

# Define constants
BATCH_SIZE = 500
NUM_EPOCHS = 30
LEARNING_RATE = 0.001
NUM_CLASSES = 47
IMG_SIZE = (128, 128)
STEP_PER_EPOCH = 15
DEBUG = False
NUM_CLIENTS = 2

# Loading Custom Data
# Define the path to the directory containing the image data
data_dir = 'F:/Plabon/Dataset2'
# Test_data_dir = 'F:/Plabon/VSCode/Models/MNISTFLWR/data/test'

# Use the image_dataset_from_directory method to load and preprocess the data
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2, # split 20% of data for validation
    subset='training',
    seed=42, # set a random seed for reproducibility
    image_size= IMG_SIZE,
    batch_size= None
)

test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='validation',
    seed=42,
    image_size= IMG_SIZE,
    batch_size= None
)

# Get the class names
class_names = train_dataset.class_names
print(class_names)

# # Configure the dataset for performance
# train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
# val_dataset = test_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# Extract all the images and labels from the dataset( and also Split the datasets into X and y sets)
x_train = []
y_train = []
for batch in train_dataset:
    batch_images, batch_labels = batch
    x_train.append(batch_images.numpy())
    y_train.append(batch_labels.numpy())

x_train = np.concatenate([np.expand_dims(i,axis=0) for i in x_train])
y_train = np.concatenate([np.expand_dims(i,axis=0) for i in y_train])

x_test = []
y_test = []
for batch in test_dataset:
    batch_images, batch_labels = batch
    x_test.append(batch_images.numpy())
    y_test.append(batch_labels.numpy())

x_test = np.concatenate([np.expand_dims(i,axis=0) for i in x_test])
y_test = np.concatenate([np.expand_dims(i,axis=0) for i in y_test])


# Preprocessing
x_train /= 255
x_test /= 255
y_train = to_categorical(y_train, len(class_names))
y_test = to_categorical(y_test, len(class_names))

x_train.shape
x_test.shape

vgg19 = VGG19(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

# Create a new model for fine-tuning
model = models.Sequential()

# Add the VGG16 base model
model.add(vgg19)

# Add custom top layers for your classification task
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(len(class_names), activation='softmax'))

# Set VGG16 layers to non-trainable (optional)
for layer in vgg19.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# Train the model
# history = model.fit(train_generator, epochs=NUM_EPOCHS, validation_data=validation_generator)

# Save the model for later use
# model.save('sign_language_vgg16.h5')


global_model = model
global_model.summary()
global_acc_list = []
global_loss_list = []

# Sequential model in Keras -> ASCII
graphics = {
    "Convolution2D": " \|/ ",
    "Activation": "|||||",
    "Flatten": "|||||",
    "MaxPooling2D": "YYYYY",
    "Dropout": " | ||",
    "Dense": "XXXXX",
    "ZeroPadding2D": "\|||/"
}

def jsonize(model):
    res = []
    for layer in model.layers:
        x = {}
        
        x["name"] = layer.name
        x["kind"] = layer.__class__.__name__
        x["input_shape"] = layer.input_shape[1:]
        x["output_shape"] = layer.output_shape[1:]
        x["n_parameters"] =  layer.count_params()
        try:
            x["activation"] = layer.activation.__name__ 
        except AttributeError:
            x["activation"] = ""
        
        res.append(x)
    return res

def compress_layers(jsonized_layers):
    res = [jsonized_layers[0]]
    for each in jsonized_layers[1:]:
        if each["kind"] == "Activation" and res[-1]["activation"] in ["", "linear"]:
            res[-1]["activation"] = each["activation"]
        else:
            res.append(each)
    return res

# data_template = "{activation:>15s}   #####   {shape} = {length}"
data_template = "{activation:>15s}   #####   {shape}"
layer_template = "{kind:>15s}   {graphics} -------------------{n_parameters:10d}   {percent_parameters:5.1f}%"

def product(iterable):
    res = 1
    for each in iterable:
        res *= each
    return res

def print_layers(jsonized_layers, sparser=False, simplify=False, header=True):
    
    if simplify:
        jsonized_layers = compress_layers(jsonized_layers)
    
    all_weights = sum([each["n_parameters"] for each in jsonized_layers])
    
    if header:
        print("      OPERATION           DATA DIMENSIONS   WEIGHTS(N)   WEIGHTS(%)\n")
    
    print(data_template.format(
            activation="Input",
            shape=jsonized_layers[0]["input_shape"],
            # length=product(jsonized_layers[0]["output_shape"])
    ))
    
    for each in jsonized_layers:
        
        if sparser:
            print("")
            
        print(layer_template.format(
                kind=each["kind"] if each["kind"] != "Activation" else "",
                graphics=graphics.get(each["kind"], "?????"),
                n_parameters=each["n_parameters"],
                percent_parameters=100 * each["n_parameters"] / all_weights
        ))
        
        if sparser:
            print("")
            
        print(data_template.format(
                activation=each["activation"] if each["activation"] != "linear" else "",
                shape=each["output_shape"],
                # length=product(each["output_shape"])
        ))
        
def sequential_model_to_ascii_printout(model, sparser=False, simplify=True, header=True):
    print_layers(jsonize(model), sparser=sparser, simplify=simplify, header=header)


# from keras_ascii_sequential import sequential_model_to_ascii_printout
sequential_model_to_ascii_printout(global_model)

# Plot model in Keras
from tensorflow.keras.utils import plot_model

plot_model(global_model, to_file='VGG_model.png', show_shapes=True, show_layer_activations=True, show_dtype=False, show_layer_names=True)

import flwr as fl
import tensorflow as tf

class Client(fl.client.NumPyClient):
    def get_parameters(self, config):
        return global_model.get_weights()

    def fit(self, parameters, config):
        global_model.set_weights(parameters)
        global_model.fit(x_train, y_train, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, steps_per_epoch=STEP_PER_EPOCH)
        return global_model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        global_model.set_weights(parameters)
        loss, accuracy = global_model.evaluate(x_test, y_test)
        global_acc_list.append(accuracy)
        global_loss_list.append(loss)
        return loss, len(x_test), {"accuracy": float(accuracy)}

fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=Client())

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Train set
y_train_pred = global_model.predict(x_train)
y_train_pred = np.argmax(y_train_pred, axis=1)
y_train_true = np.argmax(y_train, axis=1)

# Test set
y_pred = global_model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Confusion Metrix
cm = confusion_matrix(y_true, y_pred)
cm_train = confusion_matrix(y_train_true, y_train_pred)

import re
from sklearn.metrics import classification_report

report = classification_report(y_true, y_pred, digits=4)
print('test metrics: ')
print(report)

accuracy = float(re.findall(r'accuracy\s+([\d.]+)', report)[0])
precision = float(re.findall(r'weighted avg\s+([\d.]+)', report)[0])
recall = float(re.findall(r'weighted avg\s+([\d.]+)\s+([\d.]+)', report)[0][1])
f1_score = float(re.findall(r'weighted avg\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', report)[0][2])

print(f"Accuracy: {accuracy:.2%}")
print(f"Precision: {precision:.2%}")
print(f"Recall: {recall:.2%}")
print(f"F1-score: {f1_score:.2%}")

report = classification_report(y_train_true, y_train_pred, digits=4)
print('-----------------------')
print('train metrics: ')
print(report)

accuracy_train = float(re.findall(r'accuracy\s+([\d.]+)', report)[0])
precision_train = float(re.findall(r'weighted avg\s+([\d.]+)', report)[0])
recall_train = float(re.findall(r'weighted avg\s+([\d.]+)\s+([\d.]+)', report)[0][1])
f1_score_train = float(re.findall(r'weighted avg\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', report)[0][2])

print(f"Accuracy: {accuracy_train:.2%}")
print(f"Precision: {precision_train:.2%}")
print(f"Recall: {recall_train:.2%}")
print(f"F1-score: {f1_score_train:.2%}")

print('For Graph Plotting\n')
print('Total EPOCHs : ', len(global_acc_list))
print('Total Clients : ', NUM_CLIENTS)
print('Global Loss list : ',global_loss_list)
print('Global Acc list : ',global_acc_list)