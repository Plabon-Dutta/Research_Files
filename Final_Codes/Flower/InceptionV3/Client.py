import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Import InceptionV3 and preprocessing function
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input

# Define constants
BATCH_SIZE = 500
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
NUM_CLIENTS = 2  # Min must be 2
IMG_SIZE = (75, 75)  # Inception V3 input shape
NO_OF_CLASSES = 47
STEP_PER_SIZE = 20

# Loading Custom Data
data_dir = 'F:/Plabon/Dataset2'

# Use the image_dataset_from_directory method to load and preprocess the data
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='training',
    seed=42,
    image_size=IMG_SIZE,
    batch_size=None
)

test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='validation',
    seed=42,
    image_size=IMG_SIZE,
    batch_size=None
)

# Get the class names
class_names = train_dataset.class_names
print(class_names)
print("No of Classes: ", end="")
print(len(class_names))

# Extract all the images and labels from the dataset and split the datasets into X and y sets
x_train = []
y_train = []
for batch in train_dataset:
    batch_images, batch_labels = batch
    x_train.append(batch_images.numpy())
    y_train.append(batch_labels.numpy())

x_train = np.concatenate([np.expand_dims(i, axis=0) for i in x_train])
y_train = np.concatenate([np.expand_dims(i, axis=0) for i in y_train])

x_test = []
y_test = []
for batch in test_dataset:
    batch_images, batch_labels = batch
    x_test.append(batch_images.numpy())
    y_test.append(batch_labels.numpy())

x_test = np.concatenate([np.expand_dims(i, axis=0) for i in x_test])
y_test = np.concatenate([np.expand_dims(i, axis=0) for i in y_test])

# Preprocessing and InceptionV3 input shape
x_train = preprocess_input(x_train)
x_test = preprocess_input(x_test)
# input_shape = (299, 299, 3)  # Inception V3 input shape

# One-hot encode labels
y_train = to_categorical(y_train, num_classes=len(class_names))
y_test = to_categorical(y_test, num_classes=len(class_names))


# # Define InceptionV3-based model
# def create_inceptionv3_model():
#     base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
#     x = base_model.output
#     x = Flatten()(x)
#     x = Dense(128, activation='relu')(x)
#     x = Dropout(0.5)(x)
#     predictions = Dense(len(class_names), activation='softmax')(x)
#     model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
#     adam = Adam(learning_rate=LEARNING_RATE)
#     model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
#     return model

input_shape = (75, 75, 3)

def create_inceptionv3_model():
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(NO_OF_CLASSES, activation='softmax')(x)  # Update the number of classes
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Optional: Freeze the layers of the base model
    for layer in base_model.layers:
        layer.trainable = False

    adam = Adam(learning_rate=LEARNING_RATE)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model

global_model = create_inceptionv3_model()
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

plot_model(global_model, to_file='cnn_model.png', show_shapes=True, show_layer_activations=True, show_dtype=False, show_layer_names=True)

import flwr as fl
import tensorflow as tf

class Client(fl.client.NumPyClient):
    def get_parameters(self, config):
        return global_model.get_weights()

    def fit(self, parameters, config):
        global_model.set_weights(parameters)
        global_model.fit(x_train, y_train, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, steps_per_epoch=STEP_PER_SIZE)
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

plt.figure(figsize=(16,4))
plt.suptitle("FedCNN Performance - TestSet")
plt.subplot(121)
plt.xlabel("epochs")
plt.ylabel("loss")
plt.plot(list(range(0,len(global_loss_list))), global_loss_list)
plt.title("Global Loss")

plt.subplot(122)
plt.xlabel("epochs")
plt.ylabel("acc")
plt.plot(list(range(0,len(global_acc_list))), global_acc_list)
plt.title("Global Acc")

print('Total EPOCHs : ', len(global_acc_list))
print('Total Clients : ', NUM_CLIENTS)