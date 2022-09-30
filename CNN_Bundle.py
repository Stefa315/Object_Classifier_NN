from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
from Save_And_Load_Models import Load_Latest_Model,Save_New_Model
import config
import tensorflow as tf
from Test_Set_Display import Test_Set_Display
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models , layers
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import model_from_json
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
# from Get_Dense_Output import Get_Dense_Output
import os
import get_dataset
ghdslb' mSML Bmnimport cv2
import multiprocessing
import pandas as pd
import time


INIT_LR = 1e-5
EPOCHS = 150
BATCH_SIZE = 16
IMAGE_SIZE = 256
Display_Names = config.Display_Names
n_classes = 8
start_time = time.time()
def Run_Different_Displays_CNN():
    start_time = time.time()

    for names_CNN in Display_Names:
        p = multiprocessing.Process(target=trainNN(names_CNN))
        p.start()
        p.join()
        end_time = time.time()
        time_Training = end_time - start_time
        print("Total Time for a TRAINING OF TRANMISION",time_Training)
        os.system("pause")

        # trainNN(names_CNN)


def trainNN(names_CNN):
    print("[INFO] loading images...")
    print("Training Display : {}".format(names_CNN))
    dataset_2 = tf.keras.preprocessing.image_dataset_from_directory(
        "C:/Users/Takas/PycharmProjects/Classifier/Dataset/Dataset_{}".format(names_CNN),
        labels='inferred',
        label_mode='int',
        batch_size=BATCH_SIZE,
        shuffle=True,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
    )
    class_names = dataset_2.class_names

    train_ds, val_ds, test_ds = get_dataset.get_dataset_partitions(dataset_2)
    # print("TRAIN LENGTH",len(train_ds) , len(val_ds),len(test_ds))

    resize_and_rescale_layer = tf.keras.Sequential([
        layers.experimental.preprocessing.Resizing(IMAGE_SIZE, IMAGE_SIZE),
        layers.experimental.preprocessing.Rescaling(1.0 / 255)

    ])

    data_augmentation_layer = tf.keras.Sequential([
        layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        layers.experimental.preprocessing.RandomRotation(0.2),
    ])

    tf.random.set_seed(0)

    input_shape_model = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3)
    regularizer = tf.keras.regularizers.L1L2(l1=0.0001, l2=0.0001)
    # # #
    model = models.Sequential([
        keras.Input(shape=(256, 256, 3)),
        resize_and_rescale_layer,
        data_augmentation_layer,
        # layers.Conv2D(128, (3,3),activation='relu',input_shape=(BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,3)),
        layers.Conv2D(128, (3, 3), kernel_regularizer=regularizer, input_shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3),
                      padding="same"),
        # layers.Conv2D(128, (3, 3), input_shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3),padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), kernel_regularizer=regularizer, padding="same"),
        # layers.Conv2D(128, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(256, (3, 3), kernel_regularizer=regularizer, padding="same"),
        # layers.Conv2D(256, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(256, (3, 3), kernel_regularizer=regularizer, padding="same"),
        # layers.Conv2D(256, (3, 3),  padding="same"),

        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),

        layers.Dense(512, activation='relu', name='dense_map'),
        layers.Dropout(0.3),
        layers.Dense(n_classes, activation='softmax'),

    ])

    print("TRAIN SHAPES", np.array(train_ds).shape, np.array(val_ds).shape)
    # model.build(input_shape=input_shape_model)
    #
    print("[INFO] compiling model...")
    opt = keras.optimizers.Adam(learning_rate=INIT_LR)
    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )

    # train the head of the network
    print("[INFO] training head...")
    history = model.fit(
        (train_ds),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        validation_data=val_ds
    )
    model_version = len(os.listdir("../Models_Saved/{}".format(names_CNN))) + 1

    model.summary()
    #Save_New_Model(model, names_CNN)
    loaded_model = Load_Latest_Model(None, names_CNN)
    print("Loaded model from disk")
    #hist_df = pd.DataFrame(history.history)
   #OR  save to csv:
    #hist_csv_file = '../Models_Saved/{}/model_{}/history.csv'.format(names_CNN,model_version)
    #with open(hist_csv_file, mode='w') as f:
    # hist_df.to_csv(f)
    #
    #
    #N = EPOCHS
    #plt.style.use("ggplot")
    #plt.figure()
    #plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
    #plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")
    #plt.plot(np.arange(0, N), history.history["accuracy"], label="train_acc")
    #plt.plot(np.arange(0, N), history.history["val_accuracy"], label="val_acc")
    #plt.title("Training Loss and Accuracy")
    #plt.xlabel("Epoch #")
    #plt.ylabel("Loss/Accuracy")
    #plt.legend(loc="lower left")
    #plt.savefig("../Models_Saved/{}/model_{}/plot.png".format(names_CNN, model_version))
    # plt.show()
    #
    # Test_Set_Display(test_ds, loaded_model)


if __name__ == '__main__':
    Run_Different_Displays_CNN()

