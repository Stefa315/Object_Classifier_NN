import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import config
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models , layers
from tensorflow.keras.layers import Dropout
from keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD,RMSprop
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.utils import image_dataset_from_directory
import pandas as pd
from keras.models import model_from_json
from Test_Set_Display import Test_Set_Display
from Save_And_Load_Models import Load_Latest_Model
from Save_And_Load_Models import Save_New_Model_Multi
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import csv
import get_dataset
import time

INIT_LR = 1e-3
BATCH_SIZE = 16
IMAGE_SIZE  = 256
EPOCHS = 150

####Classifier with Probabilities Creates a csv file with all the probabilities that our CNNs gave as output as 2-D Array and each row's last element is his label
####We have one hot encoding with alphabetical order (Blood , Fiber , Glass , Hair , Skin , Sand and the length of the array depends on the numbers of microscope modalities we like to use
####This module also creates the model of the Fully-Connected Network which works as an decoder
def Load_Predictions(Modes_Needed):

    for names in Modes_Needed:
        globals()["dataset_" + names] = tf.keras.utils.image_dataset_from_directory(
            "../Dataset/Dataset_{}".format(names),
            labels='inferred',
            label_mode='int',
            batch_size=BATCH_SIZE,
            shuffle=False,
            image_size=(IMAGE_SIZE, IMAGE_SIZE),
        )

        globals()["train_"+names], globals()["val_trans_"+names],globals()["test_"+names] = get_dataset.get_dataset_partitions(globals()["dataset_"+names])

        globals()["model_" + names] = Load_Latest_Model(None,names)

        globals()["ds_prediction_" + names] = (globals()["model_"+names]).predict(globals()["dataset_"+names])
        yield names, globals()["ds_prediction_" + names], dataset_Transmission_Color


def Header_Creation(Modes_Needed):
    header = []
    for modes in Modes_Needed:
        for i in range(1,len(config.img_labels_str)+1):
            header.append(modes+"_"+str(i))
    header.append('Label')
    print(header)
    return header


def Create_CSV(Modes_Needed,prediction_list,dataset_Transmission_Color,path_for_files):
    header = Header_Creation(Modes_Needed)
    num_classes = len(config.img_labels_str)
    with open(path_for_files+'/train.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        f.close()

    counter=0
    csv_line__list = []
    all_images=np.array([])
    # print(len(dataset_Transmission_Color))
    for images,labels in dataset_Transmission_Color:

        for i in range(BATCH_SIZE):
            tsofli = 0
            if i < len(labels):
                for indx,mode in enumerate(Modes_Needed):
                    if tsofli==0:
                        csv_line__list = prediction_list[indx][counter]
                    else:
                        csv_line__list = np.append(csv_line__list, (prediction_list[indx][counter]))
                    tsofli += 1
                csv_line__list = np.append(csv_line__list, int(labels[i]))
                counter+=1
                with open(path_for_files+'/train.csv', 'a', encoding='UTF8', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(csv_line__list)
                    f.close()

    return

def Classifier_With_Probabilities(bool_1,bool_2,bool_3,bool_4,bool_5,bool_6,bool_7,bool_8,bool_9,bool_10,bool_11,bool_12,bool_13,bool_14,bool_15,bool_16,bool_17,bool_18,bool_19,bool_20):
    arg_list =[bool_1,bool_2,bool_3,bool_4,bool_5,bool_6,bool_7,bool_8,bool_9,bool_10,bool_11,bool_12,bool_13,bool_14,bool_15,bool_16,bool_17,bool_18,bool_19,bool_20]
    Display_Names = list(zip(arg_list, config.Display_Names))
    Reading_img_Names = list(zip(arg_list,config.Reading_Img_Names))
    prediction_list = []
    Modes_Needed = [y for x,y in Display_Names if x==True]
    Image_Directories = [y for x,y in Reading_img_Names if x==True]

    model_FC_no = len( os.listdir("../Models_Saved/Classifier_Multimodes"))+1
    os.mkdir("../Models_Saved/Classifier_Multimodes/model_{}".format(model_FC_no))
    path_for_files = "../Models_Saved/Classifier_Multimodes/model_{}".format(model_FC_no)
    with open('../Models_Saved/Classifier_Multimodes/model_{}/requirements.csv'.format(model_FC_no), 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(Modes_Needed)
        f.close()

    print("Models",Modes_Needed,Image_Directories)
    for names,ds_prediction,dataset in Load_Predictions(Modes_Needed):
        prediction_list.append((ds_prediction))

    Create_CSV(Modes_Needed,prediction_list,dataset,path_for_files)

    for diff_models in os.listdir("../Models_Saved/Classifier_Multimodes"):
        print("Mode: ",diff_models)
        requirement_check = pd.read_csv("../Models_Saved/Classifier_Multimodes/"+diff_models+"/requirements.csv")

        if Modes_Needed==(requirement_check.columns.values.tolist()):
            print("Great, this combination of modes exists, and CNNs are already trained!",requirement_check)
            df = pd.read_csv("../Models_Saved/Classifier_Multimodes/"+diff_models+'/train.csv')
            dataset = df.values

    number_of_classes = len(config.img_labels_str)
    column_number = number_of_classes*len(Modes_Needed)
    X = dataset[:,0:column_number]
    print("X data", np.array(X).shape)
    Y = dataset[:,column_number]
    print("Y data ", Y)
    min_max_scaler = preprocessing.MinMaxScaler()
    X_scale = min_max_scaler.fit_transform(X)
    X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scale, Y, test_size=0.3)
    X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)
    print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)


    input_shape = (column_number,)
    model = Sequential([
        Dense(1024, activation='relu', input_shape=input_shape),
        Dropout(0.3),
        Dense(512, activation='relu'),
        Dropout(0.3),
        Dense(number_of_classes, activation='softmax'),
    ])

    opt = keras.optimizers.Adam(learning_rate=INIT_LR,epsilon=0.1)
    # opt= keras.optimizers.RMSprop(lr=0.001, momentum=0.9,epsilon=0.1)
    model.compile(
        optimizer=opt,
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )

    history = model.fit(X_train, Y_train,
                     batch_size=BATCH_SIZE, epochs=EPOCHS,
                     validation_data=(X_val, Y_val))

    model_version = len(os.listdir("../Models_Saved/Classifier_Multimodes"))

    hist_df = pd.DataFrame(history.history)
    # save to json:
    hist_csv_file = '../Models_Saved/Classifier_Multimodes/model_{}/history.csv'.format(model_version)
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)


    model.evaluate(X_test, Y_test)
    model.summary()
    Save_New_Model_Multi(model,"Classifier_Multimodes")
    # loaded_model = Load_Latest_Model(None,"Classifier_FC")
    # Test_Set_Display((X_test,Y_test),loaded_model)  ##If you want to see the test set displayed in plots

    N = EPOCHS
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), history.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), history.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig("../Models_Saved/{}/model_{}/plot.png".format("Classifier_Multimodes", model_version))
    return

start_time = time.time()
##The 20 parameters of Classifier_With_Probabilities are the 20 modalities of the microscope.1st RGB 2-14 Hyperspectrals.15 Reflectance UV.16 Reflectance VIS.17-19 Fluorescence.20 Polarized
Classifier_With_Probabilities(True,True,False,False,False,False,False,True,False,False,True,False,False,True, True,True,True,True,True,True)#OLa ektos ta 5 prwta hyper telos
end_time = time.time()
print("Training Classifier takes ",start_time-end_time)
