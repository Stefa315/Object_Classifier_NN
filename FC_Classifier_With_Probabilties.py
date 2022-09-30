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
from IPython.display import display
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

        # print("LENGTH OF DATASET",len(dataset_Transmission_Color))
        globals()["train_"+names], globals()["val_trans_"+names],globals()["test_"+names] = get_dataset.get_dataset_partitions(globals()["dataset_"+names])

        globals()["model_" + names] = Load_Latest_Model(None,names)

        globals()["ds_prediction_" + names] = (globals()["model_"+names]).predict(globals()["dataset_"+names])
        # print("OOOOOOOOOOOO",globals()["ds_prediction_"+names])
        # yield names,globals()["ds_prediction_"+names],dataset_Transmission_Color
        yield names, globals()["ds_prediction_" + names], dataset_Transmission_Color

    # model_trans_325 = Load_Latest_Model(None, "Transmission_Hyperspectral_325")
    # model_trans_340 = Load_Latest_Model(None, "Transmission_Hyperspectral_340")
    # model_trans_365 = Load_Latest_Model(None, "Transmission_Hyperspectral_365")
    # model_trans_385 = Load_Latest_Model(None, "Transmission_Hyperspectral_385")
    # model_trans_405 = Load_Latest_Model(None, "Transmission_Hyperspectral_405")
    # model_trans_450 = Load_Latest_Model(None, "Transmission_Hyperspectral_450")
    # model_trans_490 = Load_Latest_Model(None, "Transmission_Hyperspectral_490")
    # model_trans_515 = Load_Latest_Model(None, "Transmission_Hyperspectral_515")
    # model_trans_590 = Load_Latest_Model(None, "Transmission_Hyperspectral_590")
    # model_trans_630 = Load_Latest_Model(None, "Transmission_Hyperspectral_630")
    # model_trans_750 = Load_Latest_Model(None, "Transmission_Hyperspectral_750")
    # model_trans_850 = Load_Latest_Model(None, "Transmission_Hyperspectral_850")
    # model_trans_980 = Load_Latest_Model(None, "Transmission_Hyperspectral_980")
    # model_trans_color = Load_Latest_Model(None,"Transmission_Color")
    # model_reflectance_UV = Load_Latest_Model(None,"Reflectance_UV")
    # model_reflectance_VIS = Load_Latest_Model(None, "Reflectance_VIS")
    # model_fluo = Load_Latest_Model(None,"Fluorescence")
    # model_fluo_2 = Load_Latest_Model( None,"Fluorescence_2")
    # model_fluo_3 = Load_Latest_Model(None, "Fluorescence_3")
    # model_polar = Load_Latest_Model(None,"Polar")
    #
    # a_0 = model_trans_color.predict(dataset_Transmission_Color)
    # a_1 = model_trans_325.predict(dataset_Transmission_Hyperspectral_325)
    # a_2 = model_trans_340.predict(dataset_Transmission_Hyperspectral_340)
    # a_3 = model_trans_365.predict(dataset_Transmission_Hyperspectral_365)
    # a_4 = model_trans_385.predict(dataset_Transmission_Hyperspectral_385)
    # a_5 = model_trans_405.predict(dataset_Transmission_Hyperspectral_405)
    # a_6 = model_trans_405.predict(dataset_Transmission_Hyperspectral_450)
    # a_7 = model_trans_490.predict(dataset_Transmission_Hyperspectral_490)
    # a_8 = model_trans_515.predict(dataset_Transmission_Hyperspectral_515)
    # a_9 = model_trans_590.predict(dataset_Transmission_Hyperspectral_590)
    # a_10 = model_trans_630.predict(dataset_Transmission_Hyperspectral_630)
    # a_11 = model_trans_750.predict(dataset_Transmission_Hyperspectral_750)
    # a_12 = model_trans_850.predict(dataset_Transmission_Hyperspectral_850)
    # a_13 = model_trans_405.predict(dataset_Transmission_Hyperspectral_980)
    #
    # c = model_reflectance_UV.predict(dataset_Reflectance_UV)
    # c_2 = model_reflectance_VIS.predict(dataset_Reflectance_VIS)
    # e = model_fluo.predict(dataset_Fluorescence)
    # e_2 = model_fluo_2.predict(dataset_Fluorescence_2)
    # e_3 = model_fluo_3.predict(dataset_Fluorescence_3)
    # f = model_polar.predict(dataset_Polar)
    #
    # return a_0,a_1,a_2,a_3,a_4,a_5,a_6,a_7,a_8,a_9,a_10,a_11,a_12,a_13,c,c_2,e,e_2,e_3,f,dataset_Transmission_Color
def Header_Creation(Modes_Needed):
    header = []
    for modes in Modes_Needed:
        for i in range(1,len(config.img_labels_str)+1):
            header.append(modes+"_"+str(i))
    header.append('Label')
    print(header)
    return header
header_csv = [
              'trans_color_1', 'trans_color_2', 'trans_color_3', 'trans_color_4', 'trans_color_5', 'trans_color_6','trans_color_7', 'trans_color_8', 'trans_color_9', 'trans_color_10',  'trans_color_11',
              'trans_325_1','trans_325_2','trans_325_3','trans_325_4','trans_325_5','trans_325_6','trans_325_7','trans_325_8','trans_325_9','trans_325_10','trans_325_11',
              'trans_340_1', 'trans_340_2', 'trans_340_3', 'trans_340_4', 'trans_340_5', 'trans_340_6', 'trans_340_7','trans_340_8', 'trans_340_9', 'trans_340_10','trans_340_11',
              'trans_365_1', 'trans_365_2', 'trans_365_3', 'trans_365_4', 'trans_365_5', 'trans_365_6','trans_365_7', 'trans_365_8', 'trans_365_9', 'trans_365_10','trans_365_11',
              'trans_385_1', 'trans_385_2', 'trans_385_3', 'trans_385_4', 'trans_385_5', 'trans_385_6','trans_385_7', 'trans_385_8', 'trans_385_9', 'trans_385_10','trans_385_11',
              'trans_405_1', 'trans_405_2', 'trans_405_3', 'trans_405_4', 'trans_405_5', 'trans_405_6','trans_405_7', 'trans_405_8', 'trans_405_9','trans_405_10', 'trans_405_11',
              'trans_450_1', 'trans_450_2', 'trans_450_3', 'trans_450_4', 'trans_450_5', 'trans_450_6','trans_450_7', 'trans_450_8', 'trans_450_9','trans_450_10', 'trans_450_11',
              'trans_490_1', 'trans_490_2', 'trans_490_3', 'trans_490_4', 'trans_490_5', 'trans_490_6','trans_490_7', 'trans_490_8', 'trans_490_9','trans_490_10', 'trans_490_11',
              'trans_515_1', 'trans_515_2', 'trans_515_3', 'trans_515_4', 'trans_515_5', 'trans_515_6','trans_515_7', 'trans_515_8', 'trans_515_9','trans_515_10', 'trans_515_11',
              'trans_590_1', 'trans_590_2', 'trans_590_3', 'trans_590_4', 'trans_590_5', 'trans_590_6','trans_590_7', 'trans_590_8', 'trans_590_9','trans_590_10', 'trans_590_11',
              'trans_630_1', 'trans_630_2', 'trans_630_3', 'trans_630_4', 'trans_630_5', 'trans_630_6','trans_630_7', 'trans_630_8', 'trans_630_9','trans_630_10', 'trans_630_11',
              'trans_750_1', 'trans_750_2', 'trans_750_3', 'trans_750_4', 'trans_750_5', 'trans_750_6','trans_750_7', 'trans_750_8', 'trans_750_9','trans_750_10', 'trans_750_11',
              'trans_850_1', 'trans_850_2', 'trans_850_3', 'trans_850_4', 'trans_850_5', 'trans_850_6','trans_850_7', 'trans_850_8', 'trans_850_9','trans_850_10', 'trans_850_11',
              'trans_980_1', 'trans_980_2', 'trans_980_3', 'trans_980_4', 'trans_980_5', 'trans_980_6','trans_980_7', 'trans_980_8', 'trans_980_9','trans_980_10', 'trans_980_11',
              'ref_UV_1','ref_UV_2','ref_UV_3','ref_UV_4','ref_UV_5','ref_UV_6','ref_UV_7','ref_UV_8','ref_UV_9','ref_UV_10','ref_UV_11' ,
              'ref_VIS_1','ref_VIS_2', 'ref_VIS_3', 'ref_VIS_4', 'ref_VIS_5', 'ref_VIS_6', 'ref_VIS_7', 'ref_VIS_8', 'ref_VIS_9','ref_VIS_10', 'ref_VIS_11',
              'fluo_1_1','fluo_1_2','fluo_1_3','fluo_1_4','fluo_1_5','fluo_1_6','fluo_1_7','fluo_1_8','fluo_1_9','fluo_1_10','fluo_1_11',
              'fluo_2_1', 'fluo_2_2','fluo_2_3','fluo_2_4','fluo_2_5', 'fluo_2_6', 'fluo_2_7','fluo_2_8','fluo_2_9','fluo_2_10','fluo_2_11',
              'fluo_3_1', 'fluo_3_2','fluo_3_3','fluo_3_4','fluo_3_5', 'fluo_3_6', 'fluo_3_7','fluo_3_8','fluo_3_9','fluo_3_10','fluo_3_11',
              'polar_1','polar_2','polar_3','polar_4','polar_5','polar_6','polar_7','polar_8','polar_9','polar_10','polar_11',
              'label']

# def Create_CSV(trans_color_pred,trans_325_pred,trans_340_pred,trans_365_pred,trans_385_pred,trans_405_pred,trans_450_pred,trans_490_pred,trans_515_pred,trans_590_pred,trans_630_pred,trans_750_pred,\
#                 trans_850_pred,trans_980_pred, ref_uv_pred, ref_vis_pred, fluo_pred, fluo_2_pred,fluo_3_pred, polar_pred,dataset_Transmission_Color):
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
                        # csv_line__list.append((prediction_list[indx][counter]))
                    tsofli += 1
                    # print("KALAMATA",prediction_list[indx][counter])
                csv_line__list = np.append(csv_line__list, int(labels[i]))
                # csv_line__list.append(int(labels[i]))
                counter+=1
                with open(path_for_files+'/train.csv', 'a', encoding='UTF8', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(csv_line__list)
                    f.close()
            # csv_line__list.clear()

    # for images,labels in dataset_Transmission_Color:
    #         for i in range(num_classes):
    #             if i < len(labels):
    #                 # print("EEEEE",i,counter,len(labels),np.array(labels[i]).astype('int'))
    #                 csv_line = trans_color_pred[counter]
    #                 csv_line = np.append(csv_line, trans_325_pred[counter])
    #                 csv_line = np.append(csv_line, trans_340_pred[counter])
    #                 csv_line = np.append(csv_line, trans_365_pred[counter])
    #                 csv_line = np.append(csv_line, trans_385_pred[counter])
    #                 csv_line = np.append(csv_line, trans_405_pred[counter])
    #                 csv_line = np.append(csv_line, trans_450_pred[counter])
    #                 csv_line = np.append(csv_line, trans_490_pred[counter])
    #                 csv_line = np.append(csv_line, trans_515_pred[counter])
    #                 csv_line = np.append(csv_line, trans_590_pred[counter])
    #                 csv_line = np.append(csv_line, trans_630_pred[counter])
    #                 csv_line = np.append(csv_line, trans_750_pred[counter])
    #                 csv_line = np.append(csv_line, trans_850_pred[counter])
    #                 csv_line = np.append(csv_line, trans_980_pred[counter])
    #                 csv_line = np.append(csv_line,ref_uv_pred[counter])
    #                 csv_line = np.append(csv_line, ref_vis_pred[counter])
    #                 csv_line = np.append(csv_line, fluo_pred[counter])
    #                 csv_line = np.append(csv_line, fluo_2_pred[counter])
    #                 csv_line = np.append(csv_line, fluo_3_pred[counter])
    #                 csv_line = np.append(csv_line, polar_pred[counter])
    #                 csv_line = np.append(csv_line,[np.array(labels[i]).astype('int')])
    #                 if counter ==0:
    #                     all_images = np.expand_dims(images[i],0)
    #                 else :
    #                     all_images = np.append(all_images,np.expand_dims(images[i],0),axis=0)
    #                 counter+=1
    #                 # print("CSV_LINe",all_images.shape,all_images[0])
    #                 print("trans_color",trans_color_pred[0])
    #                 print("trans_325",trans_325_pred[0])
    #                 print("CSV_LINe", csv_line)
    #
    #             with open('train.csv', 'a', encoding='UTF8', newline='') as f:
    #                 writer = csv.writer(f)
    #                 writer.writerow(csv_line)
    #                 f.close()
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

    # trans_color_pred,trans_325_pred,trans_340_pred,trans_365_pred,trans_385_pred,trans_405_pred,trans_450_pred,trans_490_pred,trans_515_pred,trans_590_pred,trans_630_pred,trans_750_pred,\
    # trans_850_pred,trans_980_pred, ref_uv_pred, ref_vis_pred, fluo_pred, fluo_2_pred,fluo_3_pred, polar_pred ,dataset_Transmission_Color = Load_Predictions(Modes_Needed)
    #
    # Create_CSV(trans_color_pred,trans_325_pred,trans_340_pred,trans_365_pred,trans_385_pred,trans_405_pred,trans_450_pred,trans_490_pred,trans_515_pred,trans_590_pred,trans_630_pred,trans_750_pred,\
    # trans_850_pred,trans_980_pred, ref_uv_pred, ref_vis_pred, fluo_pred, fluo_2_pred,fluo_3_pred, polar_pred , dataset_Transmission_Color)
    for diff_models in os.listdir("../Models_Saved/Classifier_Multimodes"):
        print("DIFF MDOEW",diff_models)
        requirement_check = pd.read_csv("../Models_Saved/Classifier_Multimodes/"+diff_models+"/requirements.csv")
        if Modes_Needed==(requirement_check.columns.values.tolist()):
            print("VRETHIKE",requirement_check)
            df = pd.read_csv("../Models_Saved/Classifier_Multimodes/"+diff_models+'/train.csv')
            dataset = df.values

    print("DATASET",dataset.shape)
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
    # Test_Set_Display((X_test,Y_test),loaded_model)
    model.summary()

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
Classifier_With_Probabilities(True,True,False,False,False,False,False,True,False,False,True,False,False,True, True,True,True,True,True,True)#OLa ektos ta 5 prwta hyper telos
end_time = time.time()
print("Training Classifier takes ",start_time-end_time)
