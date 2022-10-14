import os
import cv2
import config
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models , layers
from collections import Counter
import Get_Tiles_256x256
import math
import imutils
import time
time1=0
time2=0

def Object_Segmentation_For_Any_Other(threshold,sharpness,subdir,Image_Transmission_Color,Modes_Needed,Images_Directories,get_tiles,bandpass):

    colormap = ((0, 12, 252), (0, 50, 100), (95, 255, 0), (252, 252, 12), (50, 220, 255), (180, 50, 255), (255, 0, 61),
                (255, 17, 0))
    Label_List_of_All_Tiles = np.empty((20,0)).tolist()    ## DHMIOURGW ENAN ADEIO PIANAKA DIASTASEWN (10,0) kai ton kanw convert se list
    predictions_main = np.zeros((15,8)).tolist()
    most_common_Label = np.empty((8,2))
    if bandpass:
        sharpen_kernel = np.array([[-1, -1, -1], [-1, sharpness, -1], [-1, -1, -1]]) ##==HIGH BAND PASS FILTER
        image_filtered = cv2.filter2D(Image_Transmission_Color, -1, sharpen_kernel)
    else:
        image_filtered = Image_Transmission_Color.copy()
    # apply binary thresholding
    img_gray = cv2.cvtColor(image_filtered, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_gray, threshold , 255, cv2.THRESH_BINARY)
    ##TA COUNTOURS EMPERIKLYOUN TA ANTIKEIMENA POY EINAI KONTA STO LEYKO XRWMA ,EMEIS THELOYME TO ANTITHETO , ETSI ANTISTREFW TA XRWMATA LEYKO->MAYRO , MAURO-> LEYKO
    for i in range(Image_Transmission_Color.shape[0]):
        for y in range(Image_Transmission_Color.shape[1]):
            if (thresh[i][y]==255):
                thresh[i][y]=0
            else:
                thresh[i][y]=255
    contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_NONE)
    # draw contours on the original image
    image_pseudo = cv2.imread(subdir + '/mode_transmition_type_color/-1.jpg')
    image_pseudo = imutils.resize(image_pseudo, width=640)
    image_copy = image_pseudo.copy()
    image_3rd_copy = image_pseudo.copy()

    temp_img = cv2.drawContours(image=Image_Transmission_Color, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=10,lineType=cv2.LINE_AA)
    font = cv2.FONT_HERSHEY_SIMPLEX
    color=(255,0,0)
    k=0
    # print("OBJECTS NEED TO BE BIGGER THAN :", 0.05*((image.shape[0]*image.shape[1])/2),   cv2.contourArea(contour))
    for contour in contours:

        if cv2.contourArea(contour)>(0.03*((Image_Transmission_Color.shape[0]*Image_Transmission_Color.shape[1])/2)):

            # Find bounding rectangles
            print("OBJECTS NEED TO BE BIGGER THAN :", 0.015 * ((Image_Transmission_Color.shape[0] * Image_Transmission_Color.shape[1]) / 2),cv2.contourArea(contour))
            x,y,w,h = cv2.boundingRect(contour)
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            predictions_main,time,time_no_load = get_tiles.Get_256x256_Tiles_Of_All(subdir,Image_Transmission_Color,Modes_Needed,Images_Directories,x,y,w,h)
            # print("PRedictions main ",predictions_main)
            for indx , predict in enumerate(predictions_main): ##ANALYSING LABELS AND CONFIDENCE OF CLASSIFICATION
                print("INDEX_of_prob",indx ,"\nMax Probability",np.max(predict),"Label",config.img_labels_str[np.argmax(predict)])#,"\n", predict)
                # print("TA PANTA OLA",indx , predict)
                if (np.max(predict)>0.9) :
                    if "Bubbles"== config.img_labels_str[np.argmax(predict)]: continue
                    print("Bubblesss",config.img_labels_str[np.argmax(predict)])
                    Label_for_Each_Tile = config.img_labels_str[np.argmax(predict)] ##K = ARITHMOS CONTOUR
                    Label_List_of_All_Tiles[k].append(Label_for_Each_Tile)  ##STHN SEIRA K PROSTHESE TO PREDICT APO KATHE TILE
                    counter = Counter(Label_List_of_All_Tiles[k])
                    most_common_Label = np.array(counter.most_common(5)).copy()
            print("LABEL_LIST",most_common_Label[0][0],most_common_Label,np.array(most_common_Label).shape)
            # print("Length of sorted ",len(np.array(most_common_Label)))
            if (((most_common_Label[0][0]=='No_Object') or (most_common_Label[0][0]=='Bubbles')) and ((len(np.array(most_common_Label)))>=2)): # EAN TO PRWTO STH LISTA MOST COMMON EINAI NO_OBJECT
                try :
                    if most_common_Label[0][0]=='Bubbles': continue
                    if (most_common_Label[1][1]>=most_common_Label[1][1]//2) :
                        text_label = most_common_Label[1][0]
                        pos = config.img_labels_str.index(text_label)
                        my_colormap = colormap[pos]
                        image_pseudo = cv2.drawContours(image_pseudo, contour, contourIdx=-1, color=my_colormap, thickness=10) ##PSEUDOCOLOR
                except :
                    text_label = 'Empty'
            elif (most_common_Label[0][0]) not in config.img_labels_str or ("Bubbles"==most_common_Label[0][0] or ("No_Object"==most_common_Label[0][0])):
                text_label = 'Empty'
            else :
                text_label = most_common_Label[0][0]
                pos = config.img_labels_str.index(text_label)
                my_colormap = colormap[pos]
                image_pseudo=cv2.drawContours(image_pseudo, contour, contourIdx=-1, color=my_colormap, thickness=10)
                k+=1 #Contour Count
            print("Final LABEL",text_label)
            text_label = 'Empty'
    return image_pseudo


