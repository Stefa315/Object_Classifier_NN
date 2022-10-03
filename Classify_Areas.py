import os
from keras.models import model_from_json
import Save_And_Load_Models
import cv2
import config
import numpy as np
import imutils
from Collect_Dataset_Auto_Mode import Read_Hyperspectrals
from Find_Objects import Object_Segmentation_For_Any_Other
import Get_Tiles_256x256
from pathlib import Path
import time


class Classify_Area():
    def __init__(self,path,bool_1, bool_2, bool_3, bool_4, bool_5, bool_6, bool_7, bool_8, bool_9, bool_10, bool_11, bool_12,
                    bool_13, bool_14, bool_15, bool_16, bool_17, bool_18, bool_19, bool_20):
        self.path = path
        arg_list = [bool_1, bool_2, bool_3, bool_4, bool_5, bool_6, bool_7, bool_8, bool_9, bool_10, bool_11, bool_12,
                    bool_13, bool_14, bool_15, bool_16, bool_17, bool_18, bool_19, bool_20]
        Formal_Names = list(zip(arg_list, config.Display_Names))
        Reading_img_Names = list(zip(arg_list, config.Reading_Img_Names))
        self.Modes_Needed = [y for x, y in Formal_Names if x == True]
        self.Images_Directories = [y for x, y in Reading_img_Names if x == True]
        self.get_tiles = Get_Tiles_256x256.Tiles(self.Modes_Needed,self.Images_Directories)

    def Classify(self,Label,single,bandpass):

        rangemin=1
        rangemax=254
        wh=15
        col_counter = 0
        row_counter = 0
        folder_counter=1

        for count in range(1,255):
            if count==226: break
            subdir = "../SampleK/{}_Area/{}".format(Label,count)
            print("COUNT",count,subdir)
            threshold = 150
            if "Blood" == Label:
                Image_Transmission_Color = cv2.imread(subdir+'/mode_transmition_type_hyper/405.jpg')
                Image_Transmission_Color = imutils.resize(Image_Transmission_Color, width=640)
                sharpness=11.5
                threshold = 145
            if "Skin" == Label:
                Image_Transmission_Color = cv2.imread(subdir+'/mode_transmition_type_color/-1.jpg')
                Image_Transmission_Color = imutils.resize(Image_Transmission_Color, width=640)
                sharpness=9.8
                threshold=150
            if "Glass" == Label:
                Image_Transmission_Color = cv2.imread(subdir+'/mode_transmition_type_color/-1.jpg')
                # Image_Transmission_Color = cv2.imread(subdir+'/mode_transmition_type_hyper/450.jpg')
                Image_Transmission_Color = imutils.resize(Image_Transmission_Color, width=640)
                sharpness=10.5
                threshold = 165
            if "Fiber" == Label:
                Image_Transmission_Color = cv2.imread(subdir+'/mode_transmition_type_hyper/450.jpg')
                Image_Transmission_Color = imutils.resize(Image_Transmission_Color, width=640)
                bandpass = True
                threshold = 135
                sharpness= 9.1
            if "Hair" == Label:
                Image_Transmission_Color = cv2.imread(subdir+'/mode_transmition_type_color/-1.jpg')
                Image_Transmission_Color = imutils.resize(Image_Transmission_Color, width=640)
                sharpness=10
            if "Sand" == Label:
                Image_Transmission_Color = cv2.imread(subdir+'/mode_transmition_type_color/-1.jpg')
                Image_Transmission_Color = imutils.resize(Image_Transmission_Color, width=640)
                threshold = 110
                sharpness=9.5
            if single:
                pseudo_image = Object_Segmentation_For_Any_Other(threshold ,sharpness,subdir,Image_Transmission_Color,self.Modes_Needed,self.Images_Directories,self.get_tiles,bandpass)
                if row_counter==0:
                    Vconc = pseudo_image
                else:
                    Vconc = np.concatenate((pseudo_image,Vconc),axis=0)
                row_counter+=1
            else:
                pseudo_image  = Object_Segmentation_For_Any_Other(threshold,sharpness,subdir,Image_Transmission_Color,self.Modes_Needed,self.Images_Directories,self.get_tiles,bandpass)
                print("New Image Loaded ")
                if row_counter == 0:
                    Vconc = pseudo_image
                else:
                    Vconc = np.concatenate((Vconc, pseudo_image), axis=0)
                row_counter += 1

            if row_counter==wh:
                if col_counter==0:
                    Hconc = Vconc
                else:
                    Hconc = np.concatenate((Hconc,Vconc),axis=1)
                col_counter+=1
                if single==True:
                    single=False
                else:
                    single=True

                row_counter = 0
                cv2.resize(Hconc,(400,300))
            folder_counter+=1
        # print("Time without the loading of the images",time_no_load)
        # print("Time with loading the images",time)
        Hconc = imutils.resize(Hconc,width=5000)
        cv2.imwrite("../SampleK/Classified_Areas_HSI_1_7_10_13/Classified_Area_{}.jpg".format(Label),Hconc)
        return Hconc

    def Concatanate_Preview_Image(self,folder_directory):
        img_blood = cv2.imread(folder_directory+"Classified_Area_Blood.jpg")
        img_skin = cv2.imread(folder_directory+"Classified_Area_Skin.jpg")
        first_row = cv2.hconcat([img_blood,img_skin])
        img_sand = cv2.imread(folder_directory + "Classified_Area_Sand.jpg")
        img_glass = cv2.imread(folder_directory + "Classified_Area_Glass.jpg")
        second_row = cv2.hconcat([img_sand, img_glass])
        img_fiber = cv2.imread(folder_directory + "Classified_Area_Fiber.jpg")
        img_hair = cv2.imread(folder_directory + "Classified_Area_Hair.jpg")
        third_row = cv2.hconcat([img_fiber, img_hair])
        two_rows = cv2.vconcat([first_row,second_row])
        full_image = cv2.vconcat([two_rows,third_row])
        print(folder_directory)
        cv2.imwrite(folder_directory+"/Preview_Image.jpg",full_image)


    ##Run_All_Areas is classifying the regions we want , in total 6 like the specimens that we wish to classify 
    def Run_All_Areas(self):
        #
        for x in ["Glass","Sand","Blood","Fiber","Hair","Skin"]:
            if x=="Blood" or x=="Hair":
                single=False
            else:
                single=True
            Joker.Classify(x,single,True)

###Classify_Area is the constructor of the class that takes as argument the path for the images that are going to be classified
### that need to be in the same order as in the SampleK folder, and the next 20 arguments are the modes that we want to combine for the classification.
Joker = Classify_Area("../SampleK/Fiber_Area",True,True,False,False,False,False,False,True,False,False,True,False,False,True, True,True,True,True,True,True)
begin = time.time()
Joker.Run_All_Areas()
end = time.time()
print("Time for reconstructing all areas after classification",end-begin)

