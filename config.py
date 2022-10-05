import os
import tensorflow as tf
from tensorflow.keras import models
from os import listdir
from keras.models import model_from_json

Display_Names = ['Transmission_Color','Transmission_Hyperspectral_325','Transmission_Hyperspectral_340','Transmission_Hyperspectral_365','Transmission_Hyperspectral_385','Transmission_Hyperspectral_405','Transmission_Hyperspectral_450',
                     'Transmission_Hyperspectral_490','Transmission_Hyperspectral_515','Transmission_Hyperspectral_590','Transmission_Hyperspectral_630','Transmission_Hyperspectral_750','Transmission_Hyperspectral_850',
                     'Transmission_Hyperspectral_980','Reflectance_UV','Reflectance_VIS','Fluorescence','Fluorescence_2','Fluorescence_3','Polar']

Reading_Img_Names = ['mode_transmition_type_color/-1.jpg','mode_transmition_type_hyper/325.jpg','mode_transmition_type_hyper/340.jpg','mode_transmition_type_hyper/365.jpg','mode_transmition_type_hyper/385.jpg','mode_transmition_type_hyper/405.jpg'
                     ,'mode_transmition_type_hyper/450.jpg','mode_transmition_type_hyper/490.jpg','mode_transmition_type_hyper/515.jpg','mode_transmition_type_hyper/590.jpg','mode_transmition_type_hyper/630.jpg'
                     ,'mode_transmition_type_hyper/750.jpg','mode_transmition_type_hyper/850.jpg','mode_transmition_type_hyper/980.jpg','mode_reflectance_uv_type_color/-1.jpg','mode_reflectance_vis_type_color/-1.jpg',
                     'mode_fluorescence_blue_type_color/-1.jpg','mode_fluorescence_uv_365_type_color/-1.jpg','mode_fluorescence_uv_405_type_color/-1.jpg','mode_polarization_type_color/polar_DoLP.jpg']


Test_Display_Name = ['Transmission_Color']

cropped_images_names = ['trans_color','trans_325','trans_340','trans_365','trans_385','trans_405','trans_450','trans_490','trans_515','trans_590','trans_630',
                        'trans_750','trans_850','trans_980','refl_UV','refl_VIS','fluo_1','fluo_2','fluo_3','polar']

mode_names = ['mode_fluorescence_uv_365_type_color','mode_fluorescence_uv_405_type_color','mode_fluorescence_blue_type_color','mode_polarization_type_color','mode_reflectance_uv_type_color'
              ,'mode_reflectance_vis_type_color','mode_transmition_type_color','mode_transmition_type_hyper']


auto_mode_names = ['mode_transmition_type_hyper','mode_transmition_type_color','mode_reflectance_vis_type_color','mode_reflectance_uv_type_color','mode_polarization_type_color','mode_fluorescence_uv_405_type_color','mode_fluorescence_blue_type_color','mode_fluorescence_uv_365_type_color']


img_labels_str = os.listdir("../Dataset/Dataset_Transmission_Hyperspectral_325")



