# Object_Classifier_NN
## Multi-Spectral Object Classification using Convolutional Neural Networks trained with unique dataset acquired from Microscope.

![example workflow](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue)
![example workflow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![example workflow](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=Keras&logoColor=white)
![example workflow](https://img.shields.io/badge/PyCharm-000000.svg?&style=for-the-badge&logo=PyCharm&logoColor=white)

This is my thesis project, and it contains object classification for specimens taken from a High-Throughput Microscope(Blood, Sand, Skin, Fiber, Hair, Glass) with the use of Convolutional Neural Networks and a Fully-Connected Feed Forward Neural Network to combine the results of the CNNs. The Dataset for the training process is unique and taken by hand by me **(Link for Dataset and the images that i reconstructed{SampleK} are inside the Dataset folder)**. 


A Microscope can capture the same specimen in many different appearances depending on its capabilities. The one i could use in the lab had 5 different modalities
for capturing an image. The dataset i created is unique so it may be difficult to be tested on other images with different specifications or appearances but it is 
programmed to work with any kind of combination that you choose between the available 20 capturing modalities(1-20). So, you can test this one with the existing dataset
or create your own dataset and train again this CNNs.


Microscope's Utilities :

- Brightfield Microscopy(RGB Image)
- Reflectance Microscopy: Ultraviolet Spectrum and Visible Spectrum
- Polarized Microscopy
- Fluorescence Microscopy: Light Source at 365nm, 405nm, 450nm
- Hyperspectral Microscopy: 13 Bands between 325nm-1000nm


# HOW TO USE :
- Download the zipfile and unzip it inside your project.
- Place the modules inside your environment.
- Create a Models_Saved folder inside your project. After training each Convolutional NN the folders will be created automatically and the models will be saved inside with numerical order. You can and must specify the combinations of modalities
that will be used for classifying in the Classify_Areas.py module and make sure you have created the model with this specific combination you want. There are already
11 combinations that i have tried and you can find them in my GoogleDrive inside the Classifier_Multimodes folder in the readme.txt file.
- Download the Dataset from my GoogleDrive, the link is inside the readme.txt in the Dataset Folder.
- The CNN_Bundle.py trains and saves the models for each modality in a folder which be created after the training phase.
- Download the SampleK folder from my GoogleDrive, the link is inside the readme.txt in the SamlpeK Folder.
- Place the SampleK folder inside your project.
- Classify_Areas.py will classify all the aeras of the SampleK folder and reconstruct 6 images with the classified specimens and place the images inside 
the folder you want. **Line 105**



