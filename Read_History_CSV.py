import pandas as pd
import matplotlib as plt
import numpy as np
import os
from scipy.signal import find_peaks



class csv_metrics:
    def __init__(self,path):
        self.path = path
        self.data =  pd.read_csv(path)
        self.model_no = os.path.basename(
        os.path.dirname(path))

    def read_data(self,path):
        data = pd.read_csv(path)
        return data

    def print_data(self):
        print("Data: ",self.data)

    def print_column(self,col_name):
        print(self.data[col_name].iloc[50:])

    def mean_col_value(self,col_name,n_last):
        print("{} : {}".format(col_name,self.model_no))
        print(self.data[col_name].iloc[n_last:].mean())
        print('\n')
    def var_col_value(self,col_name,n_last):
        print("Variance of {} : {}".format(col_name,self.model_no))
        print(self.data[col_name].iloc[n_last:].var())
        print('\n')

    def find_peakss(self,col_name):
        peaks,_ =find_peaks(self.data[col_name],distance=10)
        print("{}".format(col_name))
        print(self.data[col_name].iloc[peaks])


c1 = csv_metrics('../Models_Saved/Transmission_Color/model_4/history.csv')
c1.mean_col_value('val_accuracy',100)
c1.var_col_value('val_accuracy',100)
c1.mean_col_value('accuracy',100)
c1.find_peakss('val_accuracy')
c1.find_peakss('accuracy')