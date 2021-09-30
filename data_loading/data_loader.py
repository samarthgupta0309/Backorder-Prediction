import pandas as pd
import os
import numpy as np

class Data_load:
    '''
    Class Data_load for obtaining the data from source for training & testing
    '''
    def __init__(self, file_object, logger_object):
        self.path = 'dataset/InputFile.csv'
        self.file_object = file_object
        self.logger_object = logger_object

    def get_data(self):
        '''
        Reads data from source
        output: A pandas DataFrame
        '''
        self.logger_object.log(self.file_object, 'Start : Data_load -> get_data')
        try:
            self.data = pd.read_csv(self.path)
            self.logger_object.log(self.file_object, 'Success : Data loaded')
            return self.data
        except Exception as e:
            self.logger_object.log(self.logger_object, 'Failure : Data loaded unsuccessful' + str(e))
            raise Exception()