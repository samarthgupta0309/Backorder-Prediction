from datetime import datetime
import os
import pandas as pd
from app_logging.logging import App_Logger
from data_loading.data_loader import Data_load
from data_preprocessing.preprocessing import Preprocessor

if __name__ == "__main__":
    file_object = open("Logs/log.txt", 'a+')
    log_writer = App_Logger()
    try:
        data_get = Data_load(file_object, log_writer)
        data = data_get.get_data()
        preprocess = Preprocessor(file_object, log_writer)
        data = preprocess.drop_column(data, ["Index_Product", "sku"])
        data = preprocess.encodeCatFeature(data)
        print(data.head(10))
        log_writer.log(file_object, 'Success in the task')
        file_object.close()
    except Exception as e:
        log_writer.log(file_object, "Failure "+ str(e))
        file_object.close()
        raise Exception


