from datetime import datetime
import joblib
import pandas as pd
from app_logging.logging import App_Logger
from data_loading.data_loader import Data_load
from data_preprocessing.preprocessing import Preprocessor
from model_find.model_dispatcher import Make_model

if __name__ == "__main__":
    file_object = open("Logs/log.txt", 'a+')
    log_writer = App_Logger()
    try:
        data_get = Data_load(file_object, log_writer)
        log_writer.log(file_object, "\n")
        data = data_get.get_data()
        preprocess = Preprocessor(file_object, log_writer)
        data = preprocess.drop_column(data, ["Index_Product", "sku"])
        data = preprocess.encodeCatFeature(data)
        null_col = preprocess.check_null(data)
        if len(null_col) != 0:
            data.dropna(inplace=True)
        data = preprocess.scale_numerical_col(data)
        X = data.drop(['went_on_backorder'], axis=1)
        y = data['went_on_backorder']
        X_train, X_test, y_train, y_test = preprocess.split(X, y)
        make_model = Make_model(file_object, log_writer)
        make_model.best_model(X_train, y_train)
        load_model = joblib.load(open('model_find/model.sav', 'rb'))
        result = load_model.score(X_test, y_test)
        log_writer.log(file_object, "Result of model "+str(result))
        log_writer.log(file_object, "Success in task !!")
        file_object.close()
    except Exception as e:
        log_writer.log(file_object, "Failure "+ str(e))
        file_object.close()
        raise Exception


