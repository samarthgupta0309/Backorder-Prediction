import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class Preprocessor:
    '''
    Clean and transform data
    functions : 

    '''
    def __init__(self, file_object, logger_object):
        self.file_object = file_object
        self.logger_object = logger_object
    
    def encodeCatFeature(self, data):
        '''
        Encode categorical feature into values
        '''
        self.logger_object.log(self.file_object, 'Started : Preprocessor -> encodeCatFeature')
        try :
            categorical_col = ["potential_issue","deck_risk","oe_constraint","ppap_risk","stop_auto_buy","rev_stop","went_on_backorder"]
            for col in categorical_col:
                data[col] = data[col].map({
                    "Yes" : 0,
                    "No" : 1
                })
            self.logger_object.log(self.file_object, 'Success : Preprocessing of categorical vairable')
            return data
        except Exception as e:
            self.logger_object.log(self.file_object, 'Failure : Preprocessing of categorical vairable'+str(e))
            raise Exception()
    
    def drop_column(self, data, column):
        '''
        Drop the column not needed
        param : 
            - data - csv file 
            - column - column name to be droped from the data
        '''
        self.logger_object.log(self.file_object, 'Started : Preprocessor -> drop_column')
        self.data = data
        self.column = column
        try:
            self.data.drop(self.column, axis=1, inplace = True)
            self.logger_object.log(self.file_object, 'Success : Removed column')
            return self.data
        except Exception as e:
            self.logger_object.log(self.file_object, "Failure : Removed column "+ str(e))
            raise Exception()
    
    def scale_numerical_col(self, data):
        '''
        Scaling value in data
        param :
            - data - dataframe which have to scaled
        '''
        self.logger_object.log(self.file_object, 'Started : Preprocessor -> scale_numerical_col')
        self.data = data
        self.numerical_data = self.data.drop(["potential_issue","deck_risk","ppap_risk","stop_auto_buy","rev_stop","went_on_backorder"],axis=1)
        try:
            self.scaler = StandardScaler()
            self.scaled_data = self.scaler.fit_transform(self.numerical_data)
            self.scaled_df = pd.DataFrame(data = self.scaled_data, columns=self.numerical_data.columns, index = self.data.index)
            self.data.drop(columns = self.scaled_df.columns, inplace=True)
            self.data = pd.concat([self.scaled_df, self.data], axis=1)
            self.data.to_csv('data_preprocessing/scaled_data.csv')
            self.logger_object.log(self.file_object, 'Success : Scaled all the numerical values ')
            return self.data
        except Exception as e:
            self.logger_object.log(self.file_object, 'Failure : Scaled all the numerical values '+ str(e))
            raise Exception()

    def check_null(self, data):
        self.logger_object.log(self.file_object, 'Started : Preprocessor -> check_null')
        self.null_present = False
        try:
            self.null_count = data.isna().sum()
            self.null_column = []
            for col, val in zip(list(data.columns), self.null_count):
                if val > 0:
                    self.null_present = True
                    self.null_column.append(col)
            if(self.null_present):
                self.logger_object.log(self.file_object, "Null columns : ")
                for col in self.null_column:
                    # print(f"{col} ")
                    self.logger_object.log(self.file_object," "+str(col))
            self.logger_object.log(self.file_object, "Success : check_null completed")
            return self.null_column
        except Exception as e:
            self.logger_object.log(self.file_object, "Failure : check_null "+ str(e))

    def split(self, X, y):
        '''
        Split train test in ratio X_train, X_test, y_train, y_test
        train : test ratio   70:30
        param : 
            - X - data excluding null value
            - y - target
        '''
        self.logger_object.log(self.file_object, "Started : Preprocessor -> split")
        try:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            self.logger_object.log(self.file_object, "Success : Split Completed")
            return self.X_train, self.X_test, self.y_train, self.y_test
        except Exception as e:
            self.logger_object.log(self.file_object, "Failure : split uncompleted")
            raise Exception() 

