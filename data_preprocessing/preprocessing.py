import pandas as pd
from sklearn.preprocessing import StandardScaler

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
        self.numerical_data = self.data.drop(["potential_issue","deck_risk","ppap_risk","stop_auto_buy","rev_stop"],axis=1)
        try:
            self.scaler = StandardScaler()
            self.scaled_data = self.scaler.fit_transform(self.numerical_data)
            self.scaled_df = pd.DataFrame(data = self.scaled_data, columns=self.numerical_data.columns, index = data.self)
            self.data.drop(columns = self.scaled_df.columns, inplace=True)
            self.data = pd.concat([self.scaled_df, self.data], axis=1)
            self.logger_object.log(self.file_object, 'Success : Scaled all the numerical values ')
            return self.data
        except Exception as e:
            self.logger_object.log(self.file_object, 'Failure : Scaled all the numerical values '+ str(e))
            raise Exception()

