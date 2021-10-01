from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
import joblib

models = {
    "decision_tree" : DecisionTreeClassifier(criterion="gini"),
    "random_forest" : RandomForestClassifier(),
    "extra_tree" : ExtraTreesClassifier(),
    "adaboost" : AdaBoostClassifier(),
    "gradient_boost" : GradientBoostingClassifier()
}
class Make_model:
    '''
    making model and giving the best one from 
    models = {
    "decision_tree" : DecisionTreeClassifier(criterion="gini"),
    "random_forest" : RandomForestClassifier(),
    "extra_tree" : ExtraTreesClassifier(),
    "adaboost" : AdaBoostClassifier(),
    "gradient_boost" : GradientBoostingClassifier()
    }
    '''
    def __init__(self, file_object, logger_object):
        self.file_object = file_object
        self.logger_object = logger_object
    
    def best_model(self, X_train, y_train):
        '''
        finding the best model using cross_validation_score 
        with CV = 5
        '''
        self.logger_object.log(self.file_object, "Started : model_dispatcher -> best_model")
        try:
            self.val_score = []
            for key in models:
                scr_train = cross_val_score(models[key], X_train, y_train, cv=5)
                self.val_score.append([key, scr_train.mean()])
            # finding info best model
            self.max_scr = self.val_score[0][1]
            self.mod = self.val_score[0][0]
            for idx in range(1, len(self.val_score)):
                self.max_scr = max(self.max_scr, self.val_score[idx][1])
                if(self.val_score[idx][1] == self.max_scr):
                    self.mod = self.val_score[idx][0]
            self.logger_object.log(self.file_object, "Success in finding best model : "+str(self.mod)+" "+str(self.max_scr))
            try:
                self.model = models[self.mod]
                self.model.fit(X_train, y_train)
                joblib.dump(self.model, open('model_find/model.sav', 'wb'))
                self.logger_object.log(self.file_object, "Success dumped model ")
                return self
            except Exception as e:
                self.logger_object.log(self.file_object, "Failure in saving the model "+ str(e))
                raise Exception()
        except Exception as e:
            self.logger_object.log(self.file_object, "Failure : Finding best_model "+str(e))
            raise Exception()



            
        
        

