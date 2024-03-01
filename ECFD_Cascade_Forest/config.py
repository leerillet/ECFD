from sklearn.metrics import mean_squared_error,r2_score
from scipy.stats import pearsonr
import numpy as np

def get_config(forests_type,n_estimator):
    config={}
    config["estimator_parameters"]=[]
    if forests_type == 'LGBMRF':
        config["estimator_parameters"].append({"n_fold":5,"type":"LGBMRegressor",
                                               "n_estimators":n_estimator,"random_state":0})
        config["estimator_parameters"].append({"n_fold":5,"type":"LGBMRegressor",
                                               "n_estimators":n_estimator,"random_state":1})
        config["estimator_parameters"].append({"n_fold":5,"type":"RandomForestRegressor",
                                               "n_estimators":n_estimator,"random_state":2,"max_features":"sqrt","n_jobs":-1})
        config["estimator_parameters"].append({"n_fold":5,"type":"RandomForestRegressor",
                                               "n_estimators":n_estimator,"random_state":3,"max_features":"sqrt","n_jobs":-1})

    config["valid_evaluation"]=R2
    config["random_state"] = 0
    config["max_layers"]=2
    return config

def MSE(y_true,y_pred):
    mse=mean_squared_error(y_true,y_pred)
    return mse

def RMSE(y_true,y_pred):
    return np.sqrt(MSE(y_true,y_pred))

def R2(y_true,y_pred):
    return r2_score(y_true,y_pred)

def Pearson(y_true,y_pred):
    return pearsonr(y_true,y_pred)[0]