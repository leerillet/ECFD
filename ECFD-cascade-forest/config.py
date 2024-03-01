from sklearn.metrics import mean_squared_error,r2_score
from scipy.stats import pearsonr
import numpy as np

def get_config(forests_type,n_estimator):
    config={}
    config["estimator_parameters"]=[]
    if forests_type == 'RFET':
        config["estimator_parameters"].append({"n_fold":5,"type":"RandomForestRegressor",
                                            "n_estimators":n_estimator,"random_state":0,"max_features":"sqrt"})
        config["estimator_parameters"].append({"n_fold":5,"type":"ExtraTreesRegressor",
                                            "n_estimators":n_estimator,"random_state":1,"max_features":"sqrt"})
        config["estimator_parameters"].append({"n_fold":5,"type":"RandomForestRegressor",
                                            "n_estimators":n_estimator,"random_state":2,"max_features":"sqrt"})
        config["estimator_parameters"].append({"n_fold":5,"type":"ExtraTreesRegressor",
                                            "n_estimators":n_estimator,"random_state":3,"max_features":"sqrt"})
        config["estimator_parameters"].append({"n_fold":5,"type":"RandomForestRegressor",
                                            "n_estimators":n_estimator,"random_state":4,"max_features":"sqrt"})
        config["estimator_parameters"].append({"n_fold":5,"type":"ExtraTreesRegressor",
                                            "n_estimators":n_estimator,"random_state":5,"max_features":"sqrt"})
        config["estimator_parameters"].append({"n_fold":5,"type":"RandomForestRegressor",
                                            "n_estimators":n_estimator,"random_state":6,"max_features":"sqrt"})
        config["estimator_parameters"].append({"n_fold":5,"type":"ExtraTreesRegressor",
                                            "n_estimators":n_estimator,"random_state":7,"max_features":"sqrt"})
    if forests_type == 'ETET':
        config["estimator_parameters"].append({"n_fold":5,"type":"ExtraTreesRegressor",
                                            "n_estimators":n_estimator,"random_state":0,"max_features":"sqrt"})
        config["estimator_parameters"].append({"n_fold":5,"type":"ExtraTreesRegressor",
                                            "n_estimators":n_estimator,"random_state":1,"max_features":"sqrt"})
        config["estimator_parameters"].append({"n_fold":5,"type":"ExtraTreesRegressor",
                                            "n_estimators":n_estimator,"random_state":2,"max_features":"sqrt"})
        config["estimator_parameters"].append({"n_fold":5,"type":"ExtraTreesRegressor",
                                            "n_estimators":n_estimator,"random_state":3,"max_features":"sqrt"})
        config["estimator_parameters"].append({"n_fold":5,"type":"ExtraTreesRegressor",
                                            "n_estimators":n_estimator,"random_state":4,"max_features":"sqrt"})
        config["estimator_parameters"].append({"n_fold":5,"type":"ExtraTreesRegressor",
                                            "n_estimators":n_estimator,"random_state":5,"max_features":"sqrt"})
        config["estimator_parameters"].append({"n_fold":5,"type":"ExtraTreesRegressor",
                                            "n_estimators":n_estimator,"random_state":6,"max_features":"sqrt"})
        config["estimator_parameters"].append({"n_fold":5,"type":"ExtraTreesRegressor",
                                            "n_estimators":n_estimator,"random_state":7,"max_features":"sqrt"})

    elif forests_type == 'XGBET':
        config["estimator_parameters"].append({"n_fold":5,"type":"XGBRegressor",
                                            "n_estimators":n_estimator,"random_state":0})
        config["estimator_parameters"].append({"n_fold":5,"type":"ExtraTreesRegressor",
                                            "n_estimators":n_estimator,"random_state":1,"max_features":"sqrt"})
        config["estimator_parameters"].append({"n_fold":5,"type":"XGBRegressor",
                                            "n_estimators":n_estimator,"random_state":2})
        config["estimator_parameters"].append({"n_fold":5,"type":"ExtraTreesRegressor",
                                            "n_estimators":n_estimator,"random_state":3,"max_features":"sqrt"})
        config["estimator_parameters"].append({"n_fold":5,"type":"XGBRegressor",
                                            "n_estimators":n_estimator,"random_state":4})
        config["estimator_parameters"].append({"n_fold":5,"type":"ExtraTreesRegressor",
                                            "n_estimators":n_estimator,"random_state":5,"max_features":"sqrt"})
        config["estimator_parameters"].append({"n_fold":5,"type":"XGBRegressor",
                                            "n_estimators":n_estimator,"random_state":6})
        config["estimator_parameters"].append({"n_fold":5,"type":"ExtraTreesRegressor",
                                            "n_estimators":n_estimator,"random_state":7,"max_features":"sqrt"})

    elif forests_type == 'LGBMET':
        config["estimator_parameters"].append({"n_fold":5,"type":"LGBMRegressor",
                                            "n_estimators":n_estimator,"random_state":0})
        config["estimator_parameters"].append({"n_fold":5,"type":"ExtraTreesRegressor",
                                            "n_estimators":n_estimator,"random_state":1,"max_features":"sqrt"})
        config["estimator_parameters"].append({"n_fold":5,"type":"LGBMRegressor",
                                            "n_estimators":n_estimator,"random_state":2})
        config["estimator_parameters"].append({"n_fold":5,"type":"ExtraTreesRegressor",
                                            "n_estimators":n_estimator,"random_state":3,"max_features":"sqrt"})
        config["estimator_parameters"].append({"n_fold":5,"type":"LGBMRegressor",
                                            "n_estimators":n_estimator,"random_state":4})
        config["estimator_parameters"].append({"n_fold":5,"type":"ExtraTreesRegressor",
                                            "n_estimators":n_estimator,"random_state":5,"max_features":"sqrt"})
        config["estimator_parameters"].append({"n_fold":5,"type":"LGBMRegressor",
                                            "n_estimators":n_estimator,"random_state":6})
        config["estimator_parameters"].append({"n_fold":5,"type":"ExtraTreesRegressor",
                                            "n_estimators":n_estimator,"random_state":7,"max_features":"sqrt"})
    elif forests_type == 'XGBLGBM':
        config["estimator_parameters"].append({"n_fold":5,"type":"XGBRegressor",
                                            "n_estimators":n_estimator,"random_state":0})
        config["estimator_parameters"].append({"n_fold":5,"type":"LGBMRegressor",
                                            "n_estimators":n_estimator,"random_state":1})
        config["estimator_parameters"].append({"n_fold":5,"type":"XGBRegressor",
                                            "n_estimators":n_estimator,"random_state":2})
        config["estimator_parameters"].append({"n_fold":5,"type":"LGBMRegressor",
                                            "n_estimators":n_estimator,"random_state":3})
        config["estimator_parameters"].append({"n_fold":5,"type":"XGBRegressor",
                                            "n_estimators":n_estimator,"random_state":4})
        config["estimator_parameters"].append({"n_fold":5,"type":"LGBMRegressor",
                                            "n_estimators":n_estimator,"random_state":5})
        config["estimator_parameters"].append({"n_fold":5,"type":"XGBRegressor",
                                            "n_estimators":n_estimator,"random_state":6})
        config["estimator_parameters"].append({"n_fold":5,"type":"LGBMRegressor",
                                            "n_estimators":n_estimator,"random_state":7})
    elif forests_type == 'RF-ET-XGB-LGBM-GDBT':
        config["estimator_parameters"].append({"n_fold":5,"type":"RandomForestRegressor",
                                            "n_estimators":n_estimator,"random_state":0,"max_features":"sqrt"})
        config["estimator_parameters"].append({"n_fold":5,"type":"ExtraTreesRegressor",
                                            "n_estimators":n_estimator,"random_state":0,"max_features":"sqrt"})
        config["estimator_parameters"].append({"n_fold":5,"type":"XGBRegressor",
                                            "n_estimators":n_estimator,"random_state":0})
        config["estimator_parameters"].append({"n_fold":5,"type":"LGBMRegressor",
                                            "n_estimators":n_estimator,"random_state":0})
        config["estimator_parameters"].append({"n_fold":5,"type":"GradientBoostingRegressor",
                                            "n_estimators":n_estimator,"random_state":0})

    config["valid_evaluation"]=R2
    config["random_state"] = 0
    config["max_layers"]=4
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