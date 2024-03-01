import numpy as np
from logger import get_logger
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from deepforest import CascadeForestRegressor
from sklearn.svm import SVR
import pandas as pd
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import sklearn.pipeline as pl
import sklearn.preprocessing as sp
# from catboost import CatBoostRegressor
# from ngboost import NGBRegressor
# from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import BayesianRidge
# import AdaNet

import torch.nn as nn

method_type = 'RF'
def make_estimator(method='RF', n_estimators = 200, random_state = 0,predictors = 'forest'):
    if method == 'RF':
        return RandomForestRegressor(n_estimators = n_estimators, random_state= random_state)
    elif method == 'gcforest':
        return CascadeForestRegressor(n_estimators=6,random_state = random_state,max_layers =4) #n_estimators=6,n_trees=50,
    elif method == 'ET':
        return ExtraTreesRegressor(n_estimators = n_estimators, random_state= random_state)
    elif method == 'LGBMRegressor':
        return LGBMRegressor(n_estimators = n_estimators, random_state= random_state)
    elif method == 'XGBRegressor':
        return XGBRegressor(n_estimators = n_estimators, random_state= random_state)
    elif method == "GDBT":
        return GradientBoostingRegressor(n_estimators = n_estimators, random_state= random_state)
    elif method == 'LR':
        return LinearRegression()
    elif method == 'KNN':
        return KNeighborsRegressor(n_neighbors=5)
    elif method == 'SVR':
        return SVR()
    elif method == "BayesianRidge":
        return BayesianRidge()
    elif method == 'Poly':
        return pl.make_pipeline(sp.PolynomialFeatures(2),LinearRegression())
    elif method == 'DNN':
        return
    elif method == 'CNN':
        return
    elif method == 'CatB':
        return CatBoostRegressor(iterations=10, depth=16, learning_rate=0.8, loss_function='RMSE',random_seed=random_state)
    elif method == 'NGB':
        return NGBRegressor(n_estimators = 1000,
                            Base=DecisionTreeRegressor(random_state= random_state,max_depth=18),
                            random_state= random_state)
    elif method == 'TNR':
        return TabNetRegressor()
    elif method == 'DT':
        return LinearRegression()

def main():
    # list = ['xgboost','lightgbm']  #'forest',
    # for j in range(len(list)):
    LOGGER = get_logger("-", 'Sep-27-RF-view')
    LOGGER.info("===================={}====================".format(method_type))

    # datafile = "reprocessdata/Data15495_pca_600.csv"
    # datafile = "lianheyongyao.csv"
    datafile = 'reprocessdata/Data4116_f_1431.csv'
    # datafile = "reprocessdata/logec50_Data4116_kbest_1400.csv"
    # datafile = "reprocessdata/maxmin_Data4116_kbest_1400.csv"
    data=pd.read_csv(datafile, header=None)
    print(data.shape)
    data = np.array(data)
    scaler = MinMaxScaler(feature_range=(0,1))
    data = scaler.fit_transform(data)
    # x = data[:,630:-1]
    x = data[:, 0:-1]
    # x = np.hstack((data[:,0:400],data[:,1000:-1]))
    y=data[:,-1]
    mses,rmses,maes,r2s,pearsonrs,spearmanrs,y_preds={},{},{},{},{},{},{}
    # train & test spilt, using k-folds cross-validation
    kf=RepeatedKFold(n_splits=5, n_repeats=1, random_state=0)
    i = 0
    importance = range(0,1430)
    for train_id, test_id in kf.split(x, y):
        print("----------Kfold-{}----------".format(i+1))
        x_train, x_test = x[train_id], x[test_id]
        y_train, y_test = y[train_id], y[test_id]
        clf = make_estimator(method = method_type, n_estimators = 100, random_state = 0)

        #TNR需要reshape
        # y_train = y_train.reshape(-1,1)

        clf.fit(x_train, y_train)
        # feature_importance = clf.feature_importances_
        # rank_index = np.argsort(feature_importance[630:1030])
        # importance_index = rank_index[-100:]
        # importance = np.intersect1d(importance, importance_index)
        # print(len(importance))

        y_pred = clf.predict(x_test)

        # deepforest需要转换
        # y_pred = y_pred.T[0]

        # 结果统计
        mses[i] = mean_squared_error(y_test, y_pred)
        rmses[i] = np.sqrt(mses[i])
        maes[i] = mean_absolute_error(y_test, y_pred)
        r2s[i] = r2_score(y_test, y_pred)
        pearsonrs[i] = pearsonr(y_test, y_pred)[0]
        #TNR需要
        # pearsonrs[i] = pearsonr(y_test, y_pred)[0][0]
        spearmanrs[i] = spearmanr(y_test, y_pred)[0]


        print("MSE mean={:.4f}".format(mses[i]))
        print("RMSE mean={:.4f}".format(rmses[i]))
        print("MAE mean={:.4f}".format(maes[i]))
        print("R^2 mean={:.4f}".format(r2s[i]))

        print("PEARSONR mean={:.4f}".format(pearsonrs[i]))
        print("SPEARMANR mean={:.4f}".format(spearmanrs[i]))
        i = i + 1

    # print(len(importance))
    # print(importance)
    # print(feature_importance[630+importance])
    mses = sorted(mses.items(), key=lambda x: x[0])
    mses = [value[1] for value in mses]
    rmses = sorted(rmses.items(), key=lambda x: x[0])
    rmses = [value[1] for value in rmses]
    maes = sorted(maes.items(), key=lambda x: x[0])
    maes = [value[1] for value in maes]
    r2s = sorted(r2s.items(), key=lambda x: x[0])
    r2s = [value[1] for value in r2s]
    pearsonrs = sorted(pearsonrs.items(), key=lambda x: x[0])
    pearsonrs = [value[1] for value in pearsonrs]
    spearmanrs = sorted(spearmanrs.items(), key=lambda x: x[0])
    spearmanrs = [value[1] for value in spearmanrs]

    LOGGER.info("data shape: {}".format(x.shape))
    LOGGER.info("MSE mean={:.4f}, std={:.4f}".format(np.mean(mses),np.std(mses)))
    LOGGER.info("RMSE mean={:.4f}, std={:.4f}".format(np.mean(rmses),np.std(rmses)))
    LOGGER.info("MAE mean={:.4f}, std={:.4f}".format(np.mean(maes),np.std(maes)))
    LOGGER.info("R^2 mean={:.4f}, std={:.4f}".format(np.mean(r2s),np.std(r2s)))
    LOGGER.info("PEARSONR mean={:.4f}, std={:.4f}".format(np.mean(pearsonrs),np.std(pearsonrs)))
    LOGGER.info("SPEARMANR mean={:.4f}, std={:.4f}".format(np.mean(spearmanrs), np.std(spearmanrs)))

if __name__ == '__main__':
    main()