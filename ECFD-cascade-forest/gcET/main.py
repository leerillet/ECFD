from logger import get_logger
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from parallel_kfold import *

def read_data(filename):
    data = pd.read_csv(filename, header=None)  #kbest
    data = np.array(data)
    print(data.shape)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)
    x = data[:, 0:-1]
    # x = np.hstack((x[:,0:730],x[:,1029:-1]))
    y = data[:, -1]
    return x,y

def main():
    # view = ['CNV','exp','Meth','RNA','CNV+exp','CNV+Meth','CNV+RNA',
    #         'exp+Meth','exp+RNA','Meth+RNA','CNV+exp+Meth','CNV+exp+RNA',
    #         'exp+Meth+RNA','CNV+exp+Meth+RNA']  #多组学视图拼接类型
    file = 'reprocessdata/Data15459_f_1031.csv'
    LOGGER = get_logger("-", '0923-gcXGBET-all')
    forests_type = ['ETET'] #深度森林森林组成类型  改动森林数量需要改两个文件里的//2
    n_estimators = [50]  # 50 森林决策树棵树
    for i in range(len(forests_type)):
        x,y = read_data(filename=file)
        for k in range(len(n_estimators)):
            LOGGER.info("====================MCgcF-{}-{}-n-estimators:{}-layer:4====================".format(forests_type[i],file,n_estimators[k]))
            rmses,r2s,pearsonrs,spearmanrs,y_tests,y_preds=par_kfold(x,y,n_spilt=5,n_repeat=1,n_jobs=5,  #多进程Kfold
                        random_state=0,forests_type = forests_type[i],n_estimator=n_estimators[k])

            print("R^2 mean={:.4f}, std={:.4f}".format(np.mean(r2s),np.std(r2s)))
            print("PEARSONR mean={:.4f}, std={:.4f}".format(np.mean(pearsonrs),np.std(pearsonrs)))
            print("RMSE mean={:.4f}, std={:.4f}".format(np.mean(rmses),np.std(rmses)))
            print("SPEARMANR mean={:.4f}, std={:.4f}".format(np.mean(spearmanrs),np.std(spearmanrs)))

            LOGGER.info("data shape: {}".format(x.shape))
            LOGGER.info("R^2 mean={:.4f}, std={:.4f}".format(np.mean(r2s),np.std(r2s)))
            LOGGER.info("PEARSONR mean={:.4f}, std={:.4f}".format(np.mean(pearsonrs),np.std(pearsonrs)))
            LOGGER.info("RMSE mean={:.4f}, std={:.4f}".format(np.mean(rmses),np.std(rmses)))
            LOGGER.info("SPEARMANR mean={:.4f}, std={:.4f}".format(np.mean(spearmanrs), np.std(spearmanrs)))

if __name__ == '__main__':
    main()