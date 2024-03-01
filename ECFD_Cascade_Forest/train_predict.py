from logger import get_logger
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from config import *
from sklearn.model_selection import RepeatedKFold,train_test_split,KFold,RepeatedStratifiedKFold
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error,mean_absolute_percentage_error
from scipy.stats import pearsonr, spearmanr
from src.DF import DF
from src.load_data import load_data

def main():
    LOGGER = get_logger("-", 'ECFD')
    forests_type = 'LGBMRF' #深度森林森林组成类型  改动森林数量需要改两个文件里的//2
    n_estimators = 200  # 50 森林决策树棵树
    label_file = "data/label/all_label_29931.csv"
    method='f'
    x, y = load_data(label_file, method)

    LOGGER.info("========Omics-DF-{}-{}-n-estimators:{}-layer:4=========".format(forests_type,label_file,n_estimators))
    mses,rmses,maes,r2s,pearsonrs,spearmanrs,fea_ids=[],[],[],[],[],[],[]
    config = get_config(forests_type, n_estimators)  #获取多通道深度森林参数
    kf=RepeatedKFold(n_splits=5,n_repeats=1,random_state=0)
    cv=[(t,v) for (t,v) in kf.split(x)]

    # label_file = "data/label/all_label_29931.csv"
    # Label = pd.read_csv(label_file, index_col=False, sep='\t')
    # Label = np.array(Label)

    # # drug blind test
    # drug = Label[:, 0]
    # drug_unique = np.unique(drug)
    # kf = RepeatedKFold(n_splits=5, n_repeats=1, random_state=0)
    # cv = [(t, v) for (t, v) in kf.split(drug_unique)]
    # blind_drug_cv = []
    # for k in range(5):
    #     (train_drug, test_drug) = cv[k]
    #     train_index = []
    #     test_index = []
    #     for i in train_drug:
    #         for j in range(len(drug)):
    #             if drug[j] == drug_unique[i]:
    #                 train_index.append(j)
    #     for i in range(len(drug)):
    #         if i not in train_index:
    #             test_index.append(i)
    #     blind_drug_cv.append((train_index, test_index))
    # cv = blind_drug_cv

    # #cellline blind test
    # cellline = Label[:, 1]
    # cellline_unique = np.unique(cellline)
    # kf = RepeatedKFold(n_splits=5, n_repeats=1, random_state=0)
    # cv = [(t, v) for (t, v) in kf.split(cellline_unique)]
    # blind_cellline_cv = []
    # for k in range(5):
    #     (train_cellline, test_cellline) = cv[k]
    #     train_index = []
    #     test_index = []
    #     for i in train_cellline:
    #         for j in range(len(cellline)):
    #             if cellline[j] == cellline_unique[i]:
    #                 train_index.append(j)
    #     for i in range(len(cellline)):
    #         if i not in train_index:
    #             test_index.append(i)
    #     blind_cellline_cv.append((train_index,test_index))
    # cv = blind_cellline_cv

    for i in range(5):
        (train_id,test_id)=cv[i]
        x_train, x_test, y_train, y_test=x[train_id],x[test_id],y[train_id],y[test_id]
        model=DF(config)
        model.fit(x_train,y_train)
        y_pred=model.predict(x_test,y_test)
        fea_id = model.fea_id

        mses.append(mean_squared_error(y_test,y_pred))
        rmses.append(np.sqrt(mses[i]))
        r2s.append(r2_score(y_test,y_pred))
        pearsonrs.append(pearsonr(y_test,y_pred)[0])
        spearmanrs.append(spearmanr(y_test, y_pred)[0])
        fea_ids = list(set(fea_ids).union(set(fea_id)))

    LOGGER.info("data shape: {}".format(x.shape))
    LOGGER.info("MSE mean={:.4f}±{:.4f}".format(np.mean(mses), np.std(mses)))
    LOGGER.info("RMSE mean={:.4f}±{:.4f}".format(np.mean(rmses), np.std(rmses)))
    LOGGER.info("R^2 mean={:.4f}±{:.4f}".format(np.mean(r2s), np.std(r2s)))
    LOGGER.info("PEARSONR mean={:.4f}±{:.4f}".format(np.mean(pearsonrs), np.std(pearsonrs)))
    LOGGER.info("SPEARMANR mean={:.4f}±{:.4f}".format(np.mean(spearmanrs), np.std(spearmanrs)))


if __name__ == '__main__':
    main()