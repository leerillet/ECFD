from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
import numpy as np
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.decomposition import PCA
from scipy.linalg import norm
np.set_printoptions(threshold=np.inf)
import random

class Forest_est(object):
    def __init__(self,config, random_state):
        self.config = config
        self.n_fold = self.config["n_fold"] #折数
        self.estimators = [None for i in range(self.config["n_fold"])]  # 有几折就有几个森林estimators
        self.config.pop("n_fold")
        self.estimator_class = globals()[self.config["type"]]
        self.config.pop("type")
        self.important_fea_idex_selected = []
        self.important_fea_idex = []
        self.predct = []
        self.household_matrix = []
        self.mu = []
        self.extractor = None
        self.important_fea_selected_num = 10

    def _init_estimator(self):
        estimator_args = self.config
        est_args = estimator_args.copy()
        return self.estimator_class(**est_args)

    def train_get_enhanced(self,x):   #这里做修改
        # enhanced = np.zeros((x.shape[0],5))
        # enhanced[:,0] = np.mean(x[:,self.important_fea_idex_selected],axis=1)
        # enhanced[:,1] = np.std(x[:,self.important_fea_idex_selected],axis=1)
        # enhanced[:,2] = np.var(x[:,self.important_fea_idex_selected],axis=1)

        #HouseHolder变换 需要获得共同的重要特征挑选过的固定数量特征
        extractor = PCA(n_components=1)
        extractor.fit(x[:,self.important_fea_idex_selected], self.predct)
        mu = extractor.components_[0]
        self.mu = mu

        I = np.diag(np.ones(len(self.important_fea_idex_selected)))
        check_ = np.sqrt(((I - mu) ** 2).sum(axis=1))
        i = np.argmax(check_)
        e = np.zeros(len(self.important_fea_idex_selected))
        e[i] = 1.0
        w = (e - mu) / norm(e - mu)
        householder_matrix = I - 2 * w[:, np.newaxis].dot(w[:, np.newaxis].T)
        self.household_matrix = householder_matrix
        X_house = x[:,self.important_fea_idex_selected].dot(householder_matrix)
        enhanced = X_house

        # #PCA变换 只需要获得共同重要特征
        # extractor = PCA(n_components=10)
        # extractor.fit(x[:,self.important_fea_idex], self.predct)
        # enhanced = extractor.transform(x[:,self.important_fea_idex])
        # self.extractor = extractor

        # 直接使用挑选后的重要特征，不进行变换
        # enhanced = x[:,self.important_fea_idex_selected]

        # enhanced[:,0] = np.mean(X_house,axis=1)
        # enhanced[:,1] = np.std(X_house,axis=1)
        # enhanced[:,2] = np.var(X_house,axis=1)

        return enhanced

    def test_get_enhanced(self,x):   #这里做修改
        # enhanced = np.zeros((x.shape[0],5))  #增强特征维度
        # enhanced[:,0] = np.mean(x[:,self.important_fea_idex],axis=1)
        # enhanced[:,1] = np.std(x[:,self.important_fea_idex],axis=1)
        # enhanced[:,2] = np.var(x[:,self.important_fea_idex],axis=1)

        #使用HouseHolder变换的获取增强向量方法
        I = np.diag(np.ones(len(self.important_fea_idex_selected)))
        check_ = np.sqrt(((I - self.mu) ** 2).sum(axis=1))
        i = np.argmax(check_)
        e = np.zeros(len(self.important_fea_idex_selected))
        e[i] = 1.0
        X_house = x[:,self.important_fea_idex_selected].dot(self.household_matrix)
        enhanced = X_house

        #使用PCA获取增强向量方法
        # enhanced = self.extractor.transform(x[:, self.important_fea_idex])

        #直接使用挑选后的重要特征，不进行变换
        # enhanced = x[:,self.important_fea_idex_selected]

        # enhanced[:,0] = np.mean(X_house,axis=1)
        # enhanced[:,1] = np.std(X_house,axis=1)
        # enhanced[:,2] = np.var(X_house,axis=1)

        return enhanced

    def fit(self, x, y):
        kf = RepeatedKFold(n_splits=self.n_fold, n_repeats=1, random_state=0)
        cv = [(t, v) for (t, v) in kf.split(x)]
        self.estimators = [None for i in range(len(cv))]
        y_train_pred = np.zeros((x.shape[0],))
        last_fea_importance = range(x.shape[1])

        for k in range(len(cv)):
            train_id, val_id = cv[k]
            x_train, y_train = x[train_id], y[train_id]
            est = self._init_estimator()
            est.fit(x_train, y_train)

            print("----------------{}--------------------".format(k+1))
            est_fea_importance = est.feature_importances_
            est_fea_importance = est_fea_importance[:1030]  #确保每次选取的是原数据的
            # est_fea_importance = est_fea_importance[:1430]  # 确保每次选取的是原数据的
            est_fea_index = np.argsort(est_fea_importance)
            importance_index = est_fea_index[-200:]
            last_fea_importance = np.intersect1d(last_fea_importance, importance_index)
            if k == 4:
                print(last_fea_importance)
                print(len(last_fea_importance))

            y_pred = est.predict(x[val_id])
            y_train_pred[val_id] += y_pred
            self.estimators[k] = est

        num = self.important_fea_selected_num  #当需要对重要特征筛选时，设置筛选特征数量
        #对共同重要特征进行Pearson相关性排序
        pearson_ = [pearsonr(y_train_pred, x[:,i])[0] for i in last_fea_importance]
        pearson_index = np.argsort(pearson_)
        important_fea_idex_selected = last_fea_importance[pearson_index[-num:]]  #pearson_长度为len(last_fea_importance)
        #从共同重要特征中随机抽取特征
        # last_fea_importance = last_fea_importance.tolist()
        # important_fea_idex_selected = random.sample(last_fea_importance, num)
        print(important_fea_idex_selected)

        self.important_fea_idex = last_fea_importance #保存共同重要特征
        self.important_fea_idex_selected = important_fea_idex_selected #保存由重要特征挑选后的特征
        self.predct = y_train_pred
        return y_train_pred

    def predict(self, x):
        pre_value = 0
        for est in self.estimators:
            pre_value += est.predict(x)
        pre_value /= len(self.estimators)
        return pre_value

