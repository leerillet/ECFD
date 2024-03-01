from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor,AdaBoostRegressor
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


class Forest_est(object):
    def __init__(self,layer_id,index,config, random_state):
        self.config = config
        self.layer_id = layer_id
        self.index = index
        self.name="layer_{}, estimstor_{}, {}".format(layer_id,index,self.config["type"])
        if random_state is not None:
            self.random_state=(random_state+hash(self.name))%1000000007
        else:
            self.random_state=None
        self.forest_type = self.config["type"]
        self.n_fold=self.config["n_fold"]
        self.estimators=[None for i in range(self.config["n_fold"])]
        self.config.pop("n_fold")
        self.estimator_class = globals()[self.config["type"]]
        self.estimator_type = self.config["type"]
        self.config.pop("type")
        self.important_fea_idex_selected = []
        self.hh_fea_idex_selected = []
        self.fea_id = []
        self.train_predict = []
        self.household_matrix = []
        self.important_fea_selected_num = 0.65
        self.hard_index = []
        self.easy_index = []

    def _init_estimator(self):
        estimator_args = self.config
        est_args = estimator_args.copy()
        est_args["random_state"] = self.random_state
        return self.estimator_class(**est_args)

    def train_get_enhanced(self,x):
        extractor = PCA(n_components=1)
        extractor.fit(x[:,self.hh_fea_idex_selected])
        mu = extractor.components_[0]

        I = np.diag(np.ones(len(self.hh_fea_idex_selected)))
        check_ = np.sqrt(((I - mu) ** 2).sum(axis=1))
        i = np.argmax(check_)
        e = np.zeros(len(self.hh_fea_idex_selected))
        e[i] = 1.0
        w = (e - mu) / norm(e - mu)
        householder_matrix = I - 2 * w[:, np.newaxis].dot(w[:, np.newaxis].T)
        self.household_matrix = householder_matrix
        X_house = x[:,self.hh_fea_idex_selected].dot(householder_matrix)
        enhanced = X_house
        return enhanced

    def test_get_enhanced(self,x):
        X_house = x[:,self.hh_fea_idex_selected].dot(self.household_matrix)
        enhanced = X_house
        return enhanced

    def fit(self, x, y,sample_weight):
        kf = RepeatedKFold(n_splits=self.n_fold, n_repeats=1, random_state=self.random_state)
        cv = [(t, v) for (t, v) in kf.split(x)]
        y_train_pred = np.zeros((x.shape[0],))
        est_fea_importance = np.zeros((x.shape[1]))
        est_fea_shap = np.zeros((x.shape[1]))
        for k in range(len(self.estimators)):
            est = self._init_estimator()
            train_id, val_id = cv[k]
            x_train, y_train, sample_weight_train = x[train_id], y[train_id],sample_weight[train_id]
            est.fit(x_train, y_train,sample_weight = sample_weight_train)
            est_fea_importance += est.feature_importances_
            y_pred = est.predict(x[val_id])
            y_train_pred[val_id] += y_pred
            self.estimators[k] = est

        if self.layer_id==0:
            feature_num = x.shape[1]
        else:
            feature_num = x.shape[1]-4

        num = int(self.important_fea_selected_num*feature_num)
        est_fea_index_all = np.argsort(est_fea_importance[:feature_num])
        important_fea_idex_selected = est_fea_index_all[-num:]
        self.important_fea_idex_selected = important_fea_idex_selected
        self.hh_fea_idex_selected = important_fea_idex_selected[-200:]

        if self.layer_id==0:
            est_omic_index_all = np.argsort(est_fea_importance[2278:4278])
            self.fea_id = est_omic_index_all[-10:]

        self.train_predict = y_train_pred
        self.hard_index = [i for i in range(len(y)) if abs(y_train_pred[i]-y[i])>0.05]
        self.easy_index = [i for i in range(len(y)) if abs(y_train_pred[i]-y[i])<=0.05]

    def predict(self, x):
        pre_value=0
        for est in self.estimators:
            pre_value+=est.predict(x)
        pre_value/=len(self.estimators)
        return pre_value

