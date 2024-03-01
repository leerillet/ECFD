import numpy as np
import math
from layer import layer
from Forest_trainest import Forest_est
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from scipy.stats import pearsonr,spearmanr
from logger import get_logger

class MCgcF(object):
    def __init__(self, config, n):
        self.estimator_parameters = config["estimator_parameters"]  #每层
        self.random_state = config["random_state"]
        self.max_layers = config["max_layers"]
        self.valid_evaluation = config["valid_evaluation"]  #判断是否继续生长层数的指标 采用R2
        self.layers = []  #存深度森林的层

    def fit(self, x, y):
        deepth = 0
        x_train, y_train = x.copy(), y.copy()
        best_evaluation = 0.0  # R2
        while deepth<self.max_layers:
            forest_num = len(self.estimator_parameters)
            y_forest_preds = np.zeros((x_train.shape[0], forest_num))  #一层森林的预测结果，样本数*森林数
            enhanced_fea_num = 10
            enhanceds = np.zeros((x_train.shape[0], forest_num*enhanced_fea_num))
            current_layer = layer(deepth)
            for i in range(forest_num):  #遍历每层所有森林
                config = self.estimator_parameters[i].copy()
                estimator = Forest_est(config, self.random_state)
                y_forest_pred = estimator.fit(x_train, y_train)  #获得训练数据验证预测结果，用于计算评价指标
                enhanced = estimator.train_get_enhanced(x_train)
                current_layer.add_estimator(estimator)  #在层中添加训练的森林
                print(self.valid_evaluation(y, y_forest_pred))
                y_forest_preds[:, i] += y_forest_pred  #两倍视图量的增强向量
                enhanceds[:, i*enhanced_fea_num:(i+1)*enhanced_fea_num] += enhanced
            y_valid = np.mean(y_forest_preds, axis=1)  # 每个森林结果求平均值  当前层的结果
            Kth_layer_evaluation = self.valid_evaluation(y_train, y_valid)  #统计层的R2
            print("-------------当前层的layer{}.evaluation:{}------------".format(deepth,Kth_layer_evaluation))
            self.layers.append(current_layer)

            #  ==========x_train============= 训练到那层的训练集预测结果，查看是否需要生长
            y_pred = self.predict(x)  #对生长到现在的所有层进行预测
            used_layer_evaluation = self.valid_evaluation(y, y_pred)
            current_layer.pred = y_pred  #记录到当前所有层的预测结果
            print("------------当前所有层evaluation:{}-----------".format(used_layer_evaluation))

            if Kth_layer_evaluation > best_evaluation:  # R2越大越好 第二层R2小于第一层就停止
                best_evaluation = Kth_layer_evaluation
            else:
                print("达到最大层数")
                print("stop layer:{}".format(deepth))
                self.layers = self.layers[0:-1]
                break

            # y_proba=np.sort(y_proba,axis=1)  #预测值进行排序再拼接
            # y_proba = np.mean(y_proba,axis=1).reshape(-1,1)  #预测向量取均值再拼接
            # x_train = np.hstack((x_train, y_forest_preds,enhanceds))  # y_proba 为森林预测向量 是否需要添加加权后的向量y_valid
            # x_train = np.hstack((x_train, y_forest_preds))
            x_train = np.hstack((x_train, enhanceds))
            deepth += 1

    def predict(self, x):
        x_test = x.copy()
        y_preds = np.zeros((x_test.shape[0], len(self.layers)))
        for i in range(len(self.layers)):
            x_test_pred = self.layers[i].predict(x_test)
            x_test_enhanced = self.layers[i].get_enhanced(x_test)  #获取增强向量
            y_pred = np.mean(x_test_pred, axis=1)
            # x_test = np.hstack((x_test, x_test_pred,x_test_enhanced))
            # x_test = np.hstack((x_test, x_test_pred))
            x_test = np.hstack((x_test, x_test_enhanced))
            y_preds[:, i] += y_pred
        return np.mean(y_preds, axis=1)

