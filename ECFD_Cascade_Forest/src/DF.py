import numpy as np
import math
from src.layer import layer
from src.Forest_trainest import Forest_est
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from scipy.stats import pearsonr,spearmanr
import random

class DF(object):
    def __init__(self, config):
        self.estimator_parameters = config["estimator_parameters"]
        self.random_state = config["random_state"]
        self.max_layers = config["max_layers"]
        self.valid_evaluation = config["valid_evaluation"]
        self.layers = []
        self.important_fea_selected = []
        self.fea_id = []

    def fit(self, x_train, y_train):
        deepth = 0
        x_train, y_train = x_train.copy(), y_train.copy()
        n_feature = x_train.shape[1]
        x_train_0,x_train_1,x_train_2,x_train_3= x_train,x_train,x_train,x_train
        hard_index_all = None
        layer_predicts = np.zeros((y_train.shape[0],self.max_layers))
        best_evaluation = 0.0
        y_preds_all = np.zeros((y_train.shape[0], self.max_layers * 4 + 2))

        sample_weight_0 = np.ones(len(y_train))
        sample_weight_1 = np.ones(len(y_train))
        sample_weight_2 = np.ones(len(y_train))
        sample_weight_3 = np.ones(len(y_train))
        while deepth<self.max_layers:
            forest_num = len(self.estimator_parameters)
            y_preds = np.zeros((y_train.shape[0], forest_num))
            y_folds_preds = np.zeros((y_train.shape[0], forest_num))
            current_layer = layer(deepth)

            config = self.estimator_parameters[0].copy()
            estimator = Forest_est(deepth,0,config, self.random_state)
            estimator.fit(x_train_0, y_train, sample_weight_0)
            print("-----------第1个森林的预测输出------------")
            print("训练集shape:{}".format(x_train_0.shape))
            print("训练集使用测试进行预测：")
            y_folds_preds[:,0] += estimator.predict(x_train_0)
            print(r2_score(y_train, y_folds_preds[:,0]))
            hard_index_0 = estimator.hard_index
            easy_index_0 = estimator.easy_index
            print("困难样本信息：")
            print(len(hard_index_0))
            enhanced_train_0 = estimator.train_get_enhanced(x_train_0)
            y_pred = estimator.train_predict
            print("------第{}层第1个森林的训练集预测r2：{:.4f}--------\n".format(deepth+1,r2_score(y_train,y_pred)))
            current_layer.add_estimator(estimator)
            self.fea_id.extend(estimator.fea_id)
            y_preds[:,0] += y_pred
            x_train_0_index = estimator.important_fea_idex_selected

            config = self.estimator_parameters[1].copy()
            estimator = Forest_est(deepth,1,config, self.random_state)
            estimator.fit(x_train_1, y_train, sample_weight_1)
            print("-----------第2个森林的预测输出------------")
            print("训练集shape:{}".format(x_train_1.shape))
            print("训练集使用测试进行预测：")
            y_folds_preds[:,1] += estimator.predict(x_train_1)
            print(r2_score(y_train, y_folds_preds[:,1]))
            hard_index_1 = estimator.hard_index
            easy_index_1 = estimator.easy_index
            print("困难样本信息：")
            print(len(hard_index_1))
            enhanced_train_1 = estimator.train_get_enhanced(x_train_1)
            y_pred = estimator.train_predict
            print("------第{}层第2个森林的训练集预测r2：{:.4f}--------\n".format(deepth+1,r2_score(y_train,y_pred)))
            current_layer.add_estimator(estimator)
            self.fea_id.extend(estimator.fea_id)
            y_preds[:, 1] += y_pred
            x_train_1_index = estimator.important_fea_idex_selected

            config = self.estimator_parameters[2].copy()
            estimator = Forest_est(deepth,2,config, self.random_state)
            estimator.fit(x_train_2, y_train, sample_weight_2)
            print("-----------第3个森林的预测输出------------")
            print("训练集shape:{}".format(x_train_2.shape))
            print("训练集使用测试进行预测：")
            y_folds_preds[:,2] += estimator.predict(x_train_2)
            print(r2_score(y_train, y_folds_preds[:,2]))
            hard_index_2 = estimator.hard_index
            easy_index_2 = estimator.easy_index
            print("困难样本信息：")
            print(len(hard_index_2))
            enhanced_train_2 = estimator.train_get_enhanced(x_train_2)
            y_pred = estimator.train_predict
            print("------第{}层第3个森林的训练集预测r2：{:.4f}--------\n".format(deepth+1,r2_score(y_train,y_pred)))
            current_layer.add_estimator(estimator)
            self.fea_id.extend(estimator.fea_id)
            y_preds[:, 2] += y_pred
            x_train_2_index = estimator.important_fea_idex_selected

            config = self.estimator_parameters[3].copy()
            estimator = Forest_est(deepth,3,config, self.random_state)
            estimator.fit(x_train_3, y_train, sample_weight_3)
            print("-----------第4个森林的预测输出------------")
            print("训练集shape:{}".format(x_train_3.shape))
            print("训练集使用测试进行预测：")
            y_folds_preds[:,3] += estimator.predict(x_train_3)
            print(r2_score(y_train, y_folds_preds[:,3]))
            hard_index_3 = estimator.hard_index
            easy_index_3 = estimator.easy_index
            print("困难样本信息：")
            print(len(hard_index_3))
            enhanced_train_3 = estimator.train_get_enhanced(x_train_3)
            y_pred = estimator.train_predict
            print("------第{}层第4个森林的训练集预测r2：{:.4f}--------\n".format(deepth+1,r2_score(y_train,y_pred)))
            current_layer.add_estimator(estimator)
            self.fea_id.extend(estimator.fea_id)
            y_preds[:, 3] += y_pred
            x_train_3_index = estimator.important_fea_idex_selected

            y_layer_pred = np.mean(y_preds, axis=1)

            if deepth ==0:
                hard_index = [i for i in range(len(y_train)) if abs(y_train[i]-y_layer_pred[i])>0.01]
                print(len(hard_index))
                hard_index_all = hard_index
            else:
                tmp = [i for i in range(len(y_train)) if abs(y_train[i]-y_layer_pred[i])>0.01]
                hard_index_all = tmp
                print(len(tmp))
                print(len(set(hard_index).intersection(set(tmp))))

            y_folds_pred = np.mean(y_folds_preds,axis=1)
            Kth_layer_evaluation = r2_score(y_train, y_layer_pred)
            Kth_layer_folds_evaluation = r2_score(y_train, y_folds_pred)
            layer_predicts[:, deepth] = y_layer_pred
            layers_train_evaluation = r2_score(y_train, np.mean(layer_predicts[:,0:deepth+1],axis=1))
            print("-------------当前层的layer{}.evaluation:{:.4f}------------".format(deepth+1,Kth_layer_evaluation))
            print("-------------当前层folds的layer{}.evaluation:{:.4f}------------".format(deepth + 1, Kth_layer_folds_evaluation))
            print("-------------当前所有层的layer.evaluation:{:.4f}------------\n".format(layers_train_evaluation))
            self.layers.append(current_layer)

            # if Kth_layer_evaluation > best_evaluation:
            #     best_evaluation = Kth_layer_evaluation
            # # if layers_train_evaluation > best_evaluation:
            # #     best_evaluation = layers_train_evaluation
            if Kth_layer_folds_evaluation > best_evaluation:
                best_evaluation = Kth_layer_folds_evaluation
            else:
                print("达到最大层数")
                print("stop layer:{}".format(deepth+1))
                self.layers = self.layers[0:-1]
                break

            if deepth+1<self.max_layers:
                sample_weight_0 = self.sample_weight_cal(x_train_0, y_train, y_preds[:, 0], sample_weight_0, hard_index_0,easy_index_0,deepth+1)
                sample_weight_1 = self.sample_weight_cal(x_train_1, y_train, y_preds[:, 1], sample_weight_1, hard_index_1,easy_index_1,deepth+1)
                sample_weight_2 = self.sample_weight_cal(x_train_2, y_train, y_preds[:, 2], sample_weight_2, hard_index_2,easy_index_2,deepth+1)
                sample_weight_3 = self.sample_weight_cal(x_train_3, y_train, y_preds[:, 3], sample_weight_3, hard_index_3,easy_index_3,deepth+1)
                #拼接预测向量
                x_train_0 = np.hstack((x_train_0[:,x_train_0_index],enhanced_train_0, y_preds))
                x_train_1 = np.hstack((x_train_1[:,x_train_1_index],enhanced_train_1, y_preds))
                x_train_2 = np.hstack((x_train_2[:,x_train_2_index],enhanced_train_2, y_preds))
                x_train_3 = np.hstack((x_train_3[:,x_train_3_index],enhanced_train_3, y_preds))

            deepth += 1
        print(self.fea_id)

    def predict(self, x,y):
        x_test = x.copy()
        n_feature = x_test.shape[1]
        y_layer_preds = np.zeros((y.shape[0], len(self.layers)))
        x_test_0,x_test_1,x_test_2,x_test_3= x_test,x_test,x_test,x_test
        print("-------------------test----------------------")
        for i in range(len(self.layers)):
            y_preds = np.zeros((y.shape[0], len(self.layers[i].estimators)))
            print("--------{}--------".format(i))

            x_test_pred = self.layers[i].estimators[0].predict(x_test_0)
            x_test_enhanced_0 = self.layers[i].estimators[0].test_get_enhanced(x_test_0)
            y_preds[:, 0] += x_test_pred
            print(r2_score(y,x_test_pred))

            x_test_pred = self.layers[i].estimators[1].predict(x_test_1)
            x_test_enhanced_1 = self.layers[i].estimators[1].test_get_enhanced(x_test_1)
            y_preds[:, 1] += x_test_pred
            print(r2_score(y, x_test_pred))

            x_test_pred = self.layers[i].estimators[2].predict(x_test_2)
            x_test_enhanced_2 = self.layers[i].estimators[2].test_get_enhanced(x_test_2)
            y_preds[:, 2] += x_test_pred
            print(r2_score(y, x_test_pred))

            x_test_pred = self.layers[i].estimators[3].predict(x_test_3)
            x_test_enhanced_3 = self.layers[i].estimators[3].test_get_enhanced(x_test_3)
            y_preds[:, 3] += x_test_pred
            print(r2_score(y, x_test_pred))

            y_pred = np.mean(y_preds, axis=1)
            x_test_0 = np.hstack((x_test_0[:,self.layers[i].estimators[0].important_fea_idex_selected],x_test_enhanced_0, y_preds))
            x_test_1 = np.hstack((x_test_1[:,self.layers[i].estimators[1].important_fea_idex_selected],x_test_enhanced_1, y_preds))
            x_test_2 = np.hstack((x_test_2[:,self.layers[i].estimators[2].important_fea_idex_selected],x_test_enhanced_2, y_preds))
            x_test_3 = np.hstack((x_test_3[:,self.layers[i].estimators[3].important_fea_idex_selected],x_test_enhanced_3, y_preds))

            print("test------当前层r2:{}".format(r2_score(y,y_pred)))
            y_layer_preds[:,i] = y_pred
            print("test------所有层r2:{}\n".format(r2_score(y, np.mean(y_layer_preds[:,:i+1], axis=1))))

        return y_pred

    def sample_weight_cal(self,x,y,y_pred,sample_weight,hard_index,easy_index,layer):
        # sample_weight = np.ones(len(y))
        error_vector = abs(y_pred - y)
        error_max = max(error_vector)
        error_vector /= error_max
        count = np.zeros(len(y_pred))
        for i in hard_index:
            easy_random_index = random.choices(easy_index, k=int(len(easy_index) /100))
            for j in easy_random_index:
                if abs(y_pred[i]-y_pred[j])<0.05 and abs(y[j]-y[i])<0.05 and pearsonr(x[i], x[j])[0]>0.6:
                    count[i]+=1
        count_pro = [i/max(count) for i in count]
        easy_count = 0
        for i in count_pro:
            if i==0:
                easy_count+=1
        for i in hard_index:
            sample_weight[i] += np.exp(error_vector[i]) * count_pro[i]  #
        print(max(sample_weight))

        return sample_weight

