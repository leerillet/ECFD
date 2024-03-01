import numpy as np

class layer(object):
    def __init__(self, layer_id):
        self.layer_id = layer_id
        self.estimators = []  # 森林个数
        self.pred = []
        self.forestweights = []  # 层内森林权重

    def add_estimator(self, estimator):
        if estimator != None:
            self.estimators.append(estimator)

    def get_layer_id(self):
        return self.layer_id

    def get_enhanced(self,x):   #这里做修改
        enhanced = np.zeros((x.shape[0], len(self.estimators)*3))
        for i in range(len(self.estimators)):
            enhanced[:, i*3:(i+1)*3] = self.estimators[i].test_get_enhanced(x)
        return enhanced

    def predict(self, x):
        predicts = np.zeros((x.shape[0], len(self.estimators)))
        for i in range(len(self.estimators)):
            tmp = self.estimators[i].predict(x)
            predicts[:, i] = tmp
        return predicts




