import numpy as np

class layer(object):
    def __init__(self, layer_id):
        self.layer_id = layer_id
        self.estimators = []
        self.pred = []

    def add_estimator(self, estimator):
        if estimator != None:
            self.estimators.append(estimator)

    def get_layer_id(self):
        return self.layer_id




