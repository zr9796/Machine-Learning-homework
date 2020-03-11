import numpy as np

class MyMLP(object):
    def __init__(self, eta = 0.0001, epoch = 20):
        self.eta = eta
        self.epoch = epoch

    def fit(self, X, y):
        self.w = np.zeros((1 + X.shape[1], y.shape[1]))
        for i in range(0,self.epoch):
            output = self.MLP(X)
            error = y - output
            self.w[1:] += self.eta * X.T.dot(error)
            self.w[0] += self.eta * error.sum()
        return self
    
    def predict(self, x):
        prediction = np.zeros(x.shape[0])
        prediction_MLP = self.MLP(x)
        for i in range(0,x.shape[0]):
            for j in range(0,10):
                if prediction_MLP[i][j] == 1:
                    prediction[i] = j
        return prediction

    def MLP(self, x):
        result = np.dot(x, self.w[1:]) + self.w[0]
        for i in range(0,x.shape[0]):
            for j in range(0,10):
                if result[i][j] >= 0:
                    result[i][j] = 1
                else:
                    result[i][j] = 0
        return result
