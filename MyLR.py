import numpy as np

class MyLogisticRegression(object):
    def __init__(self, eta = 0.0001, epoch = 20, lambda_ = 0.001):
        self.eta = eta
        self.epoch = epoch
        self.lambda_ = lambda_

    def fit(self, X, y):
        self.w = np.zeros(1 + X.shape[1])
        for i in range(0,self.epoch):
            output = self.Logistic(np.dot(X, self.w[1:]) + self.w[0])
            error = y - output
            self.w[1:] += self.eta * X.T.dot(error) + self.w[1:] * self.lambda_ / X.shape[1]  #加正则项
            self.w[0] += self.eta * error.sum()
        return self
    
    def predict(self, x):
        prediction = np.zeros(x.shape[0])
        prediction_score = self.Logistic(np.dot(x, self.w[1:]) + self.w[0])
        for i in range(0,x.shape[0]):
            if prediction_score[i] >= 0.5:
                prediction[i] = 1
            else:
                prediction[i] = 0
        return prediction

    def Logistic(self, x):
        return 1 / (1 + np.exp(-x))

