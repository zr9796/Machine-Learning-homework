import numpy as np

class MyLinearRegression(object):
    def __init__(self, eta = 0.0001, epoch = 20):
        self.eta = eta
        self.epoch = epoch

    def fit(self, X, y):
        self.w = np.zeros((1 + X.shape[1],1))
        for i in range(0,self.epoch):
            output = np.dot(X, self.w[1:]) + self.w[0]
            error = y - output
            # print(y.shape)
            # print(output.shape)
            # print(error.shape)
            # print(np.dot(X.T, error).shape)
            self.w[1:] += self.eta * X.T.dot(error)
            self.w[0] += self.eta * error.sum()
            # print('error: ', error)
        return self

    def predict(self, x):
        prediction = np.dot(x, self.w[1:]) + self.w[0]
        return prediction

def Polynomial(x, degree=3):
    return x ** degree

def Gaussian(x, mean, var):
    return (x - mean) / (2 * (var ** 2))

def Sigmoid(x):
    return 1 / (1 + np.exp(-x))

