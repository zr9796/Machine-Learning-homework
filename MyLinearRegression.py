import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

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

data = np.loadtxt('Dataset/kaggle_house_price_train.csv', delimiter=',')

dataX = data[:, :34]
datay = data[:, 34]

trainX, testX, trainY, testY = train_test_split(dataX, datay, test_size = 0.4, random_state = 32)
trainY = StandardScaler().fit_transform(trainY.reshape(-1,1))

    
############################################################

trainX_poly = Polynomial(trainX, degree=3)
trainX_poly = StandardScaler().fit_transform(trainX_poly)
testX_poly = Polynomial(testX, degree=3)
testX_poly = StandardScaler().fit_transform(testX_poly)

model_poly = MyLinearRegression()
model_poly.fit(trainX_poly,trainY)
predict = model_poly.predict(testX_poly)

poly = StandardScaler()
poly.fit(predict.reshape(-1,1))
y_pred_poly = poly.inverse_transform(predict)

############################################################

trainX_gau = Gaussian(trainX, np.mean(trainX), np.var(trainX))
trainX_gau = StandardScaler().fit_transform(trainX_gau)
testX_gau = Gaussian(testX, np.mean(testX), np.var(testX))
testX_gau = StandardScaler().fit_transform(testX_gau)

model_gau = MyLinearRegression()
model_gau.fit(trainX_gau,trainY)
predict = model_gau.predict(testX_gau)

gau = StandardScaler()
gau.fit(predict.reshape(-1,1))
y_pred_gau = gau.inverse_transform(predict)

# ############################################################

trainX_sig = Sigmoid(trainX)
trainX_sig = StandardScaler().fit_transform(trainX_sig)
testX_sig = Sigmoid(testX)
testX_sig = StandardScaler().fit_transform(testX_sig)

model_sig = MyLinearRegression()
model_sig.fit(trainX_sig,trainY)
predict = model_sig.predict(testX_sig)

sig = StandardScaler()
sig.fit(predict.reshape(-1,1))
y_pred_sig = sig.inverse_transform(predict)


print("Polynomial:")
print('MAE为：',mean_absolute_error(testY,y_pred_poly))
print('RMSE为：',np.sqrt(mean_squared_error(testY,y_pred_poly)))

print("Gaussian:")
print('MAE为：',mean_absolute_error(testY,y_pred_gau))
print('RMSE为：',np.sqrt(mean_squared_error(testY,y_pred_gau)))

print("Sigmoid")
print('MAE为：',mean_absolute_error(testY,y_pred_sig))
print('RMSE为：',np.sqrt(mean_squared_error(testY,y_pred_sig)))


