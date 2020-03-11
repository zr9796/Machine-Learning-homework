import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

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


data = np.loadtxt('data/kaggle_mnist/mnist_train.csv', delimiter = ",")
data_X = data[:, :784]
# data_X = data_X.reshape((28,28))
data_y = data[:, 0]
X_Standard = StandardScaler().fit_transform(data_X)
trainX, testX, trainY, testY = train_test_split(X_Standard, data_y, test_size = 0.4, random_state = 32)
y_MLP = np.zeros((trainY.shape[0], 10))
for i in range(0,trainY.shape[0]):
    y_MLP[i][int(trainY[i])] = 1

model = MyMLP()
model.fit(trainX, y_MLP)
y_pred = model.predict(testX)

a_score = accuracy_score(testY, y_pred)
p_score = np.mean(precision_score(testY, y_pred, average= None))
r_score = np.mean(recall_score(testY, y_pred, average=None))
f_score = np.mean(f1_score(testY, y_pred, average=None))
print("accuracy_score:" , a_score)
print("precision_score:", p_score)
print("recall_score:", r_score)
print("f1_score:", f_score)