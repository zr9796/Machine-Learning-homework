import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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


spambase = np.loadtxt('data/spambase/spambase.data', delimiter = ",")
spamx = spambase[:, :57]
spamy = spambase[:, 57]

X_Standard = StandardScaler().fit_transform(spamx)
trainX, testX, trainY, testY = train_test_split(X_Standard, spamy, test_size = 0.4, random_state = 32)

model = MyLogisticRegression()
model.fit(trainX,trainY)
y_pred = model.predict(testX)
# y_pred = cross_val_predict(model, X_Standard, spamy, cv=10)
a_score = accuracy_score(testY, y_pred)
p_score = precision_score(testY, y_pred)
r_score = recall_score(testY, y_pred)
f_score = f1_score(testY, y_pred)

print("accuracy_score:" , a_score)
print("precision_score:", p_score)
print("recall_score:", r_score)
print("f1_score:", f_score)