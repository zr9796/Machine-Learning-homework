import numpy as np
spambase = np.loadtxt('data/spambase/spambase.data', delimiter = ",")
spamx = spambase[:, :57]
spamy = spambase[:, 57]

from sklearn.model_selection import train_test_split
trainX, testX, trainY, testY = train_test_split(spamx, spamy, test_size = 0.4, random_state = 32)
trainX.shape, trainY.shape, testX.shape, testY.shape

# YOUR CODE HERE
# class myGaussianNB():
#     def __init__(self):
#         self.lambda_=0    #贝叶斯系数 取0时，即为极大似然估计
#         self.y_types_count=None #y的（类型：数量）
#         self.y_types_proba=None #y的（类型：概率）
#         self.x_types_proba=dict() #（xi 的编号,xi的取值，y的类型）：概率

#     def fit(self,X_train,y_train):
#         self.y_types=np.unique(y_train)  #y的所有取值类型
#         X=pd.DataFrame(X_train)          #转化成pandas DataFrame数据格式，下同
#         y=pd.DataFrame(y_train)
#         # y的（类型：数量）统计
#         self.y_types_count=y[0].value_counts()
#         # y的（类型：概率）计算
#         self.y_types_proba=(self.y_types_count+self.lambda_)/(y.shape[0]+len(self.y_types)*self.lambda_)

#         # （xi 的编号,xi的取值，y的类型）：概率的计算
#         for idx in X.columns:       # 遍历xi
#             for j in self.y_types:  # 选取每一个y的类型
#                 p_x_y=X[(y==j).values][idx].value_counts() #选择所有y==j为真的数据点的第idx个特征的值，并对这些值进行（类型：数量）统计
#                 for i in p_x_y.index: #计算（xi 的编号,xi的取值，y的类型）：概率
#                     self.x_types_proba[(idx,i,j)]=(p_x_y[i]+self.lambda_)/(self.y_types_count[j]+p_x_y.shape[0]*self.lambda_)

#     def predict(self,X_new):
#         res=[]
#         for y in self.y_types: #遍历y的可能取值
#             p_y=self.y_types_proba[y]  #计算y的先验概率P(Y=ck)
#             p_xy=1
#             for idx,x in enumerate(X_new):
#                 p_xy*=self.x_types_proba[(idx,x,y)] #计算P(X=(x1,x2...xd)/Y=ck)
#             res.append(p_y*p_xy)
#         for i in range(len(self.y_types)):
#             print("[{}]对应概率：{:.2%}".format(self.y_types[i],res[i]))
#         #返回最大后验概率对应的y值
#         return self.y_types[np.argmax(res)]
from scipy.stats import multivariate_normal
class myGaussianNB:
    
    mu = None
    cov = None
    n_classes = None
    
    def __init__(self):
        a = None
    
    def predict(self,x):
        prob_vect = np.zeros((self.n_classes, np.size(x,0)))

        mnormal_0 = multivariate_normal(mean=self.mu[0], cov=self.cov[0])
        mnormal_1 = multivariate_normal(mean=self.mu[1], cov=self.cov[1])
        temp = np.zeros((self.n_classes, np.size(x,0)))
        temp[0] = mnormal_0.pdf(x)
        temp[1] = mnormal_1.pdf(x)
        for i in range(self.n_classes):

            # We use uniform priors
            prior = 1./self.n_classes
            prob_vect[i] = prior*temp[i]
            sumatory = np.zeros(np.size(x,0))
            for j in range(self.n_classes):
                sumatory += prior*temp[j]
            prob_vect[i] = prob_vect[i]/sumatory

        prediction = np.zeros(np.size(x,0))
        for i in range((np.size(x,0))):
            if prob_vect[0][i] > prob_vect[1][i]:
                prediction[i] = 0
            else:
                prediction[i] = 1
        return prediction
        
    def fit(self, X,y):
        self.mu = []
        self.cov = []
        
        self.n_classes = int(np.max(y))+1
        
        for i in range(self.n_classes):
            Xc = X[y==i]
            
            mu_c = np.mean(Xc, axis=0)
            self.mu.append(mu_c)
            
            cov_c = np.zeros((X.shape[1], X.shape[1]))
            for j in range( Xc.shape[0]):
                a = Xc[j].reshape((X.shape[1],1))
                b = Xc[j].reshape((1,X.shape[1]))
                cov_ci = np.multiply(a, b)
                cov_c = cov_c+cov_ci
            cov_c = cov_c/float(X.shape[0])
            self.cov.append(cov_c)
        self.mu = np.asarray(self.mu)
        self.cov = np.asarray(self.cov)



# test case
from sklearn.metrics import accuracy_score
model = myGaussianNB()
model.fit(trainX, trainY)
# accuracy_score(testY, model.predict(testX))

# YOUR CODE HERE
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
y_pred = model.predict(testX)
a_score = accuracy_score(testY, y_pred)
p_score = precision_score(testY, y_pred)
r_score = recall_score(testY, y_pred)
f_score = f1_score(testY, y_pred)

print("accuracy_score:" , a_score)
print("precision_score:", p_score)
print("recall_score:", r_score)
print("f1_score:", f_score)
