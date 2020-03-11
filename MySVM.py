import numpy as np

class MySVM(object):
    def __init__(self, eta = 0.0001, epoch = 20, kernel = 'linear', C = 1.0):
        self.eta = eta
        self.epoch = epoch
        self.kernel = kernel
        self.C = C

    def fit(self, X, y):
        self.w = np.zeros(1 + X.shape[1])
        for i in range(0,1 + X.shape[1]):
            self.w[i] = np.random.rand()
            # print(self.w)
        self.alpha = np.zeros(X.shape[0])
        for count in range(0,self.epoch):
            list_alpha = []
            output = np.dot(self.w[1:], X.T) + self.w[0]
            # print(output)
            # print(self.alpha)
            # print(self.w)
            #SMO
            for i in range(0, X.shape[0]):
                temp = y[i]*output[i]
                #不满足KKT条件
                if (temp<=1 and self.alpha[i]<self.C) or (temp>=1 and self.alpha[i]>0) or \
                    (temp==1 and (self.alpha[i]==self.C or self.alpha[i]==0)):
                    list_alpha.append(i)
                else:
                    if (temp>=1 and self.alpha[i]==0) or (temp==1 and self.alpha[i]>=0 and self.alpha[i]<=self.C) or \
                        (temp<=1 and self.alpha[i]==self.C):
                        continue
                    else:
                        list_alpha.append(i)
            if(len(list_alpha)==0):
                break
            elif(len(list_alpha)==1):
                list_alpha.append(list_alpha[0]+1)
            alpha_1, alpha_2 = np.random.choice(list_alpha,2,replace=False)
            # print(list_alpha)

            #更新alpha
            E1 = output[alpha_1] - y[alpha_1]
            E2 = output[alpha_2] - y[alpha_2]
            eta = 2*np.dot(X[alpha_1],X[alpha_2].T) - np.dot(X[alpha_1],X[alpha_1].T) - np.dot(X[alpha_2],X[alpha_2].T)
            temp = y[alpha_2] * (E1 - E2) / eta
            self.alpha[alpha_2] = self.alpha[alpha_2] - temp
            self.alpha[alpha_1] = self.alpha[alpha_1] + y[alpha_1]*y[alpha_2]*temp     

            #更新w和b
            self.w[1:] = (self.alpha * y * X.T).sum(axis = 1)
            b_1 = self.w[0] - E1 - y[alpha_1]*y[alpha_1]*y[alpha_2]*temp*np.dot(X[alpha_1],X[alpha_1].T) + \
                y[alpha_2]*temp*np.dot(X[alpha_1],X[alpha_2].T)
            b_2 = self.w[0] - E2 - y[alpha_1]*y[alpha_1]*y[alpha_2]*temp*np.dot(X[alpha_1],X[alpha_2].T) + \
                y[alpha_2]*temp*np.dot(X[alpha_2],X[alpha_2].T)
            if self.alpha[alpha_1]>=0 and self.alpha[alpha_1]<=self.C:
                self.w[0] = b_1
            elif self.alpha[alpha_2]>=0 and self.alpha[alpha_2]<=self.C:
                self.w[0] = b_2
            else:
                self.w[0] = (b_1+b_2)/2
            print(count)
        return self

    def predict(self, x): 
        result = np.dot(x, self.w[1:]) + self.w[0]
        for i in range(0,x.shape[0]):
            if result[i] >= 0:
                result[i] = 1
            else:
                result[i] = -1 
        return result

    def kernel(self, x):
        if self.kennel == 'linear':
            return np.dot(x, x.T)

