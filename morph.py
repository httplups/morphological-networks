from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn import preprocessing
import numpy as np

def get_intercept(X, y):
    w = list()
    for i in range(2):
        ind = (y == i)
        w.append(
            np.hstack([-np.min(X[ind, :], axis=0), np.max(X[ind, :], axis=0)]))
    return np.minimum(w[0], w[1])

def get_boxes(X, y, w, label):
    boxes = list()
    n = X.shape[1]
    for i in range(n):
        try:
            ind = np.logical_and((y == label), (X[:, i] < -w[i]))
            boxes.append(
                np.hstack([-np.min(X[ind, :], axis=0), np.max(X[ind, :], axis=0)]))
        except:
            pass

        try:
            ind = np.logical_and((y == label), (X[:, i] > w[i+n]))
            boxes.append(
                np.hstack([-np.min(X[ind, :], axis=0), np.max(X[ind, :], axis=0)]))
        except:
            pass
    return boxes


def get_inside(X, y, w):
    n = X.shape[1]
    ind = np.min(np.minimum(X+w[0:n],-X+w[n:]),axis=1)>0
    return X[ind], y[ind]

class MLMP(BaseEstimator, ClassifierMixin):
    
    def __init__(self, verbose = False, myInf = 1.e+10):
        self.verbose = verbose
        self.myInf = 1.e+10 
    
    def fit(self, Xtr, ytr):
        self.Nclasses_ = np.size(np.unique(ytr))
        self.le = preprocessing.LabelEncoder()
        self.le.fit(ytr)
        self.classes = np.unique(self.le.transform(ytr))
        ytr = self.le.transform(ytr)
        self.boxes_ = [[] for i in range(self.Nclasses_)]
        
        self.dim_X_ = np.min(Xtr, axis=0) != np.max(Xtr, axis=0)
        Xtr = Xtr[:, self.dim_X_]
        
        for label in self.classes:
            self.fit_class(Xtr, ytr, label)
        return self

    def fit_class(self, Xtr, ytr, label):
        ytrl = (ytr == label)
        if (Xtr[ytrl].shape[0] == 0):
            return self
        
        K, N = Xtr.shape
        
        if np.unique(ytrl).shape[0] > 1:
        
            w = get_intercept(Xtr, ytrl)
            [self.boxes_[label].append(a) for a in get_boxes(Xtr,ytr,w,label)]

            Xi, yi = get_inside(Xtr,ytr,w)
            return self.fit_class(Xi, yi, label)
        else:
            w = np.hstack([-np.min(Xtr, axis=0), np.max(Xtr, axis=0)])
            self.boxes_[ytr[0]].append(w)
        return self
            
    
    def decision_function(self,X, label):

        X = X[:, self.dim_X_]
        X = np.hstack([X,-X])
        if self.boxes_[label] == []:
            return np.ones(X.shape[0],) * -1.e+12
        
        return np.max(np.vstack([np.min(X+w, axis=1) for w in self.boxes_[label]]),axis=0)
    
    def predict(self,X):
        Y = np.vstack([self.decision_function(X,label) for label in self.classes])
        pred = np.argmax(Y,axis=0)
        return self.le.inverse_transform(pred)
    
    
class SLMP(BaseEstimator, ClassifierMixin):

    def __init__(self, verbose = False, myInf = 1.e+10):
        self.verbose = verbose
        self.myInf = 1.e+10 

    def fit(self, Xtr, ytr):
        
        self.le = preprocessing.LabelEncoder()
        self.le.fit(ytr)
        ytr = self.le.transform(ytr)
    
        K, N = Xtr.shape
        
        self.w_ = -np.max(Xtr[ytr==0,:],axis=0)      
        return self

    def decision_function(self,X):
        return np.max(X+self.w_,axis=1)

    def predict(self,X):
        pred = self.decision_function(X)>0
        pred = [int(x) for x in pred]
        return self.le.inverse_transform(pred)
        
class SLMPbox(BaseEstimator, ClassifierMixin):

    def __init__(self, verbose = False, myInf = 1.e+10):
        self.verbose = verbose
        self.myInf = 1.e+10 

    def fit(self, Xtr, ytr):
        
        self.le = preprocessing.LabelEncoder()
        self.le.fit(ytr)
        ytr = self.le.transform(ytr)
        
        Xtr = np.hstack([Xtr,-Xtr])

        K, N = Xtr.shape
        self.w_ = -np.max(Xtr[ytr==0,:],axis=0)     
        return self

    def decision_function(self,X):
        X = np.hstack([X,-X])
        return np.max(X+self.w_,axis=1)

    def predict(self,X):
        pred = self.decision_function(X)>0
        pred = [int(x) for x in pred]
        return self.le.inverse_transform(pred)
