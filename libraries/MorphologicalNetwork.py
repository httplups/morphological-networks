from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

def get_intercept(X, y):
    # quando y == 1, y == label (one vs all)
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
            # print(i,'<', i)
            ind = np.logical_and((y == label), (X[:, i] < -w[i]))
            boxes.append(
                np.hstack([-np.min(X[ind, :], axis=0), np.max(X[ind, :], axis=0)]))
        except:
            # print('not', i, '<', i)
            pass

        try:
            # print(i, '>', i+n)

            ind = np.logical_and((y == label), (X[:, i] > w[i+n]))
            boxes.append(
                np.hstack([-np.min(X[ind, :], axis=0), np.max(X[ind, :], axis=0)]))
        except:
            # print('not', i, '<', i+n)
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
        self.boxes_ = [[] for i in range(self.Nclasses_)]
        for label in range(self.Nclasses_):
            self.fit_class(Xtr, ytr, label)
        return self

    def fit_class(self, Xtr, ytr, label):
        
        # a condição de parada agora é quando não tiver mais pontos da classe específica
        if (Xtr[ytr == label].shape[0] == 0):
            return self

        K, N = Xtr.shape
        
        w = get_intercept(Xtr,ytr == label)
        [self.boxes_[label].append(a) for a in get_boxes(Xtr,ytr,w, label)]
        
        Xi, yi = get_inside(Xtr,ytr,w)
        return self.fit_class(Xi, yi, label)
    
    def decision_function(self,X, label):
        X = np.hstack([X,-X])
        return np.max(np.vstack([np.min(X+w, axis=1) for w in self.boxes_[label]]),axis=0)
    
    def predict(self,X):
        Y = np.vstack([self.decision_function(X,label) for label in range(self.Nclasses_)])

        return np.argmax(Y,axis=0)
  
