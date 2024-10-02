import pandas as pd
import numpy as np
from sklearn.datasets import make_multilabel_classification

class MyLogReg():
    def __init__(self, n_iter: int = 10, lr: float = 0.1):
        self.n_iter = n_iter
        self.lr = lr
        self.weights = np.array([])

    def log_loss(self, y_true, y_pred):
        eps = 1e-15
        return -np.mean(y_true * np.log(y_pred  + eps) + (1 - y_true) * np.log(1 - y_pred + eps))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: int = False):
        X.insert(0, 'feature_ones', 1)
        print('X:\n', X)
        print('y:\n', y)
        self.weights = np.ones(X.shape[1])

        for iter in range(1, self.n_iter+1):

            y_pred = np.dot(X, self.weights)
            y_pred = self.sigmoid(y_pred)
            loss = self.log_loss(y, y_pred)

            dw = np.dot(X.T, (y_pred - y)) / y.size

            self.weights -= self.lr * dw

            if verbose and (iter % verbose == 0 or iter == 1):
                if iter == 1:
                    print(f'start | loss: {loss}')
                else:
                    print(f'{iter} | loss: {loss}')




    def __repr__(self):
        return f'MyLineReg class: n_iter={self.n_iter}, learning_rate={self.lr}'

    def get_coef(self):
        return self.weights[1:]



if __name__ == '__main__':
    """
    test1
    logreg = MyLogReg()
    print(logreg)
    """


    """
    
    #test2
    model1 = MyLogReg(n_iter=100, lr=0.009)

    X1, y1 = make_multilabel_classification(n_samples=5, n_features=5, n_classes=1)
    X1 = pd.DataFrame(X1, columns=[f'feature_{i}' for i in range(X1.shape[1])])
    y1 = pd.Series(y1.ravel())

    model1.fit(X=X1, y=y1)
    
    """







