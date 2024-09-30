import random

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression

from metric import Metrics

class MyLineReg:
    def __init__(self, n_iter: int, lr, weights: list[float] = None, metric_name: str = None,
                 reg: str = None, l1_coef: float = 0.0, l2_coef: float = 0.0,
                 sgd_sample: float = None, random_state: int = 42,
                 ):

        self.n_iter = n_iter
        self.lr = (lambda iter: 0.05 * (0.85 ** iter) if lr == 'lambda' else lr)
        self.weights = weights
        self.metric = Metrics(metric_name)
        self.metric_name = metric_name
        self.metric_score = None
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.seed = random_state
        self.sgd_sample = sgd_sample

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: int = False):
        random.seed(self.seed)
        X.insert(0, 'feature_ones', 1)
        print('X', X)
        print('y:', y)
        self.weights = np.ones(X.shape[1])

        for iter in range(1, self.n_iter + 1):

            if isinstance(self.sgd_sample, int):
                batch_size = self.sgd_sample
            elif isinstance(self.sgd_sample, float):
                batch_size = max(1, round(self.sgd_sample * X.shape[0]))
            else:
                batch_size = X.shape[0]

            sample_row_idx = random.sample(range(X.shape[0]), batch_size)

            X_batch = X.iloc[sample_row_idx]
            y_batch = y.iloc[sample_row_idx]
            current_lr = self.lr(iter)

            y_pred = np.dot(X_batch, self.weights)

            dw = (2/len(X)) * np.dot(X_batch.T, (y_pred - y_batch.values))

            if self.reg == 'l1':
                dw += self.l1_coef * np.sign(self.weights)
            elif self.reg == 'l2':
                dw += 2 * self.l2_coef * self.weights
            elif self.reg == 'elasticnet':
                dw += self.l1_coef * np.sign(self.weights) + 2 * self.l2_coef * self.weights

            self.weights -= current_lr * dw
            y_pred_full_data = np.dot(X, self.weights)
            loss = self.MSEloss(y, y_pred_full_data)

            if self.metric and callable(self.metric.metric):
                self.metric_score = self.metric(y, y_pred_full_data)

            if verbose:
                if ((iter % verbose == 0) or iter == 1) and self.metric_name is not None:
                    print(f'{iter} | loss: {loss} | {self.metric_name}: {self.metric_score} | lr: { current_lr}')
                else:
                    if (iter % verbose == 0 or iter == 1) and self.metric_name is None:
                        print(f'{iter} | loss: {loss} | lr: { current_lr}')


    def predict(self, X: pd.DataFrame):
        predictions = []
        X_values = X.values
        num_examples, num_features = np.shape(X_values)
        ones_column = np.ones((num_examples, 1))
        X_stack = np.hstack((ones_column, X_values))

        for row in range(num_examples):
            yhat = 0
            for col in range(num_features):
                yhat += X_stack[row, col] * self.weights[col]
            predictions.append(yhat)

        return np.array(predictions)

    def MSEloss(self, y_true: float, y_pred: float):
        loss = np.mean((y_true - y_pred) ** 2)

        if self.reg == 'l1':
            loss += self.l1_coef * np.sum(np.abs(self.weights))
        elif self.reg == 'l2':
            loss += self.l2_coef * np.sum(self.weights ** 2)
        elif self.reg == 'elasticnet':
            loss += self.l1_coef * np.sum(np.abs(self.weights)) + self.l2_coef * np.sum(self.weights ** 2)

        return loss

    def get_coef(self):
        return self.weights[1:]

    def get_best_score(self):
        return self.metric_score

    def __repr__(self):
        return f'MyLineReg class: n_iter={self.n_iter}, learning_rate={self.lr}'


if __name__ == '__main__':
    model1 = MyLineReg(n_iter=100, metric_name='mse', lr=0.006, sgd_sample=0.8)
    # model2 = MyLineReg(n_iter=100, lr=0.1)
    # model3 = MyLineReg(n_iter=100, lr=0.1)

    X1, y1 = make_regression(n_samples=5, n_features=5, noise=1, random_state=42)
    # X2, y2 = make_regression(n_samples=5, n_features=3, noise=2, random_state=42)
    # X3, y3 = make_regression(n_samples=10, n_features=15, noise=5, random_state=42)

    X1 = pd.DataFrame(X1, columns=[f'feature_{i}' for i in range(X1.shape[1])])
    y1 = pd.Series(y1)

    # X2 = pd.DataFrame(X2, columns=[f'feature_{i}' for i in range(X2.shape[1])])
    # y2 = pd.Series(y2)
    #
    # X3 = pd.DataFrame(X3, columns=[f'feature_{i}' for i in range(X3.shape[1])])
    # y3 = pd.Series(y3)

    model1.fit(X=X1, y=y1, verbose=10)
    print('best score: ', model1.get_best_score())
    # model2.fit(X=X2, y=y2, verbose=10)
    # model3.fit(X=X3, y=y3, verbose=10)
    # print('Mean of weights model1:', np.mean(model1.get_coef()))
    # print('Mean of weights model2:', np.mean(model2.get_coef()))
    # print('Mean of weights model3:', np.mean(model3.get_coef()))
    #
    # pred1 = model1.predict(X1)
    # pred2 = model2.predict(X2)
    # pred3 = model3.predict(X3)
    # print('pred 1', np.sum(pred1))
    # print('pred 2', np.sum(pred2))
    # print('pred 3', np.sum(pred3))
    # print('Pred sum:', np.sum(pred1) + np.sum(pred2) + np.sum(pred3))



