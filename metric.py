import numpy as np

class Metrics:
    def __init__(self, metric_name: str = None):
        self.metric = self.get_metric(metric_name) if metric_name else None

    @staticmethod
    def mae(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    @staticmethod
    def mse(y_true, y_pred):
        return np.mean((y_true - y_pred)**2)

    @staticmethod
    def rmse(y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred)**2))

    @staticmethod
    def mape(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    @staticmethod
    def r2_score(y_true, y_pred):
        return 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))

    def get_metric(self, metric_name):

        if metric_name == 'mae':
            return self.mae
        elif metric_name == 'mse':
            return self.mse
        elif metric_name == 'rmse':
            return self.rmse
        elif metric_name =='mape':
            return self.mape
        elif metric_name == 'r2_score':
            return self.r2_score
        else:
            return None

    def __call__(self, y_true, y_pred):
         return self.metric(y_true, y_pred)

if __name__ == '__main__':
    metric = Metrics('mse')
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([2.5, 0, 2.8, 8])
    print(metric(y_true, y_pred))
