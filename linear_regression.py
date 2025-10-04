from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

""" This class is for Ridge Linear Regression and Lasso Linear Regression
Args:
    model_name: can only be "ridge" or "lasso"
    alpha: lambda / n_samples, lambda is the regularization parameter
"""
class LinearRegression:
    def __init__(self, model_name: str, alpha):
        if model_name not in ["ridge", "lasso"]:
            print("Only serves Ridge and Lasso linear regression")
            exit()
        self.model_name = model_name
        if model_name == "ridge":
            self.model = Ridge(alpha = alpha)
        else:
            self.model = Lasso(alpha = alpha)
    
    @property
    def name(self):
        return self.model_name

    """ Train model
    Args:
        X: [n_samples x n_features]
        Y: [n_samples,]
    """
    def train(self, X, Y):
        self.model.fit(X, Y)
    
    
    def get_weights(self):
        return np.concatenate(([self.model.intercept_], self.model.coef_))
    
    
    """
    Return:
        RMSE loss: scalar
    """
    def get_rmse_loss(self, X, Y):
        y_pred = self.model.predict(X)
        mse = mean_squared_error(Y, y_pred)
        return np.sqrt(mse)
    

    def print_params(self):
        params = self.get_weights()
        df = pd.DataFrame({"weights": [f"w{i}" for i in range(params.shape[0])], 
                           "value": np.round(params, 4)})
        return df

    
    def weight_to_csv(self, filename):
        df = self.print_params()
        df.to_csv(filename)