from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error
import numpy as np

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
    
    
    """
    Returns:
        bias: scalar
        weights: [n_features,]
    """
    def get_weights(self):
        return {"bias": self.model.intercept_,
                "weights": self.model.coef_}
    
    """
    Returns:
        MSE loss: scalar
    """
    def get_mse_loss(self, X, Y):
        y_pred = self.model.predict(X)
        mse = mean_squared_error(Y, y_pred)
        return mse
    

    def print_params(self, params: dict):
        print(f"{self.model_name.capitalize()} bias and weights:")
        print(f"\tBias w0 = {np.round(params['bias'], 4)}")
        print(f"\tw.T = {np.round(params['weights'], 4)}")