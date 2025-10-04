from linear_regression import LinearRegression
from sklearn.model_selection import KFold
import utils

LAMBDAS = [0, 0.01, 0.1, 1, 10, 100, 1000]


class KFoldCrossValidation:
    def __init__(self, n_splits, model_name):
        self.n_splits = n_splits
        self.model_name = model_name
        self.kf = KFold(n_splits=n_splits)


    def k_fold_cross_validate(self, l, X, Y):
        val_error = []
        for train_idx, val_idx in self.kf.split(X):
            n_samples = train_idx.shape[0]
            X_train_kf, X_val_kf = X[train_idx], X[val_idx]
            Y_train_kf, Y_val_kf = Y[train_idx], Y[val_idx]

            model = LinearRegression(self.model_name, l / n_samples)
            model.train(X_train_kf, Y_train_kf)
            loss_val = model.get_rmse_loss(X_val_kf, Y_val_kf)
            val_error.append(loss_val)

        return sum(val_error) / len(val_error)
    

    def find_best_lambda(self, X, Y, lambdas = LAMBDAS):
        val_errors = []
        for l in lambdas:
            val_loss = self.k_fold_cross_validate(l, X, Y)
            val_errors.append((l, val_loss))
        
        return min(val_errors, key= lambda x: x[1])[0]
    

def kf_best_lambda_test_loss(X_train, Y_train, X_test, Y_test, k):
    for name in utils.model_names:
        kf_cross_validate = KFoldCrossValidation(n_splits=k, model_name=name)
        best_lambda = kf_cross_validate.find_best_lambda(X_train, Y_train, LAMBDAS)
        print(f"Best lambda for {name} is {round(best_lambda, 4)}")

        # Train again with all X_train
        model = LinearRegression(name, alpha=best_lambda / X_train.shape[0])
        model.train(X_train, Y_train)
        rmse = model.get_rmse_loss(X_test, Y_test)
        print(f"Test error at best lambda for {name} is {round(rmse, 4)}")