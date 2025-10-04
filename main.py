from linear_regression import LinearRegression
import utils
from sklearn.model_selection import KFold
import numpy as np

LAMBDA = 1000
LAMBDAS = [0, 0.01, 0.1, 1, 10, 100, 1000]
CSV_FILENAME = "home_scaled_csv.csv"
TXT_FILENAME = "home_scaled.txt"

# Parse text to csv
utils.parse_txt_file_to_csv(TXT_FILENAME, CSV_FILENAME)


# Read data to matrix and list, numpy
X, Y = utils.load_data_to_array(CSV_FILENAME)
n_samples, n_features = X.shape


# ============================================ Question a ===================================================
# Ridge and Lasso Regression
ridge = LinearRegression("ridge", 1000 / n_samples)
lasso = LinearRegression("lasso", 1000 / n_samples)
models = [ridge, lasso]

for model in models:
    model.train(X, Y)
    w = model.get_weights()
    model.print_params(w)


# ============================================ Question b ===================================================
model_names = ["ridge", "lasso"]
errors = {}
for names in model_names:
    errors[names] = {
        "training_errors": {},
        "testing_errors": {}
    }

X_train, X_test = X[:400], X[400:]
Y_train, Y_test = Y[:400], Y[400:]
for l in LAMBDAS:
    ridge = LinearRegression("ridge", l/n_samples)
    lasso = LinearRegression("lasso", l/n_samples)
    models = [ridge, lasso]
    
    for model in models:
        model.train(X_train, Y_train)
        errors[model.name]["training_errors"][l] = model.get_mse_loss(X_train, Y_train)
        errors[model.name]["testing_errors"][l] = model.get_mse_loss(X_test, Y_test)


# ============================================ Question c ===================================================
k = 5
kf = KFold(n_splits=5, shuffle=True)
val_errors = {name: {} for name in model_names}

    
for l in LAMBDAS:
    val_errors_lambda = {name: [] for name in model_names}

    for train_idx, val_idx in kf.split(X_train):
        X_train_kf, X_val_kf = X_train[train_idx], X_train[val_idx]
        Y_train_kf, Y_val_kf = Y_train[train_idx], Y_train[val_idx]

        ridge = LinearRegression("ridge", l/n_samples)
        lasso = LinearRegression("lasso", l/n_samples)
        models = [ridge, lasso]
        
        for model in models:
            model.train(X_train_kf, Y_train_kf)
            val_errors_lambda[model.name].append(model.get_mse_loss(X_val_kf, Y_val_kf))
    
    for name in model_names:
        val_errors[name][l] = np.mean(val_errors_lambda[name])
            


