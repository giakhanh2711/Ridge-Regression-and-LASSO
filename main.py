from linear_regression import LinearRegression
import utils
from kfold_cross_validation import kf_best_lambda_test_loss

import numpy as np
import warnings
warnings.filterwarnings("ignore")

LAMBDA = 1000
LAMBDAS = [0, 0.01, 0.1, 1, 10, 100, 1000]
CSV_FILENAME = "house_scale_csv.csv"
TXT_FILENAME = "house_scale.txt"
k = 5

# Parse text to csv
utils.parse_txt_file_to_csv(TXT_FILENAME, CSV_FILENAME)

# Read data to matrix and list, numpy
X, Y = utils.load_data_to_array(CSV_FILENAME)
_, n_features = X.shape


# ============================================ Question a ===================================================
# Ridge and Lasso Regression
n_samples, _ = X.shape
for name in utils.model_names:
    model = LinearRegression(name, LAMBDA / n_samples)
    model.train(X, Y)
    model.weight_to_csv(f"w_{name}.csv")


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
n_samples, _ = X_train.shape

for l in LAMBDAS:
    ridge = LinearRegression("ridge", l/n_samples)
    lasso = LinearRegression("lasso", l/n_samples)
    models = [ridge, lasso]
    
    for model in models:
        model.train(X_train, Y_train)
        errors[model.name]["training_errors"][l] = model.get_rmse_loss(X_train, Y_train)
        errors[model.name]["testing_errors"][l] = model.get_rmse_loss(X_test, Y_test)

print("\n", "*"*20, "QUESTION B - Training and Testing with different lambda and choose the best on test", "*"*20)
for name in utils.model_names:
    print(name)
    a = np.argmin(list(errors[name]["testing_errors"].values()))
    print(LAMBDAS[a], round(np.min(list(errors[name]["testing_errors"].values())), 4))
print("*"*50)

# ============================================ Question c ===================================================
k = 5
X_train, X_test = X[:400], X[400:]
Y_train, Y_test = Y[:400], Y[400:]

print("QUESTION c")
kf_best_lambda_test_loss(X_train, Y_train, X_test, Y_test, k)
print("\n", "*"*50, "\n")

# ============================================ Question d ===================================================
utils.parse_txt_file_to_csv("house.txt", "house_origin.csv")
Xo, Yo = utils.load_data_to_array("house_origin.csv")

k = 5
Xo_train, Xo_test = Xo[:400], Xo[400:]
Yo_train, Yo_test = Yo[:400], Yo[400:]

print("QUESTION d")
kf_best_lambda_test_loss(Xo_train, Yo_train, Xo_test, Yo_test, k)
print("\n", "*"*50, "\n")


# ============================================ Question e ===================================================
num_noise_features = 10000
noise = np.random.rand(Xo.shape[0], num_noise_features)
Xo_noise = np.hstack([Xo, noise])

k = 5
Xo_noise_train, Xo_noise_test = Xo_noise[:400], Xo_noise[400:]
Yo_train, Yo_test = Yo[:400], Yo[400:]

print("QUESTION e")
kf_best_lambda_test_loss(Xo_noise_train, Yo_train, Xo_noise_test, Yo_test, k)
