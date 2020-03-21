

__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "GPL"

import numpy as np
from numpy.linalg import inv
import time
import math


def calculate_rsme(y, y_hat):
    rownumber = len(y)
    sum = 0

# rsme calculation requires input of two lists (y and y_hat values)
    for i in range(len(y)):
        x = (y[i] - y_hat[i])*(y[i] - y_hat[i])
        sum = sum + x
    rsme_variable = (1/rownumber*sum) ** 0.5
    return rsme_variable

# analytical solver for the ridge regression equation


def minimize_ridge_regression(y, X, reg_param):
    prod = X.transpose() @ X+reg_param * np.eye(13)
    w_opt = inv(prod) @ X.transpose() @ y
    return w_opt


# for isolated testing purposes only:
if __name__ == "__main__":
    data = np.array([[0.06724, 0, 3.24, 0, 0.46, 6.333, 17.2, 5.2146, 4, 430, 16.9, 375.21, 7.34],
                     [0.06724, 0, 3.24, 0, 0.46, 6.333, 17.2, 5.2146, 4, 430, 16.9, 375.21, 7.34]])
    y = np.array([22.6, 22.6])
    optimal_weights = minimize_ridge_regression(data, y, 1)
    print(optimal_weights)
    print("rmse:", calculate_rsme(y, np.inner(optimal_weights, data)))
