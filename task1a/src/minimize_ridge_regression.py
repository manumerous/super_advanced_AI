import numpy as np
import time
import math


class RidgeRegressor():

    def __init__(self, y, x):
        self.y = y
        self.x = x
        self.data_length = len(y)
        self.grad_res = 0.01

    def calculate_rsme(self, y_hat):
        sum = 0
        for i in range(self.data_length):
            x = (self.y[i] - y_hat[i])*(self.y[i] - y_hat[i])
            sum = sum + x
        rsme_variable = (1/len(self.y)*sum) ** 0.5
        return rsme_variable

    def _calculate_optimization_cost(self, w, reg_param):
        # cost = np.linalg.norm(w)
        cost = 0
        for i in range(self.data_length):
            cost += (self.y[i] - np.inner(w, self.x[i, :]))*(self.y[i] -
                        np.inner(w, self.x[i, :])) + reg_param*np.linalg.norm(w)
        return cost

    # gradient calculation using the difference quotient
    def _calculate_gradient(self, w, reg_param):
        gradient = np.zeros(len(w))
        for i in range(len(w)):
            ### error might be here
            current_w = w
            current_w[i] += self.grad_res
            upper_cost = self._calculate_optimization_cost(current_w, reg_param)
            # print(i)
            # print(current_w)
            # print('upper cost:', upper_cost)
            current_w[i] -= 2*self.grad_res
            lower_cost = self._calculate_optimization_cost(current_w, reg_param)
            # print(current_w)
            # print('lower cost:', lower_cost)
            gradient[i] = (upper_cost-lower_cost)/(2*self.grad_res)
            current_w[i] += self.grad_res
            # print(gradient[i])
        return gradient

    def minimize_ridge_regression(self, reg_param):

        start = time.time()
        data_length = 13
        current_w = np.ones(data_length)
        current_gradient = np.ones(data_length)
        print('norm', np.linalg.norm(current_w+current_w))
        grad_factor = 0.001
        print('grad factor', grad_factor)
        while(np.linalg.norm(current_gradient) > 2):
            current_gradient = self._calculate_gradient(
                current_w, reg_param)
            # print(current_gradient)
            # print(grad_factor*current_gradient)
            # print('current_w before:', current_w)
            current_w = np.subtract(current_w, grad_factor*current_gradient)
            # print('current_w after:', current_w)
            print('gradient step:', np.linalg.norm(current_gradient))
            print('current cost:', self._calculate_optimization_cost(current_w, reg_param))
            time.sleep(0.1)
        end = time.time()
        print('calculation time:', end - start)
        return current_w


def main():

    data = np.array([[0.06724, 0, 3.24, 0, 0.46, 6.333, 17.2, 5.2146, 4, 430, 16.9, 375.21, 7.34],
                     [0.06724, 0, 3.24, 0, 0.46, 6.333, 17.2, 5.2146, 4, 430, 16.9, 375.21, 7.34]])
    y = np.array([22.6, 22.6])
    rr = RidgeRegressor(y, data)
    optimal_weights = rr.minimize_ridge_regression(1)
    print(optimal_weights)
    # print(np.inner(optimal_weights, data))
    print("rmse:", rr.calculate_rsme(np.inner(optimal_weights, data)))


# for isolated testing purposes only:
if __name__ == "__main__":
    main()
