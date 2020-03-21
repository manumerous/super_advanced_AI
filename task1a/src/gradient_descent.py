import numpy as np
import time
import math

# Multidimensional gradient descent object
class GradientDescent():

    def __init__(self, y, x):
        self.y = y
        self.x = x
        self.data_length = len(y)
        self.grad_res = 0.00001

    def calculate_rsme(self, y_hat):
        sum = 0
        for i in range(self.data_length):
            x = (self.y[i] - y_hat[i])*(self.y[i] - y_hat[i])
            sum = sum + x
        rsme_variable = (1/len(self.y)*sum) ** 0.5
        return rsme_variable

    def _calculate_optimization_cost(self, w, reg_param):
        cost = 0
        for i in range(self.data_length):
            cost += (self.y[i] - np.inner(w, self.x[i, :]))*(self.y[i] -
                        np.inner(w, self.x[i, :])) + reg_param*np.linalg.norm(w)
        return cost

    # gradient calculation using the difference quotient
    def _calculate_gradient(self, w, reg_param):
        gradient = np.zeros(len(w))
        for i in range(len(w)):
            current_w = w
            current_w[i] += self.grad_res
            upper_cost = self._calculate_optimization_cost(current_w, reg_param)
            current_w[i] -= 2*self.grad_res
            lower_cost = self._calculate_optimization_cost(current_w, reg_param)
            gradient[i] = (upper_cost-lower_cost)/(2*self.grad_res)
            current_w[i] += self.grad_res
        return gradient

    def minimize_ridge_regression(self, reg_param):

        start = time.time()
        data_length = 13
        current_w = np.zeros(data_length)
        current_gradient = np.ones(data_length)
        learning_rate = 1
        print('grad factor', learning_rate)
        iteration = 0

        # gradient descent loop
        while(np.linalg.norm(learning_rate*current_gradient) > 0.0000000000001 and iteration < 1000):
            current_gradient = self._calculate_gradient(
                current_w, reg_param)
            new_w = np.subtract(current_w, learning_rate*current_gradient)
            new_cost = self._calculate_optimization_cost(new_w, reg_param)
            current_cost = self._calculate_optimization_cost(current_w, reg_param)
            if(new_cost > current_cost):
                while( new_cost > current_cost ):
                    learning_rate = learning_rate/2
                    new_w = np.subtract(current_w, learning_rate*current_gradient)
                    new_cost = self._calculate_optimization_cost(new_w, reg_param)
            else:
                learning_rate = learning_rate*1.1
            current_w = np.subtract(current_w, learning_rate*current_gradient)
            iteration += 1 
            # print(current_w)
            print(current_cost)
            print(iteration)
        end = time.time()
        print('calculation time:', end - start)
        return current_w


def main():

    data = np.array([[0.06724, 0, 3.24, 0, 0.46, 6.333, 17.2, 5.2146, 4, 430, 16.9, 375.21, 7.34],
                     [0.06724, 0, 3.24, 0, 0.46, 6.333, 17.2, 5.2146, 4, 430, 16.9, 375.21, 7.34]])
    y = np.array([22.6, 22.6])
    gd = GradientDescent(y, data)
    optimal_weights = gd.minimize_ridge_regression(1)
    print(optimal_weights)
    print("rmse:", gd.calculate_rsme(np.inner(optimal_weights, data)))


# for isolated testing purposes only:
if __name__ == "__main__":
    main()
