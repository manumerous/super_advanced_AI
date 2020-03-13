import numpy as np


def calculate_rsme(y, y_hat):
    sum = 0
# rsme calculation requires input of two lists (y and y_hat values)
    for i in range(len(y)):
        x = (y[i] - y_hat[i])*(y[i] - y_hat[i])
        sum = sum + x
    rsme_variable = (1/len(y)*sum) ** 0.5
    return rsme_variable


def _calculate_optimization_cost(y, x, w, regularization_parameters):
    cost = 0
    for i in range(len(y)):
        cost = cost + (y[i] - np.inner(w, x[i, :]))*(y[i] -
                                                     np.inner(w, x[i, :])) + np.linalg.norm(w)
    return cost


def minimize_ridge_regression(y, x, regularization_parameter):
    data_length = 13
    current_weights = np.ones(data_length)
    gradient_step = 0.1
    gradient_dir = 1
    cost_gradient = 1
    while(abs(cost_gradient) > 0.0000000001):
        for i in range(data_length):
            
            current_cost = _calculate_optimization_cost(
                y, x, current_weights, regularization_parameter)
            new_weights = current_weights
            new_weights[i] = new_weights[i] + gradient_step*gradient_dir
            new_cost = _calculate_optimization_cost(
                y, x, new_weights, regularization_parameter)
            cost_gradient = (new_cost-current_cost)*gradient_dir/gradient_step
            if (new_cost < current_cost):
                current_weights = new_weights
            else:
                gradient_dir = gradient_dir*(-1)
        print('cost gradient:', cost_gradient)
        gradient_step = abs(0.1*cost_gradient/10000)
        print('gradient step:', gradient_step)
    return current_weights

def main():
    data = np.array([[0.06724, 0, 3.24, 0, 0.46, 6.333, 17.2, 5.2146, 4, 430, 16.9, 375.21, 7.34],
            [0.06724, 0, 3.24, 0, 0.46, 6.333, 17.2, 5.2146, 4, 430, 16.9, 375.21, 7.34]])
    y = np.array([22.6, 22.6])
    print(len(y))
    optimal_weights = minimize_ridge_regression(y, data, 1)
    print(optimal_weights)
    # print(np.inner(optimal_weights, data))
    print("rsme:", calculate_rsme(y, np.inner(optimal_weights, data)))

# for isolated testing purposes only:
if __name__== "__main__":
    main()
    

