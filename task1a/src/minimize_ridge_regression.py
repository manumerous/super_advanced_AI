import numpy as np

def calculate_rsme(y, y_hat):
    rownumber = len(y)
    sum = 0

# rsme calculation requires input of two lists (y and y_hat values)
    for i in range(len(y)):
        x = (y[i] - y_hat[i])*(y[i] - y_hat[i])
        sum = sum + x
    rsme_variable = (1/rownumber*sum) ** 0.5
    return rsme_variable

def _calculate_optimization_cost(y, x, w, regularization_parameters):
    cost = 0
    for i in range(len(y)):
        cost = cost + (y[i] - np.inner(w, x[i, :]))*(y[i] -
                np.inner(w, x[i, :])) + np.linalg.norm(w, 'fro')
    return cost


def minimize_ridge_regression(y, x, regularization_parameter):

    data_length = 13
    current_weights = np.ones(data_length)
    gradient_step = 0.05
    gradient_dir = 1
    cost_gradient = 1

    while(abs(cost_gradient)< 0.05):
        for i in range(data_length):
            current_cost = _calculate_optimization_cost(y,x, current_weights, regularization_parameter)
            new_weights = current_weights
            new_weights[i] = new_weights[i] + gradient_step*gradient_dir
            new_cost = _calculate_optimization_cost(y,x, new_weights, regularization_parameter)
            
            cost_gradient = (new_cost-current_cost)*gradient_dir/gradient_step

            if (new_cost > current_cost):
                current_weights = new_weights
            else:
                gradient_dir = gradient_dir*(-1)
                
        return 0

    return current_weights


# for isolated testing purposes only:
if __name__ == "main":
    data = [3, 3, 3, 3, 3, 3, 3]
