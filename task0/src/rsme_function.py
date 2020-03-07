def rsme_function (y, y_hat):
    rownumber = len(y)
    sum = 0

# rsme calculation requires input of two lists (y and y_hat values)
    for i in range(len(y)):
        x = (y[i] - y_hat[i])*(y[i] - y_hat[i])
        sum = sum + x
    rsme_variable = (1/rownumber*sum) ** 0.5
    return rsme_variable

#to test function: run file as main file, otherwise use function in other file 
if __name__ == "main":
    data = [3,3,3,3,3,3,3]
    print(data)
    real_mean = 0
    error = rsme_function(data, real_mean)
    print(error)
