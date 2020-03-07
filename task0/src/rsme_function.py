def rsme_function (y, y_hat):
    rownumber = len(y)
    for element in y:
        x = (element - y_hat)*(element - y_hat)
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
