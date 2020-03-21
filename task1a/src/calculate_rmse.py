# _________        .__               .__          __           __________    _____    ____________________
# \_   ___ \_____  |  |   ____  __ __|  | _____ _/  |_  ____   \______   \  /     \  /   _____/\_   _____/
# /    \  \/\__  \ |  | _/ ___\|  |  \  | \__  \\   __\/ __ \   |       _/ /  \ /  \ \_____  \  |    __)_ 
# \     \____/ __ \|  |_\  \___|  |  /  |__/ __ \|  | \  ___/   |    |   \/    Y    \/        \ |        \
#  \______  (____  /____/\___  >____/|____(____  /__|  \___  >  |____|_  /\____|__  /_______  //_______  /
#         \/     \/          \/                \/          \/          \/         \/        \/         \/ 
        
'''Contains a function to calculate the root mean square error'''

__author__ = "Jsmea Hug"
__maintainer__ = "Jsmea Hug, Manuel Galliker"
__license__ = "GPL"

# rsme calculation requires input of two lists (y and y_hat values)
def calculate_rmse(y, y_hat):
    rownumber = len(y)
    sum = 0
    for i in range(len(y)):
        y_hat
        x = (y[i] - y_hat[i])*(y[i] - y_hat[i])
        sum = sum + x
    rsme_variable = (1/rownumber*sum) ** 0.5
    return rsme_variable


# to test function: run file as main file, otherwise use function in other file
if __name__ == "main":
    data = [3, 3, 3, 3, 3, 3, 3]
    print(data)
    real_mean = 0
    error = rsme_function(data, real_mean)
    print(error)
