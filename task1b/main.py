#    _____         .__
#   /     \ _____  |__| ____
#  /  \ /  \\__  \ |  |/    \
# /    Y    \/ __ \|  |   |  \
# \____|__  (____  /__|___|  /
#         \/     \/        \/
'''The main file is used to start up the program and initialize the needed objects'''

__author__ = "Lukas Reichart"
__maintainer__ = "Lukas Reichart"
__license__ = "GPL"

import pandas as pd
import math as math
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

def main():
    # Read the data from the train.csv file (into a pandas data object)
    data = pd.read_csv('./data/train.csv')
    yData = data["y"]
    xData = data.drop(['Id','y'],1)

    # Transform the data according to the model
    print(f'The input data looks like: {data.head()}\n')

    headers = xData.columns

    xData[['x6','x7','x8','x9','x10']] = data[headers].applymap(lambda x: x*x)
    xData[['x11','x12','x13','x14','x15']] = data[headers].applymap(math.exp)
    xData[['x16','x17','x18','x19','x20']] = data[headers].applymap(math.cos)
    xData['x21'] = 1

    print(f'The feature transformed data looks like: {xData.head()}')

    clf = linear_model.SGDRegressor()
    clf.fit(xData.to_numpy(),yData.to_numpy())

    predict = clf.predict(xData)

    print(f'Linear Coefficients: {clf.coef_}\n')

    print(f'Mean squared error: {mean_squared_error(yData, predict)}\n')

    print(f'Root mean squared error: {math.sqrt(mean_squared_error(yData, predict))}\n')

    # Write the CSV result
    pd.DataFrame(clf.coef_).to_csv('./data/result.csv', header=False, index=False)

    return


if __name__ == "__main__":

    main()
