__author__ = "Lukas Reichart"

import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV


# Helper function that generates a one-hot-encoder
# One Hot-Encoding
# Transform the Letters into vectors of the format [0,0,1,0...,0] for letter C etc.
# put the vectors for different letters after each other
def get_one_hot_encoder(data):
    onehot_encoder = OneHotEncoder(handle_unknown='ignore')
    enc = onehot_encoder.fit(data.to_numpy())
    return enc

def get_best_classifier(xData, yData):
    # create the classifier
    mlp = MLPClassifier(solver='adam', alpha=1e-5, learning_rate='constant', activation='relu',
                    # hidden_layer_sizes=(5,2),
                    random_state=1)
    
    # First iteration (2_result.csv)
    # {'activation': 'relu', 'alpha': 1e-05, 'learning_rate': 'constant', 'solver': 'adam'}

    # Second iteration (did not run yet)
    #{'activation': 'relu', 'alpha': 1e-08, 'learning_rate': 'constant', 'solver': 'adam'}
    
    mlp.fit(xData, yData)
    return mlp


def search_best_classifier(xData, yData):
    mlp = MLPClassifier(random_state=1)

    # Evaluate
    # Different solvers: lbfgs, sgd, adam
    # L2 penalty alpha: 10e-8 to 1
    # for sgd: learning rate
    # max_iter: default 200, in 50er steps up to 500
    # turn verbose on for better output
    parameter_space = {
        'hidden_layer_sizes': [(50,50,50),(50,100,50),(100,)],
        # TODO: also use hidden layer sizes
        # 'activation': ['tanh', 'relu'],
        'activation': ['relu'],
        # 'solver': ['sgd', 'adam'],
        'solver': ['adam'],
        # 'alpha': [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        'alpha': [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3],
        'learning_rate': ['constant']
    }

    clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3, verbose=True)
    clf.fit(xData, yData)

    print('Best parameters found: \n', clf.best_params_)

    return clf


def main():
    # read the data
    data = pd.read_csv('./data/train.csv')

    xData_raw = data['Sequence']
    xData = xData_raw.apply(lambda x: pd.Series(list(x)))
    yData = data['Active']

    # get the one-hot encoding
    enc = get_one_hot_encoder(xData)

    # encode the data into one_hot
    xData_one_hot = enc.transform(xData.to_numpy())

    # Here you can switch between using the gridsearch or just using the classifier that we found
    # mlp = search_best_classifier(xData_one_hot, yData)
    mlp = get_best_classifier(xData_one_hot, yData)

    # predict on the data we just used
    print("Predicting on the data we just trained with:")
    train_predicted = mlp.predict(xData_one_hot)
    print("F1 score", f1_score(data['Active'], train_predicted))

    print("Predicting on the data")
    test_data = pd.read_csv('./data/test.csv')
    transformed_test_data = test_data['Sequence'].apply(lambda x: pd.Series(list(x)))

    print("Transformed test data generated")

    one_hot_encoded_test_data = enc.transform(transformed_test_data.to_numpy())
    print("Doing the actual prediction")
    test_predicted = mlp.predict(one_hot_encoded_test_data)

    pd.DataFrame(test_predicted).to_csv('./data/result.csv', header=False, index=False)


if __name__ == "__main__":
    main()
