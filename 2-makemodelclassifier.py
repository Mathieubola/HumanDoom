import os, pickle
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings("ignore")

treated_user_data = os.listdir("./rawdatacorridor")

y_labels = [
    "MOVE_LEFT ", "MOVE_RIGHT ", "ATTACK ", "MOVE_FORWARD", "MOVE_BACKWARD", "TURN_LEFT", "TURN_RIGHT"
]

big_data = None

for i in treated_user_data:
    data = pd.read_csv(f"./rawdatacorridor/{i}")

    if big_data is None:
        big_data = data
    else:
        big_data = pd.concat([big_data, data])


X = big_data.drop(columns=y_labels)
y = big_data[y_labels]

train, test = train_test_split(big_data, test_size=0.2, random_state=42)
X_train = train.drop(columns=y_labels)
y_train = train[y_labels]
X_test = test.drop(columns=y_labels)
y_test = test[y_labels]

regressors = {
    "MLP": MLPClassifier(random_state=42),
    "RF": RandomForestClassifier(random_state=42),
    "KNR": KNeighborsClassifier(),
}

param = {
    "MLP": {
        "hidden_layer_sizes": [(100,), (50, 50), (25, 50, 25)],
        "activation": ["identity", "logistic", "tanh", "relu"],
    },
    "RF": {
        "n_estimators": [10, 50, 100, 200],
        "max_depth": [2, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
    },
    "KNR": {
        "n_neighbors": [1, 5, 10, 20],
        "weights": ["uniform", "distance"],
    },
}

param = {
    "MLP": {
        "hidden_layer_sizes": [(100,)],
        "activation": ["logistic"],
    },
    "RF": {
        "n_estimators": [200],
        "max_depth": [20],
        "min_samples_split": [2],
    },
    "KNR": {
        "n_neighbors": [1],
        "weights": ["uniform"],
    },
}

# Best Parameters
# param = {
#     "MLP": {'activation': 'logistic', 'hidden_layer_sizes': (100,)},
#     "RF": {'max_depth': 20, 'min_samples_split': 2, 'n_estimators': 200},
#     "KNR": {'n_neighbors': 1, 'weights': 'uniform'},
# }

best_models = {}

for regressor_name, regressor in regressors.items():
    print(f"Grid Search {regressor_name}...")
    gscv = GridSearchCV(regressor, param_grid=param[regressor_name])
    gscv.fit(X_train, y_train)
    print(f"Best parameters for {regressor_name} : {gscv.best_params_}")
    best_models[regressor_name] = gscv.best_estimator_

for name, model in best_models.items():
    # model.fit(X_train, y_train)
    print(f"{name} Score : ", model.score(X_test, y_test))
    with open(f"./models/model_{name}.pkl", "wb") as f:
        pickle.dump(model, f)
