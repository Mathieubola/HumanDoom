import os, pickle
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

treated_user_data = os.listdir("./rawdatacorridor")

y_labels = [
    "MOVE_LEFT ", "MOVE_RIGHT ", "ATTACK ", "MOVE_FORWARD", "MOVE_BACKWARD", "TURN_LEFT", "TURN_RIGHT"
]

for size in [1, 5, 10, 20, 50, 100]:
    big_data = None
    for i in treated_user_data[:size]:
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

    regressor = MLPRegressor(
        random_state=42,
        activation='logistic',
        hidden_layer_sizes = (100,)
    )

    regressor.fit(X_train, y_train)

    print(f"{size} Score : ", regressor.score(X_test, y_test))
    with open(f"./models_training_size/model_{size}.pkl", "wb") as f:
        pickle.dump(regressor, f)
