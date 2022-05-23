import os, pickle
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

# Get all treated user data in one dataframe
treated_user_data = os.listdir("./rawdatacorridor")
# read dataframe from csv
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

# train, test = train_test_split(big_data, test_size=0.2, random_state=42)

X = big_data.drop(columns=y_labels)
y = big_data[y_labels]

# X_test = test.drop(columns=y_labels)
# y_test = test[y_labels]

regressors = {
    "Ridge": Ridge(random_state=42),
    "MLP": MLPRegressor(random_state=42),
    "RF": RandomForestRegressor(random_state=42),
    "KNR": KNeighborsRegressor(),
}

for name, model in regressors.items():
    model.fit(X, y)
    print(f"{name} Score : ", model.score(X, y))
    with open(f"./models/model_{name}.pkl", "wb") as f:
        pickle.dump(model, f)
