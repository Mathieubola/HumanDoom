from os import kill
from vizdoom import DoomGame
import time, json, pickle, sklearn
import numpy as np
import pandas as pd

game = DoomGame()

game.load_config("deadly_corridor.cfg")
game.set_labels_buffer_enabled(True)
game.set_window_visible(False)
game.init()

available_game_var = ["AMMO2", "HEALTH", "ARMOR", "VELOCITY_X", "VELOCITY_Y", "VELOCITY_Z", "KILLCOUNT", "DAMAGE_TAKEN", "ANGLE"]
available_buttons = ["MOVE_LEFT ", "MOVE_RIGHT ", "ATTACK ", "MOVE_FORWARD", "MOVE_BACKWARD", "TURN_LEFT", "TURN_RIGHT"]

col = [
    *available_game_var,
    "height_1", "width_1", "lx_1", "ly_1", "enemie_present_1",
    "height_2", "width_2", "lx_2", "ly_2", "enemie_present_2",
]

models = {
    "MLP": "./models/model_MLP.pkl",
    "RF": "./models/model_RF.pkl",
    "KNR": "./models/model_KNR.pkl",
}

for model_name, model_path in models.items():
# import model
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    thresh = [0.5] # np.arange(0.1, 0.4, 0.025)
    best_thresh = {
        "thresh": None,
        "live_count": -1,
        "kill_count": -1,
    }
    for th in thresh:
        live_count = 0
        kill_count = 0

        for _ in range(20):
            game.new_episode()
            while not game.is_episode_finished():
                state = game.get_state()

                n      = state.number
                labels = state.labels
                vars   = state.game_variables

                lab = [[i.height, i.width, i.x, i.y, i.height * i.width] for i in labels if i.object_name != "DoomPlayer"]
                lab.sort(key=lambda x: x[4], reverse=True)

                lab1 = lab[0][0:4]+[True] if len(lab) > 0 else [0]*4+[False]
                lab2 = lab[1][0:4]+[True] if len(lab) > 1 else [0]*4+[False]

                X = [*vars, *lab1, *lab2]

                df = pd.DataFrame(columns=col, data=[X])
                # Keep only useful columns

                y = model.predict(df)
                # y = [i > th for i in y[0]]
                y = [i == 1 for i in y[0]]

                r = game.make_action(y)
                # time.sleep(1/60)

                r = game.get_last_reward()
            
            if not game.is_player_dead() and n < 1000:
                live_count += 1
            
            kill_count += int(vars[6])

        kill_count /= 20
        live_count /= 20

        if live_count * 4 + kill_count > best_thresh["live_count"] * 4 + best_thresh["kill_count"]:
            best_thresh["live_count"] = live_count
            best_thresh["kill_count"] = kill_count
            best_thresh["thresh"] = th
        

    print(model_name)
    print(best_thresh)

game.close()
