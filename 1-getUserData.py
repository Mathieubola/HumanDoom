from vizdoom import DoomGame, Mode
import time, json
import numpy as np
import pandas as pd

game = DoomGame()

game.load_config("deadly_corridor.cfg")
game.set_labels_buffer_enabled(True)
# Enable freelook in game
# game.add_game_args("+freelook 1")
game.set_mode(Mode.SPECTATOR)

available_game_var = ["AMMO2", "HEALTH", "ARMOR", "VELOCITY_X", "VELOCITY_Y", "VELOCITY_Z", "KILLCOUNT", "DAMAGE_TAKEN", "ANGLE"]
available_buttons = ["MOVE_LEFT ", "MOVE_RIGHT ", "ATTACK ", "MOVE_FORWARD", "MOVE_BACKWARD", "TURN_LEFT", "TURN_RIGHT"]

col = [
    *available_game_var,
    "height_1", "width_1", "lx_1", "ly_1", "enemie_present_1",
    "height_2", "width_2", "lx_2", "ly_2", "enemie_present_2",
    *available_buttons
]

iteration = int(input("Number of iteration : "))
game.init()
input()
for iter in range(iteration):
    game.new_episode()
    output = []
    while not game.is_episode_finished():
        state = game.get_state()
        game.advance_action()

        n           = state.number
        labels      = state.labels
        vars        = state.game_variables

        a = game.get_last_action()

        lab = [[i.height, i.width, i.x, i.y, i.height * i.width] for i in labels if i.object_name != "DoomPlayer"]
        lab.sort(key=lambda x: x[4], reverse=True)

        lab1 = lab[0][0:4]+[True] if len(lab) > 0 else [0]*4+[False]
        lab2 = lab[1][0:4]+[True] if len(lab) > 1 else [0]*4+[False]

        X = [*vars, *lab1, *lab2, *a]

        output.append(X)

    pd.DataFrame(columns=col, data=output).to_csv(f"./rawdatacorridor/output-{time.time_ns()}.csv", index=False)

game.close()
