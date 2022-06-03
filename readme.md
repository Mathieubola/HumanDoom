# HumanDoom

The goal of this project is to evaluate how to make an AI that act like a human while playing doom.

To realise this objectif, we created 4 groups of scripts that can be slightly edited and lauched to create and compare models.

## Get user data

Launching the script `1-getUserData.py` will create a Doom client that will be playable. The user can then play some games that will be saved in the folder `rawdatacorridor`. If you want to try it for yourself I'd advise to empty this folder before recording the data so that the AI trains only on your play-style

## Make model

Launching the script `2-makemodel.py` will read the user generated data and create multiple model that will be later used to predict the user's action. `2-makemodeltrain.py` will do the same but will only create MLP models while varying the size of the training dataset.

## Get data

In order to get the models performance out from the models, we will make them play each 20 rounds per threshold tested (the displayed data will be the best performing one). This data can be get for the models that vary the size of the training dataset by using the `3-gettraindata.py` script.

## Play model

Now that we have the models, we can watch them pay the game. This can be used to juge the human-likeness of the models by real human. This can be done by using the `4-playmodel.py` script.