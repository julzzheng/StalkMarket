# Stalk Market Tracker
Animal Crossing New Horizon - Stalk Market Tracker


# Usage

## Install Requirements:
``
pip install -r requirements.txt
``

## Run Tracker:

Parameters:
1. ``price``: for how many bells one stalk can be selled for, e.g 46
2. ``time``: current time to track the price for, e.g. mon_am
3. ``user``: name of the person to track the current price for, e.g. james

``
python src/track.py --price=46 --time=mon_am --user=james
``

## Run Prediction:

Parameters:
1. ``new_model``: whether a new model should be trained or not (optional)

``
python src/predict.py --new_model=True
``

## Generate training data:


``
python src/generate.py
``

