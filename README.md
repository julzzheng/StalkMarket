# Stalk Market Tracker
Animal Crossing New Horizon - Stalk Market Tracker and Predictor

In the game Animal Crossing New Horizon there is the Stalk Market feature where one can buy turnips on any Sunday morning and sell them at any day prior to the next Sunday.
Hereby the selling price of the turnips changes twice a day and the goal is to sell the turnips with the highest possible profit, but how the price will change cannot be foreseen in advance.

However, based on the [reddit post](https://www.reddit.com/r/AnimalCrossing/comments/fr2cuq/guide_how_to_beat_the_stock_turnip_market_playing/) of [ssyl9](https://www.reddit.com/user/ssyl9/), there are patterns that indicate how the price might move in the current week.

Furthermore, the twitter user [@KnightCarmine](https://twitter.com/KnightCarmine) illustrated in his [tweet](https://twitter.com/KnightCarmine/status/1244392945056276482) how the price development for the Stalk Market could look like.

Lastly, in [@_Ninji](https://twitter.com/_Ninji)'s tweet he showed in his [findings](https://twitter.com/_Ninji/status/1244818665851289602) the code responsible for the turnip prices.

<div style="text-align:center"><img src="https://pbs.twimg.com/media/EUT4ZTWVAAA7S9V?format=jpg" width="450">
<img src="https://pbs.twimg.com/media/EUT4aS1UEAAz73u?format=jpg" width="450"></div>

## Getting Started

### Prerequisites
- [Python 3 (with pip)](https://www.python.org/downloads/)

### Install Requirements
````
pip install -r requirements.txt
````

### Run Tracker

Parameters:
1. ``price``: for how many bells can be selled for, e.g 46
2. ``time``: current time to track the price for, e.g. mon_am
3. ``user``: name of the person to track the current price for, e.g. james

````
python src/track.py --price=46 --time=mon_am --user=james
````

### Run Prediction

Parameters:
1. ``new_model``: whether a new model should be trained or not (optional)

````
python src/predict.py --new_model=True
````

### Generate training data

Parameters:
1. ``n_samples``: number of samples per pattern should be generated (optional), e.g. 1000 results in 4000 samples

````
python src/generate.py --n_samples=1000
````

## License
This project is licensed under the Apache 2.0 License - see the [LICENSE.md](https://github.com/kemoyin/StalkMarket/blob/master/LICENSE) file for details



