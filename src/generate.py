import pandas as pd
import random
import os
from nltk.corpus import names
from argparse import ArgumentParser

user_names = [name.lower() for name in names.words()]


def generate_random_sample(week, sunday_price, prices, patterns):
    username = random.choice(user_names)
    random_choice = random.randrange(0,2)
    price = [
        random.randrange(int(sunday_price*.91), 140),
        random.randrange(int(sunday_price*.6), int(sunday_price*.8))
    ][random_choice]
    for j in range(12):
        prices = prices.append({'week': week, 'time': j, 'user': username, 'price': price},
                               ignore_index=True)
        price = random.randrange(40, 200)
    patterns = patterns.append({'week': week, 'user': username, 'class': 0},
                               ignore_index=True)

    return prices, patterns


def generate_decreasing_sample(week, sunday_price, prices, patterns):
    username = random.choice(user_names)
    price = random.randrange(int(sunday_price * .85), int(sunday_price * .91))
    for j in range(12):
        prices = prices.append({'week': week, 'time': j, 'user': username, 'price': price},
                               ignore_index=True)
        price = random.randrange(int(price * .9), price)

    patterns = patterns.append({'week': week, 'user': username, 'class': 1},
                               ignore_index=True)

    return prices, patterns


def generate_small_spike_sample(week, sunday_price, prices, patterns):
    username = random.choice(user_names)
    random_choice = random.randrange(0,5)
    price = [
        random.randrange(int(sunday_price*.85), int(sunday_price*.91)),
        random.randrange(int(sunday_price*.8), int(sunday_price*.85)),
        random.randrange(int(sunday_price * .91), 140),
        random.randrange(int(sunday_price * .6), int(sunday_price * .8)),
        random.randrange(45, int(sunday_price * .60))
    ][random_choice]
    low = random.randrange(3, 5)
    high = low + 2

    for j in range(12):
        prices = prices.append({'week': week, 'time': j, 'user': username, 'price': price},
                               ignore_index=True)

        if low < j <= high:
            price = random.randrange(int(price * 1.5), int(price * 2))
        else:
            price = random.randrange(int(price * .9), price)

    patterns = patterns.append({'week': week, 'user': username, 'class': 2},
                               ignore_index=True)

    return prices, patterns


def generate_camel_hump_sample(week, sunday_price, prices, patterns):
    username = random.choice(user_names)
    random_choice = random.randrange(0,2)
    price = [
        random.randrange(int(sunday_price*.85), int(sunday_price*.91)),
        random.randrange(int(sunday_price*.6), int(sunday_price*.8))
    ][random_choice]
    wave_1 = random.randrange(3, 5)
    wave_3 = wave_1 + 3

    for j in range(12):
        prices = prices.append({'week': week, 'time': j, 'user': username, 'price': price},
                               ignore_index=True)
        if j <= wave_1 + 1:
            price = random.randrange(int(price * .9), int(price * 1.1))
        elif wave_1 + 1 < j <= wave_3:
            price = random.randrange(price*2, price * 3)
        else:
            price = random.randrange(int(price * .5), int(price * .75))

    patterns = patterns.append({'week': week, 'user': username, 'class': 3},
                               ignore_index=True)
    return prices, patterns


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--n_samples", required=False, type=int,
                        help="how many samples per pattern should be generated", default=1000)
    args = parser.parse_args()

    prices_path = 'data/stalk_prices_samples.csv'
    patterns_path = 'data/stalk_patterns_samples.csv'
    if not os.path.exists(prices_path):
        with open(prices_path, 'w'):
            pass
    if not os.path.exists(patterns_path):
        with open(patterns_path, 'w'):
            pass

    df_prices = pd.DataFrame(columns=['week', 'time', 'user', 'price'])
    df_patterns = file = pd.DataFrame(columns=['week', 'user', 'class'])

    for i in range(args.n_samples):
        sun = random.randrange(90, 111)
        df_prices, df_patterns = generate_random_sample(i, sun, df_prices, df_patterns)
        df_prices, df_patterns = generate_decreasing_sample(i, sun, df_prices, df_patterns)
        df_prices, df_patterns = generate_small_spike_sample(i, sun, df_prices, df_patterns)
        df_prices, df_patterns = generate_camel_hump_sample(i, sun, df_prices, df_patterns)

    df_prices.to_csv(prices_path, index=False)
    df_patterns.to_csv(patterns_path, index=False)
