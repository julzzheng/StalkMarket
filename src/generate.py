import pandas as pd
import random
import os
from math import ceil
from nltk.corpus import names
from joblib import Parallel, delayed
from argparse import ArgumentParser

user_names = [name.lower() for name in names.words()]


def generate_random_sample(wk, pr, pt):
    sunday_price = random.randrange(90, 111)
    username = random.choice(user_names)
    pr.extend([
        {'week': wk, 'time': 0, 'user': username, 'price': sunday_price},
        {'week': wk, 'time': 1, 'user': username, 'price': sunday_price}
    ])
    decPhaseLen1 = random.randrange(2, 4)
    decPhaseLen2 = 5 - decPhaseLen1
    hiPhaseLen1 = random.randrange(0, 7)
    hiPhaseLen2and3 = 7 - hiPhaseLen1
    hiPhaseLen3 = random.randrange(0, hiPhaseLen2and3)
    rate = random.uniform(0.8, 0.6)
    for j in range(2, 14):
        if j < hiPhaseLen1:
            price = ceil(random.uniform(0.9, 1.4) * sunday_price)
            rate = random.uniform(0.8, 0.6)
        elif j < hiPhaseLen1 + decPhaseLen1:
            price = ceil(rate * sunday_price)
            rate -= 0.04
            rate -= random.uniform(0, 0.06)
        elif j < hiPhaseLen1 + decPhaseLen1 + hiPhaseLen2and3 - hiPhaseLen3:
            price = ceil(random.uniform(0.9, 1.4) * sunday_price)
            rate = random.uniform(0.8, 0.6)
        elif j < hiPhaseLen1 + decPhaseLen1 + hiPhaseLen2and3 + decPhaseLen2 - hiPhaseLen3:
            price = ceil(rate * sunday_price)
            rate -= 0.04
            rate -= random.uniform(0, 0.06)
        else:
            price = ceil(random.uniform(0.9, 1.4) * sunday_price)
        pr.append({'week': wk, 'time': j, 'user': username, 'price': price})
    pt.append({'week': wk, 'user': username, 'class': 0})
    return pr, pt


def generate_decreasing_sample(wk, pr, pt):
    sunday_price = random.randrange(90, 111)
    username = random.choice(user_names)
    pr.extend([
        {'week': wk, 'time': 0, 'user': username, 'price': sunday_price},
        {'week': wk, 'time': 1, 'user': username, 'price': sunday_price}
    ])
    rate = 0.9
    rate -= random.uniform(0, 0.05)
    for j in range(2, 14):
        price = ceil(rate * sunday_price)
        rate -= 0.03
        rate -= random.uniform(0, 0.02)
        pr.append({'week': wk, 'time': j, 'user': username, 'price': price})
    pt.append({'week': wk, 'user': username, 'class': 1})
    return pr, pt


def generate_small_spike_sample(wk, pr, pt):
    sunday_price = random.randrange(90, 111)
    username = random.choice(user_names)
    pr.extend([
        {'week': wk, 'time': 0, 'user': username, 'price': sunday_price},
        {'week': wk, 'time': 1, 'user': username, 'price': sunday_price}
    ])
    peakStart = random.randrange(2, 10)
    rate = random.uniform(0.9, 0.4)
    for j in range(2, 14):
        if j < peakStart:
            price = ceil(rate * sunday_price)
            rate -= 0.03
            rate -= random.uniform(0, 0.02)
        elif j < peakStart + 2:
            price = ceil(random.uniform(0.9, 1.4) * sunday_price)
            rate = random.uniform(1.4, 2.0)
        elif j < peakStart + 3:
            price = ceil(random.uniform(1.4, rate) * sunday_price) - 1
        elif j < peakStart + 4:
            price = ceil(rate * sunday_price)
        elif j < peakStart + 5:
            price = ceil(random.uniform(1.4, rate) * sunday_price) - 1
            rate = random.uniform(0.9, 0.4)
        else:
            price = ceil(rate * sunday_price)
            rate -= 0.03
            rate -= random.uniform(0, 0.02)
        pr.append({'week': wk, 'time': j, 'user': username, 'price': price})
    pt.append({'week': wk, 'user': username, 'class': 2})
    return pr, pt


def generate_large_spike_sample(wk, pr, pt):
    sunday_price = random.randrange(90, 111)
    username = random.choice(user_names)
    pr.extend([
        {'week': wk, 'time': 0, 'user': username, 'price': sunday_price},
        {'week': wk, 'time': 1, 'user': username, 'price': sunday_price}
    ])
    peakStart = random.randrange(3, 10)
    rate = random.uniform(0.9, 0.85)
    for j in range(2, 14):
        if j < peakStart:
            price = ceil(rate * sunday_price)
            rate -= 0.03
            rate -= random.uniform(0, 0.02)
        elif j < peakStart + 1:
            price = ceil(random.uniform(0.9, 1.4) * sunday_price)
        elif j < peakStart + 2:
            price = ceil(random.uniform(1.4, 2.0) * sunday_price)
        elif j < peakStart + 3:
            price = ceil(random.uniform(2.0, 6.0) * sunday_price)
        elif j < peakStart + 4:
            price = ceil(random.uniform(1.4, 2.0) * sunday_price)
        elif j < peakStart + 5:
            price = ceil(random.uniform(0.9, 1.4) * sunday_price)
        else:
            price = ceil(random.uniform(0.4, 0.9) * sunday_price)
        pr.append({'week': wk, 'time': j, 'user': username, 'price': price})
    pt.append({'week': wk, 'user': username, 'class': 3})
    return pr, pt


def generate_samples(wk):
    pr = []
    pt = []
    generate_random_sample(wk, pr, pt)
    generate_decreasing_sample(wk, pr, pt)
    generate_small_spike_sample(wk, pr, pt)
    generate_large_spike_sample(wk, pr, pt)

    return pr, pt


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--n_samples", required=False, type=int,
                        help="how many samples per pattern should be generated", default=10000)
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

    results = [generate_samples(i) for i in range(args.n_samples)]

    prices, patterns = zip(*results)

    df_prices = df_prices.append([price for l in prices for price in l], ignore_index=True)
    df_patterns = df_patterns.append([pattern for l in patterns for pattern in l], ignore_index=True)

    df_prices.to_csv(prices_path, index=False)
    df_patterns.to_csv(patterns_path, index=False)
