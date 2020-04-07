import pandas as pd
import matplotlib.pyplot as plt
import datetime
import os
from argparse import ArgumentParser

stalk_time = ['mon_am', 'mon_pm', 'tue_am', 'tue_pm', 'wed_am', 'wed_pm', 'thu_am', 'thu_pm', 'fri_am', 'fri_pm',
              'sat_am', 'sat_pm']
markers = ['s', 'o', 'v', '^', 'v', '.', 'p', '*', 'h', 'D']


def track_stalk(path, price, time, user):
    current_week = datetime.date.today().isocalendar()[1]

    try:
        if not os.path.exists(path):
            with open(path, 'w'):
                pass
            file = pd.DataFrame(columns=['week', 'time', 'user', 'price'])
        else:
            file = pd.read_csv(path)

        if ((file.week == current_week) & (file.user == user) & (file.time == stalk_time.index(time))).any():
            raise IOError('Entry already exists for given user and time.')
    except IOError as ex:
        print(ex)

    file = file.append({'week': current_week, 'time': stalk_time.index(time), 'user': user, 'price': price},
                       ignore_index=True)
    file.to_csv(path, index=False)


def plot_prices(path):
    data = pd.read_csv(path)
    current_week = datetime.date.today().isocalendar()[1]

    entries = data[data.week == current_week]
    users = entries.user.unique().tolist()

    for i in users:
        user_entries = entries[entries.user == i]
        plt.plot(user_entries.time.tolist(), user_entries.price.tolist(), lw=2, label='{}'.format(i),
                 marker=markers[users.index(i)])

    plt.ylim(0, 700)
    plt.xlim(0, 12)
    plt.xticks(range(12), stalk_time, rotation=-45)
    plt.xlabel("Timeframe")
    plt.ylabel("Price")
    plt.legend(loc="best")
    plt.title("The Stalk Market for week %s" % current_week)

    plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--price", required=True, type=int, help="price for stalks", )
    parser.add_argument("--time", required=True, type=str, help="for when the price should be tracked",
                        choices=stalk_time)
    parser.add_argument("--user", required=True, type=str, help="for whom the price should be tracked")
    args = parser.parse_args()

    prices_path = 'data/stalk_prices.csv'
    track_stalk(prices_path, args.price, args.time, args.user)
    plot_prices(prices_path)
