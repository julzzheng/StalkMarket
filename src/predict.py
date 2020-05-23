import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from argparse import ArgumentParser
import datetime
import os
import pickle
import random

stalk_time = ['sun_am', 'sun_pm', 'mon_am', 'mon_pm', 'tue_am', 'tue_pm', 'wed_am', 'wed_pm', 'thu_am', 'thu_pm',
              'fri_am', 'fri_pm', 'sat_am', 'sat_pm']
pattern = ['Random', 'Decreasing', 'Small Spike', 'Large Spike']


# pattern = ['Random', 'Decreasing', 'Spike']


def hyperparameter_optimization(x, y, nj):
    pipeline = Pipeline([
        ('select', SelectKBest()),
        ('clf', KNeighborsClassifier()),
    ])
    parameters = {
        'clf__n_neighbors': np.arange(1, 51, 2),
        'clf__weights': ['uniform', 'distance'],
        'clf__algorithm': ['auto'],
        'clf__leaf_size': np.arange(5, 65, 1),
        'clf__p': [1, 2],
        'select__score_func': [chi2],
        'select__k': ['all'],
    }
    stratified_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

    random_search = RandomizedSearchCV(pipeline, parameters, scoring='balanced_accuracy', cv=stratified_fold, n_jobs=nj,
                                       verbose=1)
    random_fit = random_search.fit(x, y.values.ravel())
    print("Best score: %0.8f" % random_fit.best_score_)
    print("Best parameters set:")
    best_parameters = random_fit.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    return random_fit.best_estimator_


def make_prediction(p, m):
    data = pd.read_csv(p)
    current_week = datetime.date.today().isocalendar()[1]

    entries = data[data.week == current_week]
    users = entries.user.unique().tolist()

    for i in users:
        features = pd.DataFrame(dict.fromkeys(stalk_time), index=[0])

        user_entries = entries[entries.user == i]

        for time in user_entries.time.unique():
            current_entry = user_entries[user_entries.time == time]
            features.loc[0, [stalk_time[time]]] = current_entry.price.values[0]
        features = features.fillna(-1)

        print('The prediction for %s is a %s pattern.' % (i, pattern[int(m.predict(features)[0])]))


def prepare_data(pr, pat):
    prices = pd.read_csv(pr)
    patterns = pd.read_csv(pat)

    df_prices = pd.DataFrame(columns=['week', 'user', 'prices'])

    def prepare_data_by_week(wk):
        lof = []
        entries = prices[prices.week == wk]
        for user in entries.user.unique():
            user_entries = entries[entries.user == user]

            user_prices = user_entries.price.tolist()

            features = dict(zip(stalk_time, user_prices))

            if bool(random.getrandbits(1)):
                for i in range(random.randrange(0, 5)):
                    features[random.choice(stalk_time)] = 0
            else:
                for t in stalk_time[random.randrange(7, 9):]:
                    features[t] = 0

            lof.append({**features, 'week': wk, 'user': user})
        return lof

    priceList = [prepare_data_by_week(w) for w in prices.week.unique()]
    df_prices = df_prices.append([price for sl in priceList for price in sl], ignore_index=True)

    data = pd.merge(df_prices, patterns, on=['week', 'user']).fillna(0)

    del data['user'], data['week'], data['prices']

    return np.split(data, [14], axis=1)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--new_model", required=False, type=bool,
                        help="if a new model should be trained before the prediction", default=False)
    parser.add_argument("--n_jobs", required=False, type=int,
                        help="how many threads should be used for the classification task", default=-1)
    args = parser.parse_args()

    new_model = args.new_model or not os.path.isfile('model/estimator.pkl')

    prices_samples_path = 'data/stalk_prices_samples.csv'
    patterns_samples_path = 'data/stalk_patterns_samples.csv'
    prices_path = 'data/stalk_prices.csv'
    if new_model:
        X_train, y_train = prepare_data(prices_samples_path, patterns_samples_path)
        model = hyperparameter_optimization(X_train, y_train, args.n_jobs)
        if not os.path.exists('model'):
            os.makedirs('model')
        pickle.dump(model, open('model/estimator.pkl', 'wb'))
    else:
        model = pickle.load(open('model/estimator.pkl', 'rb'))
    make_prediction(prices_path, model)
