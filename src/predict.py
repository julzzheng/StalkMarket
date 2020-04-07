import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split, cross_val_score
from argparse import ArgumentParser
import datetime
import os
import pickle

stalk_time = ['mon_am', 'mon_pm', 'tue_am', 'tue_pm', 'wed_am', 'wed_pm', 'thu_am', 'thu_pm', 'fri_am',
              'fri_pm', 'sat_am', 'sat_pm']

pattern = ['Ramdom', 'Decreasing', 'Small Spike', 'Camel Hump']


def hyperparameter_optimization(x, y):
    pipeline = Pipeline([
        ('clf', OneVsRestClassifier(KNeighborsClassifier()))
    ])
    parameters = {
        'clf__estimator__n_neighbors': np.arange(1, 31, 2),
        'clf__estimator__weights': ['uniform', 'distance'],
        'clf__estimator__algorithm': ['auto'],
        'clf__estimator__leaf_size': np.arange(5, 101, 1),
        'clf__estimator__p': [1, 2],
    }
    stratified_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

    random_search = RandomizedSearchCV(pipeline, parameters, scoring='balanced_accuracy', cv=stratified_fold, n_jobs=-1,
                                       verbose=1)
    random_fit = random_search.fit(x, y.values.ravel())
    print("Best score: %0.8f" % random_fit.best_score_)
    print("Best parameters set:")
    best_parameters = random_fit.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    return random_fit.best_estimator_


def nested_cv_evaluation(x, y):
    pipeline = Pipeline([
        ('clf', OneVsRestClassifier(KNeighborsClassifier()))
    ])
    parameters = {
        'clf__estimator__n_neighbors': np.arange(1, 31, 2),
        'clf__estimator__weights': ['uniform', 'distance'],
        'clf__estimator__algorithm': ['auto'],
        'clf__estimator__leaf_size': np.arange(5, 101, 1),
        'clf__estimator__p': [1, 2],
    }

    outer_fold = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
    inner_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=2)

    random_search = RandomizedSearchCV(pipeline, parameters, scoring='balanced_accuracy', cv=inner_fold, n_jobs=-1,
                                       verbose=1)
    nested_score = cross_val_score(random_search, x, y, scoring='balanced_accuracy', cv=outer_fold,
                                   n_jobs=-1, verbose=1)
    return nested_score.mean()


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
        features = features.fillna(0)
        print('The prediction for %s is the %s pattern.' % (i, pattern[int(m.predict(features)[0])]))


def prepare_data(pr, pat):
    prices = pd.read_csv(pr)
    patterns = pd.read_csv(pat)

    df_prices = pd.DataFrame(columns=['week', 'user', 'prices'])

    for week in prices.week.unique():
        entries = prices[prices.week == week]
        for user in entries.user.unique():
            user_entries = entries[entries.user == user]

            user_prices = user_entries.price.tolist()

            features = dict(zip(stalk_time, user_prices))

            features = {**features, 'week': week, 'user': user}

            df_prices = df_prices.append(features, ignore_index=True)

    data = pd.merge(df_prices, patterns, on=['week', 'user']).fillna(method='backfill')
    del data['user'], data['week'], data['prices']

    return np.split(data, [12], axis=1)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--new_model", required=False, type=bool,
                        help="if a new model should be trained before the prediction", default=False)
    args = parser.parse_args()

    new_model = args.new_model or not os.path.isfile('model/estimator.pkl')

    prices_samples_path = 'data/stalk_prices_samples.csv'
    patterns_samples_path = 'data/stalk_patterns_samples.csv'
    prices_path = 'data/stalk_prices.csv'

    if new_model:
        X_train, y_train = prepare_data(prices_samples_path, patterns_samples_path)
        model = hyperparameter_optimization(X_train, y_train)
        os.mkdir('model/');
        pickle.dump(model, open('model/estimator.pkl', 'wb'))
    else:
        model = pickle.load(open('model/estimator.pkl', 'rb'))
    make_prediction(prices_path, model)
