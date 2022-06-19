from sklearn import datasets
from sklearn.model_selection import train_test_split
from dset import load_pickle
from lightgbm.sklearn import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import numpy as np
from extract_features import FEATURE_NAMES

def load_predictor(n_estimators=1000, skip_validation=False):
    model = LGBMRegressor(n_estimators=n_estimators)
    X, y = load_pickle('../data/diverse_ten_feats_10k.pkl')

    if not skip_validation:
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        model.fit(X_train, y_train)
        y_hat = model.predict(X_test)

        print('MAE:', mean_absolute_error(y_test, y_hat))
        print('MAPE: ', mean_absolute_percentage_error(y_test, y_hat))

    model.fit(X, y, feature_name=FEATURE_NAMES)
    return model

def load_predictor_4d(n_estimators=1000):
    model = LGBMRegressor(n_estimators=n_estimators)
    X, y = load_pickle('../data/diverse_4d_feats_10k.pkl')
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    model.fit(X_train, y_train)
    y_hat = model.predict(X_test)

    print('MAE:', mean_absolute_error(y_test, y_hat))
    print('MAPE: ', mean_absolute_percentage_error(y_test, y_hat))

    model.fit(X, y)
    return model
