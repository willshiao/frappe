from dset import generate_diverse_tens
from sklearn.model_selection import train_test_split
from lightgbm.sklearn import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from dset import save_pickle

X, y = generate_diverse_tens(n=1000, order=4)
save_pickle((X, y), '../data/diverse_4d_feats_1k_recursive.pkl')
model = LGBMRegressor(n_estimators=1000)
X_train, X_test, y_train, y_test = train_test_split(X, y)

model.fit(X_train, y_train)
y_hat = model.predict(X_test)

print('MAE:', mean_absolute_error(y_test, y_hat))
print('MAPE: ', mean_absolute_percentage_error(y_test, y_hat))
