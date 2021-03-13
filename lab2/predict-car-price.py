import pandas as pd
import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt

class CarPrice:

    def __init__(self):

        self.df = pd.read_csv('data/data.csv')
        
        self.df.columns = self.df.columns.str.lower().str.replace(' ', '_')
        string_columns = list(self.df.dtypes[self.df.dtypes == 'object'].index)
        for col in string_columns:
            self.df[col] = self.df[col].str.lower().str.replace(' ', '_')

        self.base = ['engine_hp', 'engine_cylinders', 'highway_mpg', 'city_mpg', 'popularity']

        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)

    def display(self, title, X, y, y_pred):
        print('========', title, '========')
        columns = ['engine_cylinders','transmission_type','driven_wheels','number_of_doors',
                   'market_category','vehicle_size','vehicle_style','highway_mpg','city_mpg','popularity']
        X = X.copy()
        X = X[columns]
        X['msrp'] = y.round(2)
        X['msrp_pred'] = y_pred.round(2)
        print(X.head(5).to_string(index=False))

    def prepare_X(self, df):
        df_num = df[self.base]
        df_num = df_num.fillna(0)
        X = df_num.values
        return X

    def rmse(self, y, y_pred):
        error = y_pred - y
        mse = (error ** 2).mean()
        return np.sqrt(mse)

    def plot_graph(self, title, y, y_pred):
        plt.figure(figsize=(6, 4))

        sns.distplot(y_train, label='target', kde=False,
                    hist_kws=dict(color='#222222', alpha=0.6))
        sns.distplot(y_pred, label='prediction', kde=False,
                    hist_kws=dict(color='#aaaaaa', alpha=0.8))

        plt.legend()

        plt.ylabel('Frequency')
        plt.xlabel('Log(Price + 1)')
        plt.title('{0}Predictions vs actual distribution'.format(title))

        plt.show()

    def linear_regression(self, X, y):
        ones = np.ones(X.shape[0])
        X = np.column_stack([ones, X])

        XTX = X.T.dot(X)
        XTX_inv = np.linalg.inv(XTX)
        w = XTX_inv.dot(X.T).dot(y)

        return w[0], w[1:]

if __name__ == "__main__":
    # execute only if run as a script
    cp = CarPrice()

    np.random.seed(2)

    n = len(cp.df)

    n_val = int(0.2 * n)
    n_test = int(0.2 * n)
    n_train = n - (n_val + n_test)

    idx = np.arange(n)
    np.random.shuffle(idx)

    df_shuffled = cp.df.iloc[idx]

    df_train = df_shuffled.iloc[:n_train].copy()
    df_val = df_shuffled.iloc[n_train:n_train+n_val].copy()
    df_test = df_shuffled.iloc[n_train+n_val:].copy()

    y_train_orig = df_train.msrp.values
    y_val_orig = df_val.msrp.values
    y_test_orig = df_test.msrp.values

    y_train = np.log1p(df_train.msrp.values)
    y_val = np.log1p(df_val.msrp.values)
    y_test = np.log1p(df_test.msrp.values)

    del df_train['msrp']
    del df_val['msrp']
    del df_test['msrp']

    X_train = cp.prepare_X(df_train)
    X_val = cp.prepare_X(df_val)
    X_test = cp.prepare_X(df_test)
    w_0, w = cp.linear_regression(X_train, y_train)
    
    y_train_pred = w_0 + X_train.dot(w)
    y_val_pred = w_0 + X_val.dot(w)
    y_test_pred = w_0 + X_test.dot(w)

    cp.display('Training', df_train, y_train, y_train_pred)
    cp.display('Validation', df_val, y_val, y_val_pred)
    cp.display('Test', df_test, y_test, y_test_pred)

    perf_train = cp.rmse(y_train, y_train_pred)
    perf_val = cp.rmse(y_val, y_val_pred)
    perf_test = cp.rmse(y_test, y_test_pred)
    print('\n')
    print('Training rmse: ', round(perf_train,4))
    print('Validation rmse: ', round(perf_val,4))
    print('Test rmse: ', round(perf_test,4))

    cp.plot_graph('Training: ', y_train, y_train_pred)
    cp.plot_graph('Validation: ', y_val, y_val_pred)
    cp.plot_graph('Test: ', y_test, y_test_pred)