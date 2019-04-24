import pickle, os
from utility.util import StocksData
from trading_env.backtest_v0 import Trading_env

import numpy as np

def loadByPickle(filename, path=None):
    path = path or FOLDER_PATH
    filename = os.path.join(path, filename + '.pkl')
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


if __name__ == '__main__':
    filename = 'simple' #default_stock'

    df = loadByPickle(filename=filename, path='Data')

    close = [True] * len(df.symbols)
    env = Trading_env(env='ratio_trade', data = df, cash=10000, fee=.001, adj_close=close)
    
    pos = [1/len(df.symbols)] * len(df.symbols)
    env.reset()

    while True:
        state, reward, done, info = env.step()
        env.action(pos, cash_per=1/len(df.symbols))
        print(info)
        if done:
            break
