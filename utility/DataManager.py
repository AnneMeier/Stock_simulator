from functools import wraps
import pandas as pd
import numpy as np
import inspect
import pickle, os

from _util import _StockDataFrame
"""
design pattern   https://hamait.tistory.com/854
                 https://sungsoo.github.io/2018/03/19/design-patterns-in-python.html

"""
FOLDER_PATH = '../KOSPI200'


def saveDataFrame(df, filename, path=None):
    path = path or FOLDER_PATH
    filename = os.path.join(path, filename + '.csv')
    df.to_csv(filename)
    print(filename + '.csv saved!')

def saveByPickle(data, filename, path=None):
    path = path or FOLDER_PATH
    filename = os.path.join(path, filename + '.pickle')
    with open(filename, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    print(filename + '.pickle saved!')

def loadDataFrame(filename, path=None):
    path = path or FOLDER_PATH
    filename = os.path.join(path, filename + '.csv')
    df = pd.read_csv(filename)
    return df

def loadByPickle(filename, path=None):
    path = path or FOLDER_PATH
    filename = os.path.join(path, filename + '.pickle')
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


NAME_CHANGE_VAL = ['log_lag', 'std']
NAME = {
    '_EWM_'  : '_EWM.STD_',
    '_MA_'   : '_MA.STD_',
    '_DIFF_' : '_DIFF.LOG_',
}

class _DFManage:
    def _getSymbols(self):
        return np.sort(np.unique(self.df[self.id]))

    def _getInstanceDF(self, col=['Date']):
        return self.df[col].copy() 

    def _groupByID(self, as_index=True):
        return self.df.groupby(self.id, as_index=as_index)

    def mergeByColumn(self, df1):
        self.df = pd.concat([self.df, df1], axis=1, sort=False)

    def dropByColumn(self, col):
        if isinstance(col, list) is False: col = list(col)
        self.df = self.df.drop(col, axis=1)
        print('Drop the {0} column'.format(col))

class Preprocess(_DFManage):
    __DEFAULT_ID = 'Symbol'
    def __init__(self, df, id=None, using_features=[]):
        df = df.sort_values(['Symbol', 'Date'], ascending=[True, True]).reset_index(drop=True)
        self.df = df
        self.id = id or Preprocess.__DEFAULT_ID
        self.__using_features = using_features

    @property
    def using_features(self):
        return self.__using_features

    @using_features.setter
    def using_features(self, features):
        self.__using_features = features

    def __naming(self, col, name, wide, other=False):
        if other in NAME_CHANGE_VAL : name = NAME[name]
        return col + name + str(wide)

    def __getIdDF(self, id):
        return self.df[self.df[self.id] == id][self.__using_features]

    def __Transform_StockDF(self):
        """
            Have to change
            Lot's of problem in here
        """
        print(self.__using_features)
        idx = self._getSymbols()
         
        _no_index_features = self.__using_features[:]
        _no_index_features.remove('Date')

        _index_feature = ['Date']

        print(_no_index_features)
        
        _df = self.__getIdDF(idx[0])
        _df = _df.rename(columns=dict(
            zip(_no_index_features, [s + '_' + str(idx[0]) for s in _no_index_features] )
        ))

        for id in idx[1:]:
            __df = self.__getIdDF(id)
            __df = __df.rename(columns=dict(
                zip(_no_index_features, [s + '_' + str(id) for s in _no_index_features])
            ))
            _df = pd.merge(_df, __df, on=_index_feature)

        return _StockDataFrame(_df)

    def save(self, filename, path='Data/', form=True):
        data = self.__Transform_StockDF() if form else self.df
        saveByPickle(data, filename, path)

    def technical_f(*outer_args,**outer_kwargs):
        """
            **outer_kwargs : argument(0, 1, 2, 3, 4) -> argument(prep_name)
        """
        def __TYPE_NAME_EXIST(name):
            return name in NAME.keys()

        def wrap(func):
            def wrapped_func(self, *args, **kwargs):
                """
                    **kwargs : (self, columns, window, saved, type)
                """
                arg_list = inspect.getargspec(func)[0][1:]
                arg_default_val = inspect.getargspec(func)[3]
                arg_default_val = tuple( None for _ in range(len(arg_list) - len(arg_default_val))) \
                                            + arg_default_val

                def __assign(arg, i):
                    if arg in kwargs: 
                        return kwargs[arg]
                    if arg in outer_kwargs:
                        kwargs[arg] = outer_kwargs[arg]
                        return outer_kwargs[arg]
                    if arg_default_val[i] is not None: 
                        return arg_default_val[i]
                    raise KeyError('You should assign argument {}'.format(arg))


                arg_columns  = __assign(arg_list[0], 0)
                arg_window   = __assign(arg_list[1], 1)
                arg_saved    = __assign(arg_list[2], 2)
                arg_prepname = outer_kwargs['prep_name']

                if __TYPE_NAME_EXIST(arg_prepname):
                    arg_type = __assign(arg_list[3], 3)
                else:
                    arg_type = None
                

                names = []
                temp = self._getInstanceDF()

                for column in arg_columns:
                    kwargs[arg_list[0]] = column
                    name = self.__naming(column, arg_prepname, arg_window, arg_type)    
                    
                    temp[name] = func(self, *args, **kwargs)
                    names.append(name)
                if arg_saved: self.mergeByColumn(temp[names])
            return wrapped_func 
        return wrap


    @technical_f(
        prep_name='_DIFF_', 
        columns=['Close'], 
        shift=1
    )
    def LagReturn(self, columns, shift, saved=True, type='mean'):
        x = self.df[columns] / self._groupByID()[columns].shift(shift)
        if type is 'log_lag': return np.log(x)
        return x


    @technical_f(
        prep_name='_MA_',
        columns=['Close'], 
        window=5
    )
    def MovingAverage(self, columns, window, saved=True, type='mean', min_periods=None):
        x = self._groupByID()[columns].rolling(window=window, min_periods=min_periods)
        if type is 'mean': 
            return x.mean().reset_index(0,drop=True)
        elif type is 'std':
            return x.std().reset_index(0,drop=True)
        else:
            raise KeyError('Type is not defined : {}'.format(type))


    @technical_f(
        prep_name='_EWM_',
        columns=['Close'],
        com=12
    )
    def WeightedAverage(self, columns, com, saved=True, type='mean'):
        x = self._groupByID()[columns]
        if type is 'mean':
            return x.apply(lambda x : x.ewm(com=(com - 1)/2).mean())
        elif type is 'std':
            return x.apply(lambda x : x.ewm(com=(com - 1)/2).std())
        else:
            raise KeyError('Type is not defined : {}'.format(type))


    @technical_f(
        prep_name='_MACD_',
        columns=['Close'],
        t=12
    )
    def macd(self, columns, t, saved=True, short=12, long=29):
        temp = self._getInstanceDF()
        x = self._groupByID()[columns]

        ema_short_days = x.apply(lambda x : x.ewm(com=(short - 1)/2).mean())
        ema_long_days = x.apply(lambda x : x.ewm(com=(long - 1)/2).mean())
        temp['temp'] = ema_short_days.sub(ema_long_days)
        self.mergeByColumn(temp['temp'])
        
        x = self._groupByID()['temp']
        ema_line_t_days = x.apply(lambda x : x.ewm(com=(t - 1)/2).mean())
        self.dropByColumn(col=['temp'])
        
        return ema_line_t_days


    @technical_f(
        prep_name='_STOC.OSC_',
        columns=['Close'],
        lookback=14
    )
    def StochasticOscillator(self, columns, lookback, saved=True, min_periods=None):
        x = self._groupByID()[columns].rolling(window=lookback, min_periods=min_periods)
        x_max = x.max().reset_index()[columns]
        x_min = x.min().reset_index()[columns]

        x = 100 * (self.df[columns] - x_min) / (x_max - x_min + 1e-8)
        return x
    

    def CommodityChannelINdex(self, window=20, min_periods=None, saved=True):
        name = 'CCI_' + str(window)
        def groupByRollingMean(df, column):
            x = df.groupby(self.id)[column].rolling(window=window, min_periods=min_periods)
            return x.mean().reset_index()[column]
        
        temp = self._getInstanceDF([self.id] + ['High', 'Low', 'Close'])
        temp['typical_price'] = (temp.High + temp.Low + temp.Close) / 3
        x = temp['typical_price'] - groupByRollingMean(temp, 'typical_price')
        temp['abs'] = abs(x)
        mean_dev = groupByRollingMean(temp, 'abs')
        temp[name] = x / (0.15 * mean_dev)
        
        if saved: self.mergeByColumn(temp[name])

        return temp[name]
        

if __name__ == '__main__':
    path_ = '../KOSPI200'
 
    df = loadDataFrame(filename='a', path=path_)

    p = Preprocess(df)
    p.MovingAverage(columns=['Close', 'High'], window=10)
    p.LagReturn(columns=['Close'], type='log_lag')
    p.WeightedAverage()
    p.macd()
    p.StochasticOscillator()
    p.CommodityChannelINdex()
    print(p.df.columns)

    p.using_features = ['Date', 'High', 'Low', 'Open', 'Close', 'Volume',
       'Adj Close', 'Symbol', 'Close_MA_10', 'High_MA_10', 'Close_DIFF.LOG_1',
       'Close_EWM_12', 'Close_MACD_12', 'Close_STOC.OSC_14', 'CCI_20']
    p.save(filename='test', path='../Data/')