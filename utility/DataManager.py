from functools import wraps
import pandas as pd
import numpy as np
import inspect
import pickle, os

from .util import StocksData

FOLDER_PATH = '../KOSPI200'


def saveDataFrame(df, filename, path=None):
    path = path or FOLDER_PATH
    filename = os.path.join(path, filename + '.csv')
    df.to_csv(filename)
    print(filename + ' saved!')

def saveByPickle(data, filename, path=None):
    path = path or FOLDER_PATH
    filename = os.path.join(path, filename + '.pkl')
    with open(filename, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    print(filename + ' saved!')

def loadDataFrame(filename, path=None):
    path = path or FOLDER_PATH
    filename = os.path.join(path, filename + '.csv')
    df = pd.read_csv(filename)
    return df

def loadByPickle(filename, path=None):
    path = path or FOLDER_PATH
    filename = os.path.join(path, filename + '.pkl')
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
    def __init__(self, df, id=None):
        df = df.sort_values(['Symbol', 'Date'], ascending=[True, True]).reset_index(drop=True)
        self.df = df
        self.__id = id or Preprocess.__DEFAULT_ID
 
    @property
    def id(self):
        return self.__id
    
    @id.setter
    def id(self, id):
        self.__id = id


    def __naming(self, col, name, wide, other=False):
        if other in NAME_CHANGE_VAL : name = NAME[name]
        return col + name + str(wide)

    def __Transform_stockDF(self, using_feature):
        return StocksData(self.df, self.__id, using_features)

    def save(self, filename, path, using_features=['Symbol', 'Date'], form=True):
        data = self.__Transform_stockDF(using_features) if form else self.df[using_features]
        saveByPickle(data, filename, path)
        return data

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
            x = df.groupby(self.__id)[column].rolling(window=window, min_periods=min_periods)
            return x.mean().reset_index()[column]
        
        temp = self._getInstanceDF([self.__id] + ['High', 'Low', 'Close'])
        temp['typical_price'] = (temp.High + temp.Low + temp.Close) / 3
        x = temp['typical_price'] - groupByRollingMean(temp, 'typical_price')
        temp['abs'] = abs(x)
        mean_dev = groupByRollingMean(temp, 'abs')
        temp[name] = x / (0.15 * mean_dev)
        
        if saved: self.mergeByColumn(temp[name])

        return temp[name]
        

if __name__ == '__main__':
    df = loadByPickle(filename='a', path='../KOSPI200')

    p = Preprocess(df)
    p.MovingAverage(columns=['Close', 'High'], window=10)
    p.LagReturn(columns=['Close'], type='log_lag')
    p.WeightedAverage()
    p.macd()
    p.StochasticOscillator()
    p.CommodityChannelINdex()
    print(p.df.columns)
    using_features = ['Date', 'High', 'Low', 'Open', 'Close', 'Volume',
       'Adj Close', 'Symbol', 'Close_MA_10', 'High_MA_10', 'Close_DIFF.LOG_1',
       'Close_EWM_12', 'Close_MACD_12', 'Close_STOC.OSC_14', 'CCI_20']
    new_data = p.save(filename='test', path='../Data', using_features=using_features)
    print(new_data.symbols)
    print(new_data.features)