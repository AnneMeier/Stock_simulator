"""
reference : https://github.com/kernc/backtesting.py
            https://github.com/Yvictor/TradingGym
            
"""

import pandas as pd
import numpy as np

class _Array(np.ndarray):
    """
    ndarray extended to supply .name and other arbitrary properties
    in ._opts dict.
    """
    def __new__(cls, array, name=None, write=False, **kwargs):
        obj = np.asarray(array).view(cls)
        obj.name = name or array.name
        obj._opts = kwargs
        if not write:
            obj.setflags(write=False)
        return obj

    def __array_finalize__(self, obj):
        if obj is not None:
            self.name = getattr(obj, 'name', '')
            self._opts = getattr(obj, '_opts', {})

    def __bool__(self):
        try:
            return bool(self[-1])
        except IndexError:
            return super().__bool__()

    def __float__(self):
        try:
             return float(self[-1]) 
        except IndexError:
            return super().__float__()

    def to_series(self):
        return pd.Series(self, index=self._opts['Date'].index, name=self.name)

class _StockDataFrame():
    def __init__(self, df):
        self.__i = len(df)
        self.__cache = {}
        self.__arrays = {col: _Array(arr, data=self) for col, arr in df.items()}
        self.__arrays['__index'] = df.index.copy()

    def getitem(self, item):
        return getattr(self, item)

    def __getitem__(self, item):
        return getattr(self, item)

    def __getattr__(self, item):
        try:
            return self.__get_array(item)
        except KeyError:
            raise KeyError("Column '{}' not in data".format(item)) from None
    
    def __get_array(self, key):
        arr = self.__cache.get(key)
        if arr is None:
            arr = self.__cache[key] = self.__arrays[key][:self.__i]
        return arr

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__ = state