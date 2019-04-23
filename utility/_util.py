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

class StocksData():
    def __init__(self, df, id, using_features):
        df = df[using_features]
        
        self.__id = id
        self.__len = df.shape[0]
        self.__features = df.columns if isinstance(df.columns, list) else list(df.columns)

        self.__symbols = np.sort(np.unique(df[id]))
        self.__data = {}

        import time
        start_time = time.time()
        print("Indexing {0}'s symbols and {1} features".format(len(self.__symbols), len(self.__features)))
        for symbol in self.__symbols:
            self.__data[symbol] = _StockDataFrame(df[df[id] == symbol])
        print("============== {} seconds ==============".format(time.time() - start_time))
        print("Finished hashing stocks")

    @property
    def symbols(self):
        return self.__symbols
    
    @symbols.setter
    def symbols(self, symbols):
        self.__symbols = symbols
    
    @property
    def features(self):
        return self.__features

    @features.setter
    def features(self, features):
        self.__features = features

    def getitem(self, item):
        if isinstance(item, int) is False: item = str(item)
        return getattr(self, item)

    def __getitem__(self, item):
        if isinstance(item, str) is False: item = str(item)
        return getattr(self, item)

    def __getattr__(self, item):
        if isinstance(item, str): item = int(item)
        try:
            return self.__data[item]
        except KeyError:
            raise KeyError("Symbol '{}' not in data".format(item)) from None

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__ = state

    def __repr__(self):
        return "ID: {0}\nLength of data: {1}\nFeatures: {2}\nSymbols : {3}"\
                .format(self.__id, self.__len, self.__features, self.__symbols)
