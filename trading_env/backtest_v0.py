import os
import logging

import numpy as np
import pandas as pd

import matplotlib
from abc import ABCMeta, abstractmethod


class _Order(metaclass=ABCMeta):
    def __init__(self, broker):
        self._broker = broker
        self._is_long = None
        self._entry = None

    @abstractmethod
    def _update(self):
        pass
    
    @abstractmethod
    def cancel(self):
        pass

    @property
    def entry(self):
        return self._entry

    @property
    def is_long(self):
        return self._is_long

class _Broker(metaclass=ABCMeta):
    def __init__(self, trading_env, data, symbol, use_adj_close):
        self._trading_env = trading_env
        self._data = data
        self._symbol = symbol
        self._position = 0

        self._close = self._data.Adj_Close[self._current] if use_adj_close \
                        else self._data.Close[self._current]

        self._order = _Order

    def reset(self):
        self._position = 0
        self._order.cancel()

    @abstractmethod
    def buy(self, price=None):
        pass
    
    @abstractmethod
    def sell(self, price=None):
        pass

    @property
    def pos(self):
        return self._position 

    @property
    def _current(self):
        return self._trading_env._broker_current_day

    @property 
    def last_close(self):
        return self._close

    @property
    def last_open(self):
        return self._data.Open[self._current]
    
    @property
    def last_volumne(self):
        return self._data.Volume[self._current]

    @property 
    def last_day(self):
        return self._data.Date[self._current]

class _Order_Ratio(_Order):
    def __init__(self, *args, **kwargs):
        super(_Order_Ratio, self).__init__(*args, **kwargs)

    def _update(self, entry, is_long=True):
        self._entry = price
        self._is_long = is_long

    def cancel(self):
        self._entry = self._is_long = None

class _Broker_Ratio(_Broker):
    def __init__(self, *args, **kwargs):
        super(_Broker_Ratio, self).__init__(*args, **kwargs)
        self._order = _Order_Ratio(self)

    def buy(self, price):
        self._order._update(price, is_long=True)
    
    def sell(self, price):
        self._order._update(price, is_long=False)


ENV_NAME_FOR_CLASS = {
    'ratio' : _Broker_Ratio
}

class Trading_env:
    class _Log:
        def __init__(self):
            pass

        def reset(self):
            self.return_per_symbol = []
            self.returns           = []
            self.positions         = []


    def __init__(self, env, data, cash, fee, leverage=1.0, adj_close=None):
        assert 0 < cash, "cash shosuld be >0, is {}".format(cash)
        assert 0 <= fee < .1, "commission should be between 0-10%, is {}".format(fee)
        # assert 0 < margin <= 1, "margin should be between 0 and 1, is {}".format(margin)
        
        self._d = 0
        self._symbols = data.symbols
        self._features = data.features
        
        self._cash = cash
        self._fee = fee
        self._leverage = leverage
        
        self._log = _Log()
        self._brokers = {}

        BROKER = ENV_NAME_FOR_CLASS[env]

        if adj_close is None:
            adj_close = [False] * len(self._symbols)
        assert len(self._symbols) == len(self.adj_close)

        for symbol, _adj_close in zip(data.symbols, adj_close):
            self._brokers[symbol] = BROKER(self, _data[symbol], symbol, _adj_close)

        ####  For Reset()  ####
        self.__cash = cash
        self._log.reset()

    def reset(self, init_state=None):
        self._d = 0
        self._cash = self.__cash
        self._log.reset()

        for broker in _brokers_generater:
            broker.reset()

        if init_state is None:
            init_state = [0] * len(self._symbols)
        self.portfolio_pos(init_state)


    def step(self):
        pass

    def action(self, portfolio, cash_per=0.0, lt=0.0, st=0.0):
        self.portfolio_pos(portfolio, cash_per=cash_per, lt=lt, st=st)
        
    
    @property
    def equity(self):
        return self._cash + np.sum([broker.pos for broker in self._brokers_generater])

    @property
    def _broker_current_day(self):
        return self._d

    @property
    def _brokers_generater(self):
        for symbol in self.symbols:
            yield self._brokers[symbol]

    def __is_portfolio_format(self, portfolio):
        return isinstance(portfolio, list) and ([p < .0 for p in portfolio] == 0)

    def portfolio_pos(self, portfolio, cash_per=0.0, lt=0.0, st=0.0):
        """
            lt : long position threshold
            st : shot position threshold
        """
        assert self.__is_portfolio_format(portfolio) 
        
        ratio_sum = np.sum(portfolio) + cash_per
        portfolio = [self._cash * p / ratio_sum for p in portfolio]
        cash =  self._cash * cash_per / ratio_sum

        brokers_cur_pos = [ broker.pos for broker in self._brokers_generater ]
        
        for broker, broker_cur_pos, order_pos in zip(self._brokers_generater, brokers_cur_pos, portfolio):
            pos = order_pos - broker_cur_pos
            if pos > lt:
                broker.buy(price=pos)
            elif pos < st:
                broker.sell(price=pos)
            else:
                cash += pos