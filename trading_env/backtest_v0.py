import os
import logging

import pandas as pd
import numpy as np

import matplotlib
from abc import ABCMeta, abstractmethod


class Strategy:
    pass


class _Exchanger:
    class _Log:
        pass
    def __init__(self, data, cash, fee):
        self._data = data
        self._cash = cash
        self._fee = fee
        self._d = 0

        self._brokers = {}
        for symbol in data.symbols:
            self._brokers[symbol] = _Broker(self, symbol)

    def _get_broker_last_close(self, symbol):
        return self._data[symbol].Close[self._d]

    def _get_broker_last_date(self, symbol):
        return self._data[symbol].Date[self._d]

    def buy(self, symbol, price=None, sl):
        self._brokers[symbol].buy(price)

    def sell(self, symbol, price=None):
        self._brokers[symbol].sell(price)  



class _Broker:
    def __init__(self, exchanger, symbol):
        self._exchanger = exchanger
        self._symbol = symbol

    @property 
    def last_close(self):
        return self._exchanger._get_broker_last_close(self._symbol)

    @property
    def last_date(self):
        return self._exchanger._get_broker_last_date(self._symbol)

    def buy(self, price=None):
        pass
    
    def sell(self, price=None):
        pass


class trading_env:
    def __init__(self, *args, **kargs):
        logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
    
    def reset(self):
        pass

