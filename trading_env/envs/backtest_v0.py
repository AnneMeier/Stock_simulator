import os
import logging

import pandas as pd
import numpy as np

import matplotlib

class trading_env:
    def __init__(self, env_id, obs_data_len, step_len
                 df, fee, deal_col_name='Close',
                 deal_symbols=None,
                 feature_cols=['Close', 'Volume'],
                 *args, **kargs):
        logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
        
        self.df = df
        self.fee = fee
        self.symbols = deal_symbols
        self.action_space = np.array([3, ])
        self.gym_actions = range(3)

        self.obs_len = obs_data_len
        self.symbols_len = len(deal_symbols)
        self.feature_len = len(feature_cols)
        self.observation_space = np.array([self.obs_len * self.feature_len, ])
        self.using_features = feature_cols
        self.price_name = deal_col_name

        self.step_len = step_len

        self.logger.info('Making new env: {}'.format(env_id))
    
    def reset(self):
        self.df_sample = self.df
        self.step_st = 0

        self.reward_sum = 0
        self.reward_ret = 0
        self.transcation_info = pd.DataFrame()
        
        self.posi_i = [0]*self.obs_len