# https://github.com/Yvictor/TradingGym
# https://github.com/kernc/backtesting.py


from .envs import available_env_module

def available_env():
    available_env = [ env_module.__name__.split('.')[-1] for env_module in available_env_module]
    return available_env

def make(env_id, *args, **kargs):
    envs = available_env()
    assert env_id in envs , "env_id: {} not exist. try one of {}".format(env_id, str(envs).strip('[]'))

    trading_env = available_env_module[envs.index(env_id)].trading_env
    print()
    env = trading_env(env_id=env_id, *args, **kargs)

    return env