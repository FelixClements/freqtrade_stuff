# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
# --------------------------------

import talib.abstract as ta
import numpy as np
import freqtrade.vendor.qtpylib.indicators as qtpylib
import datetime
from technical.util import resample_to_interval, resampled_merge
from datetime import datetime, timedelta
from freqtrade.persistence import Trade
from freqtrade.strategy import stoploss_from_open, merge_informative_pair, DecimalParameter, IntParameter, CategoricalParameter
import technical.indicators as ftt



def EWO(dataframe, ema_length=5, ema2_length=35):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['close'] * 100
    return emadif


class BinHModWhiteHOV0(IStrategy):
    INTERFACE_VERSION = 2

    # ROI table:
    minimal_roi = {
      "0": 10
    }
    
    # Buy hyperspace params:
    buy_params = {
        "bb_bottom_delta_diff": 0.006,
        "bb_delta_diff": 0.022,
        "bb_tail_diff": 0.132,
        "bb_top_delta_diff": 0.013,
        "bb_width": 0.069,
        "ema_rise": 54,
        "ema_slow": 160,
        "ema_trend": 90,
        "ewo_base": 70,
        "ewo_high": 2.9,
        "ewo_step": 90,
        "protection_one": True,  # value loaded from strategy
        "protection_three": True,  # value loaded from strategy
        "protection_two": True,  # value loaded from strategy
    }

    # Sell hyperspace params:
    sell_params = {
        "high_offset": 1.0,  # value loaded from strategy
    }
    
    # Stoploss:
    stoploss = -0.3

    # Protections
    protection_one = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    protection_two = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    protection_three = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    
    # Buy
    ema_trend = CategoricalParameter([50, 70, 90, 110], default=buy_params['ema_trend'], space='buy', optimize=True, load=True)
    ema_slow = CategoricalParameter([140, 160, 180, 200, 220, 240, 260, 280, 300], default=buy_params['ema_slow'], space='buy', optimize=True, load=True)
    ema_rise = CategoricalParameter([6, 12, 18, 24, 30, 36, 42, 48, 54, 60], default=buy_params['ema_rise'], space='buy', optimize=True, load=True)
    
    # Strat
    bb_width = DecimalParameter(0.001, 0.070, default=buy_params['bb_width'], decimals=3, space='buy', optimize=True)
    bb_delta_diff = DecimalParameter(0.001, 0.06, default=buy_params['bb_delta_diff'], decimals=3, space='buy', optimize=True)
    bb_tail_diff = DecimalParameter(0.050, 0.450, default=buy_params['bb_tail_diff'], decimals=3, space='buy', optimize=True)
    bb_bottom_delta_diff = DecimalParameter(0.005, 0.040, default=buy_params['bb_bottom_delta_diff'], decimals=3, space='buy', optimize=True)
    bb_top_delta_diff = DecimalParameter(0.005, 0.090, default=buy_params['bb_top_delta_diff'], decimals=3, space='buy', optimize=True)
    
    ewo_base = CategoricalParameter([50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150], default=buy_params['ewo_base'], space='buy', optimize=True, load=True)
    ewo_step = CategoricalParameter([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150], default=buy_params['ewo_step'], space='buy', optimize=True, load=True)
    ewo_high = DecimalParameter(1.0, 6.0, default=buy_params['ewo_high'], decimals=1, space='buy', optimize=True)
    
    # Sell
    high_offset = DecimalParameter(1.000, 1.020, default=sell_params['high_offset'], decimals=3, space='sell', optimize=False, load=True)
   
    # Trailing stop:
    trailing_stop = False
    trailing_only_offset_is_reached = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.05

    # Sell signal
    use_sell_signal = True
    sell_profit_only = False
    sell_profit_offset = 0.01
    ignore_roi_if_buy_signal = True

    # Optimal timeframe for the strategy
    timeframe = '5m'
    informative_timeframe = '1h'

    process_only_new_candles = True
    startup_candle_count = 500

    use_custom_stoploss = False

    def informative_pairs(self):

        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.informative_timeframe) for pair in pairs]

        return informative_pairs

    def get_informative_indicators(self, metadata: dict):

        dataframe = self.dp.get_pair_dataframe(
            pair=metadata['pair'], timeframe=self.informative_timeframe)

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    
        # EMA protect
        for val in range(50, 130, 20):
            dataframe[f'ema_trend_{val}'] = ta.EMA(dataframe, timeperiod=val)        
        for val in range(140, 320, 20):
            dataframe[f'ema_slow_{val}'] = ta.EMA(dataframe, timeperiod=val)
                         
        # Elliot
        for val in range(50, 170, 10):
            for step in range(10, 170, 10):
            	dataframe[f'EWO_{val}_{step}'] = EWO(dataframe, (val), (val + step))
            	
        # Bottoms and Tops
        dataframe['bottom'] = dataframe[['open', 'close']].min(axis=1)
        dataframe['top'] = dataframe[['open', 'close']].max(axis=1)
        
        # BB 40
        bb_40 = qtpylib.bollinger_bands(dataframe['close'], window=40, stds=2)
        dataframe['lower'] = bb_40['lower']
        dataframe['mid'] = bb_40['mid']
        dataframe['bbwidth'] = (bb_40['upper'] - bb_40['lower']) / bb_40['mid']
        dataframe['space'] = dataframe['bbwidth'].rolling(60).mean()
        dataframe['bbdelta'] = (bb_40['mid'] - bb_40['lower']).abs()
        dataframe['bottomdelta'] = (dataframe['bottom'] - dataframe['bottom'].shift()).abs()
        dataframe['topdelta'] = (dataframe['top'] - dataframe['top'].shift()).abs()
        dataframe['tail'] = (dataframe['bottom'] - dataframe['low']).abs()
       
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        
        if self.protection_one.value:
        	conditions.append(
        		(
        			# Bull trend
                		(dataframe[f'ema_trend_{self.ema_trend.value}'] > dataframe[f'ema_slow_{self.ema_slow.value}'])
                	)
                )
                
        if self.protection_two.value:
        	conditions.append(
        		(
        			# Rising trend
        			(dataframe[f'ema_slow_{self.ema_slow.value}'] > dataframe[f'ema_slow_{self.ema_slow.value}'].shift(self.ema_rise.value)) &
                		(dataframe[f'ema_slow_{self.ema_slow.value}'].shift(self.ema_rise.value) > dataframe[f'ema_slow_{self.ema_slow.value}'].shift(self.ema_rise.value + self.ema_rise.value))
                	)
                )
                
        if self.protection_three.value:
        	conditions.append(
        		(
        			# EWO
                		(dataframe[f'EWO_{self.ewo_base.value}_{self.ewo_step.value}'] > self.ewo_high.value)
                	)
                )
                
        # Main
        conditions.append(
            (                   
                # Strat
                (dataframe['space'] > self.bb_width.value) &
                (dataframe['bottom'] < dataframe['lower'].shift()) &
                (dataframe['bottom'] < dataframe['bottom'].shift()) &                
                (dataframe['bbdelta'] > dataframe['bottom'] * self.bb_delta_diff.value) &
                (dataframe['tail'] < dataframe['bbdelta'] * self.bb_tail_diff.value) & 
                (dataframe['bottomdelta'] > (dataframe['bottom'] * self.bb_bottom_delta_diff.value)) &       
                (dataframe['topdelta'] < (dataframe['top'] * self.bb_top_delta_diff.value)) &       
                (dataframe['lower'].shift() > 0) &
                
                # Volume
                (dataframe['volume'] > 0)
            )
        )


        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'buy'
            ]=1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        conditions.append(
            (
                (dataframe['close'] > dataframe['mid'] * self.high_offset.value) &
                
                (dataframe['volume'] > 0)
            )
        )

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'sell'
            ]=1

        return dataframe
