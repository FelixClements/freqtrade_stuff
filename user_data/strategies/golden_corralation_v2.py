# --- Do not remove these libs ---
from freqtrade.strategy import (IStrategy, merge_informative_pair, stoploss_from_open,
                                IntParameter, DecimalParameter, CategoricalParameter)
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.persistence import Trade
import numpy as np
from technical.util import resample_to_interval, resampled_merge
from freqtrade.strategy import merge_informative_pair
from datetime import datetime, timedelta
from freqtrade.exchange import timeframe_to_prev_date, timeframe_to_seconds

# --------------------------------


class golden_corralation_v2(IStrategy):
    """
    Correlates pairs to offset BTC behavior using 1m and 5m timeframes to predict next candle return. Works well during crashes.

    """

    minimal_roi = {
         "0": 0.015,
         "30": 0.01,
         "60": 0.008,
         "90": 0.008
    }

    # minimal_roi = {
    #      "0": 0.
    # }

    use_custom_sell_function=False
    use_btc_information=True

    # Custom Sell
    roi_factor = DecimalParameter(2, 5, default=2, space='buy', optimize=True, load=True)
    stoploss_factor = DecimalParameter(2, 5, default=5, space='buy', optimize=True, load=True)
    macro_candle_size = DecimalParameter(2, 5, default=5, space='buy', optimize=True, load=True)
    micro_candle_size = DecimalParameter(2, 5, default=3.5, space='buy', optimize=True, load=True)
    # sell_trend_type = CategoricalParameter(['rmi', 'ssl', 'candle', 'any'], default='any', space='sell', optimize=False, load=True)
    # sell_endtrend_respect_roi = CategoricalParameter([True, False], default=False, space='sell', optimize=False, load=True)
    oslevel1_1h = IntParameter(-40,-10,default=-20,space='buy',optimize=False,load=True)
    oslevel1_5m = IntParameter(-40,-10,default=-25,space='buy',optimize=False,load=True)
    oslevel1_25m = IntParameter(-40,-10,default=-15,space='buy',optimize=False,load=True)
    n1 = IntParameter(0,20,default=10,space='buy',optimize=False,load=True)
    n2 = IntParameter(10,30,default=21,space='buy',optimize=False,load=True)

    buy_min_corr_coef = DecimalParameter(0.50, 0.95, default=0.5, space='buy', optimize=True, load=True)


    custom_trade_info = {}

    # Sell hyperspace params:
    sell_params = {
    }

    # Stoploss:
    stoploss = -0.05

    # Optimal timeframe for the strategy
    timeframe = '1m'

	  # uses 15m candles to ensure general trend is up at 5m scale
    timescale = 12

    use_order_book = True

    order_types = {
        "buy": "limit",
        "sell": "market",
        "emergencysell": "market",
        "stoploss": "market",
        "stoploss_on_exchange": False,
        "stoploss_on_exchange_interval": 600,
        "stoploss_on_exchange_limit_ratio": 0.99,
    }

    use_custom_stoploss = False #use_custom_sell_function

     # Trailing stop:
    trailing_stop = False
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True

    startup_candle_count: int = 10

    plot_config = {
        'main_plot': {
             'resample_12_bbg_lowerband': {'color':'blue'},
            'resample_12_bbg_upperband': {'color': 'purple'},
        },
        'subplots': {
            "RSI":{
                'rsi':{'color':'green'}
            },
            "BB_spread":{
                'bb_spread_ma':{'color':'brown'}
            }
        }
    }


    def get_ticker_indicator(self):
        return int(self.timeframe[:-1])

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        if not metadata['pair'] in self.custom_trade_info:
            self.custom_trade_info[metadata['pair']] = {}
            if not 'had-trend' in self.custom_trade_info[metadata["pair"]]:
                self.custom_trade_info[metadata['pair']]['had-trend'] = False

        # Get info for big daddy BTC

        # Informative
        if not self.dp:
            # Don't do anything if DataProvider is not available.
            return dataframe

        # if self.use_btc_information:
        #     informative = self.dp.get_pair_dataframe("BTC/USDT", "1m")
        #     informative2 = self.dp.get_pair_dataframe("BTC/USDT", "5m")
        #     informative['btc_pct_change'] = (informative['close'] - informative['close'].shift(1)) / informative['close']
        #     informative2['btc_pct_change'] = (informative2['close'] - informative2['close'].shift(1)) / informative2['close']
        #
        #     dataframe = merge_informative_pair(dataframe, informative, self.timeframe, '1m', ffill=True)
        #     dataframe = merge_informative_pair(dataframe, informative2, self.timeframe, '5m', ffill=True)

        dataframe['pct_change'] = (dataframe['close'] - dataframe['close'].shift(1)) / dataframe['close']
        dataframe['total_corr'] = dataframe['close'] - dataframe['close']
        self.pair_list = self.dp.current_whitelist()

        #self.df_correlation = DataFrame()

        for pair in self.pair_list:
            coin, _ = pair.split('/')
            df = self.dp.get_pair_dataframe(pair=pair, timeframe="1m")
            #df2 = self.dp.get_pair_dataframe(pair=pair, timeframe="5m")
            df[coin+'_pct_change'] = (df['close'] - df['close'].shift(1)) / df['close']
            #df2[coin+'_pct_change_5m'] = (df2['close'] - df2['close'].shift(1)) / df2['close']
            dataframe = merge_informative_pair(dataframe, df, self.timeframe, '1m', ffill=True)
            #dataframe = merge_informative_pair(dataframe, df2, self.timeframe, '5m', ffill=True)
            dataframe[coin+'_corr_coeff'] = dataframe['pct_change'].rolling(60).corr(dataframe[coin+'_pct_change_1m'].shift(1))
            dataframe['total_corr'] += dataframe[coin+'_corr_coeff']


        dataframe['total_corr'] = dataframe['total_corr'] / len(self.pair_list)
        #dataframe['buy_corr_met'] = np.where(dataframe['total_corr'] > self.buy_min_corr_coef.value)





        #dataframe['corr_coeff'] = dataframe['pct_change'].rolling(60).corr(dataframe['btc_pct_change_1m'].shift(1).rolling(60))

        # dataframe_macro = resample_to_interval(dataframe, self.get_ticker_indicator()*5) # 5 min
        # dataframe_macro['high-low'] = dataframe_macro['high'] - dataframe_macro['low']
        # dataframe_macro['open-close'] = dataframe_macro['open'] - dataframe_macro['close']
        # dataframe_macro['avg_candle_size'] = dataframe_macro['high-low'].rolling(20).mean()
        # dataframe_macro['stoploss_amount'] = dataframe_macro['high-low']/self.stoploss_factor.value # 3
        # dataframe_macro['roi_amount'] = dataframe_macro['high-low']/self.roi_factor.value # 2
        # dataframe_macro['ATR'] = ta.ATR(dataframe_macro, timeperiod=14)
        # dataframe_macro['custom_stoploss'] = dataframe_macro['low'] - dataframe_macro['ATR']*1.5
        # dataframe_macro['roi_price'] = dataframe_macro['close'] + dataframe_macro['ATR']*1.5
        #
        # dataframe_macro['pct_change'] = (dataframe_macro['close'] - dataframe_macro['close'].shift(1)) / dataframe_macro['close']
        # dataframe = resampled_merge(dataframe, dataframe_macro)
        #
        # dataframe['custom_stoploss'] = dataframe['low'] - dataframe['resample_5_ATR']*0.8
        # dataframe['roi_price'] = dataframe['close'] + dataframe['resample_5_ATR']*1.5
        #
        #
        # dataframe['corr_coeff_5m'] = dataframe['resample_{}_pct_change'.format(self.get_ticker_indicator()*5)].rolling(60).corr(dataframe['btc_pct_change_5m'].shift(1).rolling(60))
        #
        # # 1 minute
        # dataframe['high-low'] = dataframe['high'] - dataframe['low']
        # dataframe['open-close'] = dataframe['open'] - dataframe['close']
        # dataframe['avg_candle_size'] = dataframe['high-low'].rolling(50).mean()
        #
        # dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        # dataframe['volume_rolling'] = dataframe['volume'].shift(14).rolling(14).mean()

        dataframe.fillna(method='ffill', inplace=True)



        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    (dataframe['total_corr'] > self.buy_min_corr_coef.value)
                    # (dataframe['corr_coeff']>0.4) &
                    # (dataframe['btc_pct_change_1m'] > 0.008) &
                    # (dataframe['corr_coeff_5m'] > 0.6) &
                    # (dataframe['btc_pct_change_5m'] > 0.015) &
                # (dataframe['close'].lt(dataframe['resample_{}_close'.format(self.get_ticker_indicator()*5)].shift(1)))

            ),
            'buy'] = 1


        return dataframe

    def populate_sell_trend(self, dataframe) -> DataFrame:
        dataframe.loc[
            (

            ),
            'sell'] = 1
        return dataframe


    """
    Custom Sell
    Designed to be a sort of override of ROI
    """

#     def custom_sell(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
#                     current_profit: float, **kwargs):
#
#         dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
#         current_candle = dataframe.iloc[-1].squeeze()
#         current_profit = trade.calc_profit_ratio(current_candle['close'])
#
#         #last_candle = dataframe.iloc[-1].squeeze()
#         trade_date = timeframe_to_prev_date("5m",trade.open_date_utc)
#         trade_candle = dataframe.loc[(dataframe['date'] == trade_date)] #, 'my_stoploss']
# #        with option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
# #            print(trade_candle)
#         if trade_candle.empty: return None
#         trade_candle = trade_candle.squeeze()
#
#         open_price = (1 + current_profit) * current_rate
#         roi_price =  trade_candle['roi_price']
#         roi_estimate = ( roi_price / open_price ) - 1
#         custom_roi = roi_estimate
#
#         stoploss_price =  trade_candle['custom_stoploss'] # (open_price - trade_candle['resample_{}_bb_lowerband_stoploss_amount'.format(self.get_ticker_indicator()*12)])
#
#         stoploss_estimate = 1 - (stoploss_price / open_price)
#         stoploss_decay = stoploss_estimate * ( 1 - ((current_time - trade.open_date_utc).seconds) / (60*60)) # linear decay
#         if stoploss_decay<0: stoploss_decay = 0
#         #stoploss_pct = (stoploss_decay / current_rate) - 1
#         if (current_profit < -stoploss_decay): return 'custom stoploss'
#
#         roi_decay = roi_estimate * ( 1 - ((current_time - trade.open_date_utc).seconds) / (60*60)) # linear decay
#         #roi_decay = roi_estimate * np.exp(-(current_time - trade.open_date_utc).seconds) / (120*60) # exponential decay
#         if roi_decay<0: roi_decay = 0.005
#
#         if roi_decay < 0.005: roi_estimate = 0.005
#         else: roi_estimate = roi_decay
#
#         if current_profit > roi_estimate:
#             return 'custom roi'


    def market_cipher(self, dataframe) -> DataFrame:
        #dataframe['volume_rolling'] = dataframe['volume'].shift(14).rolling(14).mean()
        #
        dataframe['ap'] = (dataframe['high'] + dataframe['low'] + dataframe['close']) / 3
        dataframe['esa'] = ta.EMA(dataframe['ap'], self.n1.value)
        dataframe['d'] = ta.EMA((dataframe['ap']-dataframe['esa']).abs(), self.n1.value)
        dataframe['ci'] = ( dataframe['ap']-dataframe['esa'] ) / (0.015 * dataframe['d'])
        dataframe['tci'] = ta.EMA(dataframe['ci'], self.n2.value)

        dataframe['wt1'] = dataframe['tci']
        dataframe['wt2'] = ta.SMA(dataframe['wt1'],4)
        dataframe['wt1-wt2'] = dataframe['wt1'] - dataframe['wt2']
        dataframe['wt2-wt1'] = dataframe['wt2'] - dataframe['wt1']  # if negative, crossed up, if positive crosseed down
        dataframe['slope_wt1'] = dataframe['wt1'] - dataframe['wt1'].shift(1)

        dataframe['crossed_above'] = qtpylib.crossed_above(dataframe['wt2'], dataframe['wt1'])
        dataframe['crossed_above'] = dataframe['wt1'].crossed_above(dataframe['wt2'])
        dataframe['crossed_below'] = dataframe['wt1'].crossed_below(dataframe['wt2'])
        #dataframe['slope_gd'] = ta.LINEARREG_ANGLE(dataframe['crossed_above'] * dataframe['wt2'], 10)

        return dataframe

    def SSLChannels_ATR(self, dataframe, length=7):
        """
        SSL Channels with ATR: https://www.tradingview.com/script/SKHqWzql-SSL-ATR-channel/
        Credit to @JimmyNixx for python
        """
        df = dataframe.copy()

        df['ATR'] = ta.ATR(df, timeperiod=14)
        df['smaHigh'] = df['high'].rolling(length).mean() + df['ATR']
        df['smaLow'] = df['low'].rolling(length).mean() - df['ATR']
        df['hlv'] = np.where(df['close'] > df['smaHigh'], 1, np.where(df['close'] < df['smaLow'], -1, np.NAN))
        df['hlv'] = df['hlv'].ffill()
        df['sslDown'] = np.where(df['hlv'] < 0, df['smaHigh'], df['smaLow'])
        df['sslUp'] = np.where(df['hlv'] < 0, df['smaLow'], df['smaHigh'])

        return df['sslDown'], df['sslUp']

    def SROC(self, dataframe, roclen=21, emalen=13, smooth=21):
        df = dataframe.copy()

        roc = ta.ROC(df, timeperiod=roclen)
        ema = ta.EMA(df, timeperiod=emalen)
        sroc = ta.ROC(ema, timeperiod=smooth)

        return sroc

    def RMI(self, dataframe, length=20, mom=5):
        """
        Source: https://github.com/freqtrade/technical/blob/master/technical/indicators/indicators.py#L912
        """
        df = dataframe.copy()

        df['maxup'] = (df['close'] - df['close'].shift(mom)).clip(lower=0)
        df['maxdown'] = (df['close'].shift(mom) - df['close']).clip(lower=0)

        df.fillna(0, inplace=True)

        df["emaInc"] = ta.EMA(df, price='maxup', timeperiod=length)
        df["emaDec"] = ta.EMA(df, price='maxdown', timeperiod=length)

        df['RMI'] = np.where(df['emaDec'] == 0, 0, 100 - 100 / (1 + df["emaInc"] / df["emaDec"]))

        return df["RMI"]
