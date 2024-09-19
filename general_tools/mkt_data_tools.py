import numpy as np
import pandas as pd
import os
import datetime as dt
import general_tools.basic_tools as gt
import general_tools.trade_math_tools as math_tools
import itertools
pd.set_option('display.max_columns', None)


class MktData:
    def __init__(self, kmeans_data):
        self.KmeansData = kmeans_data
        self.intra_df = pd.DataFrame
        self.daily_df = pd.DataFrame
        self.daily_working_df = pd.DataFrame
        self.intra_atr = pd.DataFrame
        self.intra_vol = pd.DataFrame
        self.input_features = []

    def load_intra_df(self):
        print('Loading Intraday Data')
        self.intra_df = (
            pd.read_csv(self.KmeansData.KParams.intraday_file, sep=","))

    def create_daily_df(self):
        daily_sets = []
        equity_set_cols = []
        print('Working on Daily Data')
        for osec in [self.KmeansData.KParams.sec_train] + self.KmeansData.KParams.other_sec:
            print(f'Adding: {osec}')
            daily_file = f'{osec}_daily_20240505_20040401.txt'
            temp_daily = pd.read_csv(f'{self.KmeansData.KParams.data_loc}\\{daily_file}', sep=",")
            temp_daily = gt.convert_date_time(temp_daily)
            temp_daily = temp_daily.sort_values(by='Date')
            """Need to cut data to the full testing range"""

            temp_daily['ATR_day'] = math_tools.create_atr(temp_daily)
            temp_daily.drop(columns=['Vol.1', 'Time'], inplace=True)

            temp_daily = math_tools.standardize_dailydata(temp_daily, self.KmeansData.KParams.lookback)
            for c in ['Open', 'High', 'Low', 'Close', 'Vol', 'OI', 'VolAvg', 'OpenInt', 'ATR_day']:
                temp_daily[c] = temp_daily[c].astype(np.float32)

            temp_daily.rename(columns={
                'Open': f'{osec}_Open',
                'High': f'{osec}_High',
                'Low': f'{osec}_Low',
                'Close': f'{osec}_Close',
                'Vol': f'{osec}_Vol',
                'OI': f'{osec}_OI',
                'VolAvg': f'{osec}_VolAvg',
                'OpenInt': f'{osec}_OpenInt',
                'ATR_day': f'{osec}_ATR_day'
            }, inplace=True)
            daily_sets.append(temp_daily)
            equity_set_cols.append([f'{osec}_Open', f'{osec}_High', f'{osec}_Low', f'{osec}_Close', f'{osec}_Vol',
                                    f'{osec}_OI', f'{osec}_VolAvg', f'{osec}_OpenInt', f'{osec}_ATR_day'])

        equity_set_cols = list(itertools.chain(*equity_set_cols))
        if len(daily_sets) > 1:
            dailydf = pd.merge(daily_sets[0], daily_sets[1], on=['Date', 'Datetime', 'Month', 'Year'])
            for df in daily_sets[2:]:
                dailydf = pd.merge(dailydf, df, on=['Date', 'Datetime', 'Month', 'Year'])
        else:
            dailydf = daily_sets[0]

        self.daily_df = dailydf
        self.input_features = equity_set_cols

    def complete_intra_df_work(self):
        print('Completing Intraday Work')
        self.intra_df.rename(columns={'Vol': 'Vol_int'}, inplace=True)
        self.intra_df['ATR_int'] = math_tools.create_atr(self.intra_df)
        self.intra_df = gt.convert_date_time(self.intra_df)
        self.intra_df = self.intra_df.sort_values(by='Datetime')
        self.intra_df = self.intra_df[
            (self.intra_df['Datetime'].dt.hour >=
             self.KmeansData.KParams.start_hour) & (self.intra_df['Datetime'].dt.hour <= 15)]
        self.intra_df['Month'] = self.intra_df['Month'].astype(int)
        self.intra_df['Year'] = self.intra_df['Year'].astype(int)
        self.intra_atr = round(self.intra_df.groupby(self.intra_df['Date'])['ATR_int'].mean(), 5)
        self.intra_vol = round(self.intra_df.groupby(self.intra_df['Date'])['Vol_int'].mean(), 5)

    def subset_daily_df(self):
        print('Subsetting Data to Workable Range')
        self.daily_working_df = self.daily_df.copy(deep=True)
        start_date = self.KmeansData.KParams.train_start_date - dt.timedelta(days=30)
        self.daily_working_df = (
            self.daily_working_df.loc)[(self.daily_working_df['Date'] > start_date)]
        self.daily_working_df = (
            self.daily_working_df.loc[self.daily_working_df['Date'] <=
                                      self.KmeansData.KParams.test_date].reset_index(drop=True))

    def merge_daily_intraday(self):
        print('Merging Input Data')
        self.daily_working_df = pd.merge(self.daily_working_df, self.intra_atr, on=['Date'], how='left')
        self.daily_working_df = pd.merge(self.daily_working_df, self.intra_vol, on=['Date'], how='left')
        self.daily_working_df = self.daily_working_df[~((self.daily_working_df['Year'] == 2004) &
                                                        self.daily_working_df['Month'].isin([3, 4, 5, 6, 7, 8]))]
        self.daily_working_df = self.daily_working_df.sort_values(by='Date')

        print(f'Final daily_working_df Columns: {self.daily_working_df.columns}')

    def finalize_daily_df(self):
        self.daily_working_df = gt.pad_months(self.daily_working_df)
        self.daily_working_df = math_tools.standardize_intradata(self.daily_working_df,
                                                                 self.KmeansData.KParams.sec_train,
                                                                 self.KmeansData.KParams.lookback)

        self.daily_working_df, rsi_cols = math_tools.create_rsi(self.daily_working_df,
                                                                self.KmeansData.KParams.other_sec,
                                                                self.KmeansData.KParams.sec_train)
        self.input_features += rsi_cols
        self.daily_working_df, self.input_features = (
            math_tools.add_high_low_diff(self.daily_working_df,
                                         self.KmeansData.KParams.other_sec,
                                         self.KmeansData.KParams.sec_train,
                                         self.input_features))
        self.daily_working_df = math_tools.get_open_close_diff(self.daily_working_df,
                                                               self.KmeansData.KParams.other_sec,
                                                               self.KmeansData.KParams.sec_train)
        self.input_features = gt.arrange_daily_df_cols(self.daily_working_df)
        self.daily_working_df = self.daily_working_df.loc[:, ['Date', 'Month', 'Year'] + self.input_features]
        self.daily_working_df = self.daily_working_df.sort_values(by='Date')


