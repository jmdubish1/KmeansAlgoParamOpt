import numpy as np
import pandas as pd
import datetime as dt
from sklearn.cluster import KMeans
import kmeans.kmeans_regimes as kr
import general_tools.basic_tools as bt
import general_tools.trade_math_tools as tmt
import os
pd.options.mode.chained_assignment = None  # default='warn'


os.environ["OMP_NUM_THREADS"] = '1'


class KmeansParams:
    def __init__(self, kmeans_param_dict):
        self.n_clusters = kmeans_param_dict['n_clusters']
        self.n_test = kmeans_param_dict['n_tests']
        self.sec_train = kmeans_param_dict['security']
        self.other_sec = kmeans_param_dict['oth_securities']
        self.train_start_date = dt.datetime.strptime(kmeans_param_dict['train_start_date'], "%Y-%m-%d").date()
        self.test_date = dt.datetime.strptime(kmeans_param_dict['test_date'], "%Y-%m-%d").date()
        self.lookback = kmeans_param_dict['days_lookback']
        self.start_hour = kmeans_param_dict['start_hour']
        self.time_frame = kmeans_param_dict['timeframe']
        self.model_loc = kmeans_param_dict['model_loc']
        self.data_loc = kmeans_param_dict['data_loc']
        self.intraday_file = self.set_intraday_file()
        self.regimes = None

    def set_intraday_file(self):
        return f'{self.data_loc}\\{self.sec_train}_{self.time_frame}_20240505_20040401.txt'


class KmeansData:
    def __init__(self, kmeans_params):
        self.KParams = kmeans_params
        self.MktData = None
        self.friday_list = []
        self.kmeans_df = []
        self.train_df = pd.DataFrame
        self.test_df = pd.DataFrame

    def add_mkt_data(self, mkt_data):
        """
        Add the class MktData to KmeansData
        :param mkt_data: MktData class
        """

        self.MktData = mkt_data

    def make_friday_list(self):
        """
        Makes a list of Fridays that the model will iterate through
        """

        friday_list = []
        current_date = self.KParams.train_start_date - dt.timedelta(days=7)
        while current_date.weekday() != 4:
            current_date += dt.timedelta(days=1)

        while current_date <= self.KParams.test_date:
            friday_list.append(current_date)
            current_date += dt.timedelta(days=7)

        self.friday_list = friday_list

    def build_kmeans_df(self):
        """
        Applies transformations to the mkt_data in order to make the kmeans dataframe
        :return:
        """
        mkt_df = self.MktData.daily_working_df
        self.kmeans_df = []
        for friday in self.friday_list:
            temp_df = mkt_df[(mkt_df['Date'] <= friday) &
                             (mkt_df['Date'] > friday - dt.timedelta(days=self.KParams.lookback))].reset_index(drop=True)

            temp_df, self.MktData.input_features = tmt.scale_open_close(temp_df,
                                                                        self.KParams.sec_train,
                                                                        self.KParams.other_sec,
                                                                        self.MktData.input_features)
            temp_df.fillna(0, inplace=True)
            temp_df.replace([np.inf, -np.inf], 0, inplace=True)
            self.kmeans_df.append(temp_df)
        print(len(self.kmeans_df))
        # self.kmeans_df = pd.concat(self.kmeans_df, ignore_index=True).reset_index(drop=True)

    def pad_df(self):
        temp_list = []
        for df in self.kmeans_df:
            temp_list.append(bt.pad_df(df, self.KParams.lookback))
        self.kmeans_df = temp_list

    def run_clusters(self):
        flattened_data = [df.iloc[:, 3:].values.flatten() for df in self.kmeans_df]
        X = np.array(flattened_data)
        for _ in range(10):
            kmeans = KMeans(n_clusters=16)
            # print(self.kmeans_df.columns)
            kmeans.fit(X)
            labels = kmeans.labels_
            print(labels)
        print(self.friday_list)
