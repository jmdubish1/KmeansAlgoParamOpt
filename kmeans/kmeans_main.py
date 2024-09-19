import pandas as pd
import numpy as np
import general_tools.mkt_data_tools as mdt
import kmeans.kmeans_tools as kt
import os

os.environ["OMP_NUM_THREADS"] = '1'

kemans_param_dict = {
    'n_clusters': [16, 24, 32],
    'n_tests': 35,
    'security': 'NQ',
    'oth_securities': ['RTY', 'ES', 'YM'],
    'timeframe': '15min',
    'data_loc': r'C:\Users\jmdub\Documents\Trading\Futures\Strategy Info\data',
    'model_loc':
        r'C:\Users\jmdub\Documents\Trading\Futures\Strategy Info\Double Candles\NQ\15min\15min_test_13years\kmeans',
    'train_start_date': '2022-04-01',
    'test_date': '2024-04-01',
    'days_lookback': 28,
    'start_hour': 8
}


def main():
    kmeans_params = kt.KmeansParams(kemans_param_dict)
    kmeans_data = kt.KmeansData(kmeans_params)

    mkt_data = mdt.MktData(kmeans_data)
    mkt_data.load_intra_df()
    mkt_data.complete_intra_df_work()
    mkt_data.create_daily_df()
    mkt_data.subset_daily_df()
    mkt_data.merge_daily_intraday()
    mkt_data.finalize_daily_df()

    kmeans_data.add_mkt_data(mkt_data)
    kmeans_data.make_friday_list()
    kmeans_data.build_kmeans_df()
    kmeans_data.pad_df()
    kmeans_data.run_clusters()



if __name__ == '__main__':
    main()