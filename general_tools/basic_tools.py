import numpy as np
import pandas as pd


def flatten_list(a_list):
    return [x for xs in a_list for x in xs]


def convert_date_time(data):
    data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'])
    data['Date'] = pd.to_datetime(data['Date']).dt.date
    data['Month'] = data['Datetime'].dt.month
    data['Year'] = data['Datetime'].dt.year

    return data


def pad_months(data):
    full_df = []
    for year in data['Year'].unique():
        for month in data['Month'].unique():
            month_df = data.loc[(data['Year'] == year) &
                                (data['Month'] == month)]
            rows_2_add = 23 - len(month_df) + 1
            add_row_beginning = []
            add_row_end = []
            if not month_df.empty:
                if len(month_df) < 23:
                    for i in range(1, rows_2_add):
                        if i % 2 == 1:
                            add_row_beginning.append(list(month_df.iloc[0]))
                        else:
                            add_row_end.append(list(month_df.iloc[-1]))

                    add_row_beginning = pd.DataFrame(add_row_beginning)
                    add_row_beginning.columns = month_df.columns
                    month_df = pd.concat([add_row_beginning, month_df])

                    if len(add_row_end) > 0:
                        add_row_end = pd.DataFrame(add_row_end)
                        add_row_end.columns = month_df.columns
                        month_df = pd.concat([month_df, add_row_end])
                full_df.append(month_df)

    full_df = pd.concat(full_df).reset_index(drop=True)

    return full_df


def pad_df(df, target_length=28):
    current_len = len(df)
    if current_len < target_length:
        rows_to_add = target_length - current_len
        first_row = df.iloc[0]
        repeated_rows = pd.DataFrame([first_row]*rows_to_add, columns=df.columns)
        df = pd.concat([repeated_rows, df], ignore_index=True).reset_index(drop=True)

    return df

def repeat_array(data, lookback):
    arr = np.repeat(data, lookback, axis=0)
    return arr


def arrange_daily_df_cols(input_features):
    new_arrange = ['ATR_int', 'Vol_int']
    suffixes = ['_Open', '_Close', '_Open_Scale', '_Close_Scale',  '_HL_diff', '_ATR_day', '_RSI_k',
                '_RSI_d', '_Vol', '_VolAvg', '_OI', '_OpenInt']
    for suffix in suffixes:
        for col_name in input_features:
            if col_name.endswith(suffix):
                new_arrange.append(col_name)

    return new_arrange


def roll_back_param_results(df):
    df['month'] -= 1
    month_zero_mask = df['month'] == 0
    df.loc[month_zero_mask, 'month'] = 12
    df.loc[month_zero_mask, 'year'] -= 1

    return df
