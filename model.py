# model with regulated delays in order execution (open 2 sec, close 1 sec)

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import talib as ta
import numpy as np
import datetime
import logging
import pyodbc
import time
import plotly.graph_objects as go
import os
import mplfinance as mpf
import pandas as pd




mpl.use("agg")  # need this to avoid picture save error (after 2 thousands of saveing)

path_to_save = 'D:/sql_pic/session_116'

def mkdir():
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    if not os.path.exists(path_to_save+'/train'):
        os.makedirs(path_to_save+'/train')
    if not os.path.exists(path_to_save+'/train/0'):
        os.makedirs(path_to_save+'/train/0')
    if not os.path.exists(path_to_save+'/train/1'):
        os.makedirs(path_to_save+'/train/1')
    if not os.path.exists(path_to_save+'/train/2'):
        os.makedirs(path_to_save+'/train/2')

    if not os.path.exists(path_to_save+'/test'):
        os.makedirs(path_to_save+'/test')
    if not os.path.exists(path_to_save+'/test/0'):
        os.makedirs(path_to_save+'/test/0')
    if not os.path.exists(path_to_save+'/test/1'):
        os.makedirs(path_to_save+'/test/1')
    if not os.path.exists(path_to_save+'/test/2'):
        os.makedirs(path_to_save+'/test/2')
mkdir()

path_to_save = path_to_save + '/train'

expected_percentage = 0.5
quantity_to_react = 400

cnxn = pyodbc.connect("Driver={SQL Server Native Client 11.0};"
                      "Server=DESKTOP-LB73L9Q;"
                      "Database=Binance;"
                      "Trusted_Connection=yes;")


df_base = pd.read_sql(
    f"select  *,dateadd(S, [open_time], '1970-01-01') as time2 from candles_second where  _quantity >{quantity_to_react} order by open_time",
    cnxn)

for index, row in df_base.iterrows():
    front_from_i = row['open_time'] - 10 * 60   # 30 минут
    front_to_i = row['open_time']
    wave_from_i = row['open_time']
    wave_to_i = row['open_time'] + 10 * 60   # 10

    df = pd.read_sql(
        f"select  *,dateadd(S, [open_time], '1970-01-01') as time2 from candles_second where open_time >= {front_from_i} and open_time <= {front_to_i} order by open_time",
        cnxn)

    df_wave = pd.read_sql(
        f"select  *,dateadd(S, [open_time], '1970-01-01') as time2 from candles_second where open_time >= {wave_from_i} and open_time <= {wave_to_i} order by open_time",
        cnxn)


    df.drop('open_time', axis=1, inplace=True)
    df = df.rename(columns={'_high': 'High', '_low': 'Low', '_open': 'Open', '_close': 'Close', 'time2': 'Date', '_quantity': 'Volume'})
    df.set_index('Date', inplace=True)
    df.index.name = 'Date'


    print(df_wave.head(3))

    initial_price = df_wave.loc[0,]['_close']
    max_price = df_wave['_close'].max()
    min_price = df_wave['_close'].min()
    print('processing', path_to_save, row['open_time'], '.jpg  ')

    if (max_price - initial_price) / initial_price * 100 >= expected_percentage:  # rise more than 2%
        mpf.plot(df, type='candle', volume=True, savefig=f'{path_to_save}/1/{row["open_time"]}.jpg')
    elif abs((min_price - initial_price) / initial_price * 100) > expected_percentage:  # drop more than 2%
        mpf.plot(df, type='candle', volume=True, savefig=f'{path_to_save}/2/{row["open_time"]}.jpg')
    else:  # nothing happened
        mpf.plot(df, type='candle', volume=True, savefig=f'{path_to_save}/0/{row["open_time"]}.jpg')





