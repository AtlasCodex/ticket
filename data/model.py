'''
Author: AtlasCodex wenlin.xie@outlook.com
Date: 2024-07-02 00:12:31
LastEditors: AtlasCodex wenlin.xie@outlook.com
LastEditTime: 2024-07-02 11:22:07
FilePath: /ticket/data/model.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from sqlalchemy import create_engine
import yaml
from logger import Logger

class LotteryModel:
    def __init__(self, config_file):
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)

        self.logger = Logger(config_file).logger
        self.db_url = config['database']['db_url']
        self.ssq_config = config['ssq_model']
        self.dlt_config = config['dlt_model']

        self.logger.info("LotteryModel initialized with config file: %s", config_file)

        self.engine = create_engine(self.db_url)
        self.logger.info("Database connected.")

    def load_data(self):
        self.ssq_df = pd.read_sql_table('ssq', self.engine)
        self.dlt_df = pd.read_sql_table('dlt', self.engine)
        self.logger.info("Data loaded from database.")

    def preprocess_data(self):
        self.ssq_red_balls = self.ssq_df[['red1', 'red2', 'red3', 'red4', 'red5', 'red6']].values
        self.ssq_blue_ball = self.ssq_df[['blue']].values
        self.dlt_red_balls = self.dlt_df[['red1', 'red2', 'red3', 'red4', 'red5']].values
        self.dlt_blue_balls = self.dlt_df[['blue1', 'blue2']].values

        self.scaler_ssq_red = MinMaxScaler()
        self.scaler_ssq_blue = MinMaxScaler()
        self.scaler_dlt_red = MinMaxScaler()
        self.scaler_dlt_blue = MinMaxScaler()

        self.ssq_red_balls_scaled = self.scaler_ssq_red.fit_transform(self.ssq_red_balls)
        self.ssq_blue_ball_scaled = self.scaler_ssq_blue.fit_transform(self.ssq_blue_ball)
        self.dlt_red_balls_scaled = self.scaler_dlt_red.fit_transform(self.dlt_red_balls)
        self.dlt_blue_balls_scaled = self.scaler_dlt_blue.fit_transform(self.dlt_blue_balls)

        self.logger.info("Data preprocessing completed.")

    def create_dataset(self, dataset, time_step):
        dataX, dataY = [], []
        for i in range(len(dataset) - time_step - 1):
            a = dataset[i:(i + time_step), :]
            dataX.append(a)
            dataY.append(dataset[i + time_step, :])
        return np.array(dataX), np.array(dataY)

    def build_model(self, input_shape):
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(input_shape[1]))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def convert_to_lottery_numbers(self, predictions, ball_type):
        if ball_type == 'ssq_red':
            return np.round(predictions).astype(int).clip(1, 33)
        elif ball_type == 'ssq_blue':
            return np.round(predictions).astype(int).clip(1, 16)
        elif ball_type == 'dlt_red':
            return np.round(predictions).astype(int).clip(1, 35)
        elif ball_type == 'dlt_blue':
            return np.round(predictions).astype(int).clip(1, 12)