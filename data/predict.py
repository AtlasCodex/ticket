'''
Author: AtlasCodex wenlin.xie@outlook.com
Date: 2024-07-02 11:10:13
LastEditors: AtlasCodex wenlin.xie@outlook.com
LastEditTime: 2024-07-17 16:15:55
FilePath: /ticket/data/predict.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from jinja2 import Environment, FileSystemLoader
import os
import yaml
from logger import Logger

class LotteryPredictor:
    def __init__(self, config_path='config.yaml'):
        self.config = self.load_config(config_path)
        self.logger = Logger(config_path)
        self.models = {}
        self.scalers = {}
        self.window_sizes = {}

    def load_config(self, config_path):
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

    def load_model(self, name):
        if name not in self.models:
            model_params = self.config['model_params'][name]
            model_path = model_params['finalmodel_paths']
            self.models[name] = tf.keras.models.load_model(model_path)
            self.logger.info(f"Loaded model for {name}")
        return self.models[name]

    def load_scaler(self, name, matrix):
        if name not in self.scalers:
            self.scalers[name] = MinMaxScaler()
            self.scalers[name].fit(matrix)
            self.logger.info(f"Fitted scaler for {name}")
        return self.scalers[name]

    def get_window_size(self, name):
        if name not in self.window_sizes:
            self.window_sizes[name] = self.config['model_params'][name]['window_size']
        return self.window_sizes[name]

    def predict(self, name, matrix):
        model = self.load_model(name)
        scaler = self.load_scaler(name, matrix)
        window_size = self.get_window_size(name)

        new_data = matrix[-window_size:]
        new_data_scaled = scaler.transform(new_data)
        input_data = new_data_scaled.reshape((1, window_size, matrix.shape[1]))

        prediction_scaled = model.predict(input_data)
        prediction = scaler.inverse_transform(prediction_scaled)

        return prediction[0]

    def process_prediction(self, name, prediction):
        if name == 'ssq':
            front_area = prediction[:33]
            back_area = prediction[33:]
            front_indices = np.argsort(front_area)[::-1][:8]
            back_indices = np.argsort(back_area)[::-1][:3]
        elif name == 'dlt':
            front_area = prediction[:35]
            back_area = prediction[35:]
            front_indices = np.argsort(front_area)[::-1][:7]
            back_indices = np.argsort(back_area)[::-1][:3]
        elif name == 'kl8':
            front_indices = np.argsort(prediction)[::-1][:20]
            back_indices = np.array([])
        else:
            self.logger.error(f"Unknown lottery name: {name}")
            raise ValueError(f"Unknown lottery name: {name}")

        front_numbers = front_indices + 1
        back_numbers = back_indices + 1

        return front_numbers, back_numbers

    def print_probabilities(self, name, prediction, front_numbers, back_numbers):
        prob_info = []
        prob_info.append(f"{name.upper()} 预测结果：")
        prob_info.append("前区号码及其概率：")
        for num in front_numbers:
            prob_info.append(f"号码 {num}: {prediction[num-1]:.6f}")
        if len(back_numbers) > 0:
            prob_info.append("后区号码及其概率：")
            for num in back_numbers:
                if name == 'ssq':
                    prob_info.append(f"号码 {num}: {prediction[num+32]:.6f}")
                elif name == 'dlt':
                    prob_info.append(f"号码 {num}: {prediction[num+34]:.6f}")
        return prob_info

    def multi_step_prediction(self, name, matrix, steps):
        model = self.load_model(name)
        scaler = self.load_scaler(name, matrix)
        window_size = self.get_window_size(name)

        initial_input = scaler.transform(matrix[-window_size:]).reshape((1, window_size, matrix.shape[1]))
        predictions = []
        current_input = initial_input

        for _ in range(steps):
            pred_scaled = model.predict(current_input)
            pred = scaler.inverse_transform(pred_scaled)
            predictions.append(pred[0])

            current_input = np.roll(current_input, -1, axis=1)
            current_input[0, -1] = pred_scaled[0]

        return np.array(predictions)

    def run_prediction(self, name, matrix, multi_step=False, steps=1):
        self.logger.info(f"Starting prediction for {name}")
        
        if multi_step:
            predictions = self.multi_step_prediction(name, matrix, steps)
            results = []
            for i, pred in enumerate(predictions):
                front_numbers, back_numbers = self.process_prediction(name, pred)
                prob_info = self.print_probabilities(name, pred, front_numbers, back_numbers)
                results.append((f"Step {i+1}", front_numbers, back_numbers, prob_info))
        else:
            prediction = self.predict(name, matrix)
            front_numbers, back_numbers = self.process_prediction(name, prediction)
            prob_info = self.print_probabilities(name, prediction, front_numbers, back_numbers)
            results = [("Single Step", front_numbers, back_numbers, prob_info)]

        self.logger.info(f"Completed prediction for {name}")
        return results

# 使用示例
# predictor = LotteryPredictor()
# matrix = create_matrix_from_db('ssq')  # 假设这个函数已经定义
# results = predictor.run_prediction('ssq', matrix, multi_step=True, steps=3)
# for step, front, back, prob_info in results:
#     print(f"\n{step}")
#     print(f"前区号码: {front}")
#     print(f"后区号码: {back}")
#     for info in prob_info:
#         print(info)

