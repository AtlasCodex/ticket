'''
Author: AtlasCodex wenlin.xie@outlook.com
Date: 2024-07-02 11:10:13
LastEditors: AtlasCodex wenlin.xie@outlook.com
LastEditTime: 2024-07-02 11:14:07
FilePath: /ticket/data/predict.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from model import LotteryModel
from tensorflow.keras.models import load_model
import os
import sys

from logger import Logger

# 初始化日志
logger = Logger('config.yaml').logger

def load_model_file(model_path, logger):
    if os.path.exists(model_path):
        logger.info("Loading model from %s", model_path)
        return load_model(model_path)
    else:
        logger.warning("Model file %s not found.", model_path)
        return None

def predict(model, X, scaler):
    predictions = model.predict(X)
    return scaler.inverse_transform(predictions)

def predict_ssq(lottery_model):
    ssq_X_red, _ = lottery_model.create_dataset(lottery_model.ssq_red_balls_scaled, lottery_model.ssq_config['red']['time_step'])
    ssq_X_blue, _ = lottery_model.create_dataset(lottery_model.ssq_blue_ball_scaled, lottery_model.ssq_config['blue']['time_step'])

    ssq_red_model = load_model_file(lottery_model.ssq_config['red']['model_path'], lottery_model.logger)
    ssq_blue_model = load_model_file(lottery_model.ssq_config['blue']['model_path'], lottery_model.logger)

    if ssq_red_model and ssq_blue_model:
        ssq_red_predictions = predict(ssq_red_model, ssq_X_red, lottery_model.scaler_ssq_red)
        ssq_blue_predictions = predict(ssq_blue_model, ssq_X_blue, lottery_model.scaler_ssq_blue)

        ssq_red_numbers = lottery_model.convert_to_lottery_numbers(ssq_red_predictions, 'ssq_red')
        ssq_blue_numbers = lottery_model.convert_to_lottery_numbers(ssq_blue_predictions, 'ssq_blue')

        lottery_model.logger.info("SSQ Red Ball Predictions: %s", ssq_red_numbers[-1])
        lottery_model.logger.info("SSQ Blue Ball Predictions: %s", ssq_blue_numbers[-1][0])
    else:
        lottery_model.logger.error("SSQ models could not be loaded. Please train the models first.")

def predict_dlt(lottery_model):
    dlt_X_red, _ = lottery_model.create_dataset(lottery_model.dlt_red_balls_scaled, lottery_model.dlt_config['red']['time_step'])
    dlt_X_blue, _ = lottery_model.create_dataset(lottery_model.dlt_blue_balls_scaled, lottery_model.dlt_config['blue']['time_step'])

    dlt_red_model = load_model_file(lottery_model.dlt_config['red']['model_path'], lottery_model.logger)
    dlt_blue_model = load_model_file(lottery_model.dlt_config['blue']['model_path'], lottery_model.logger)

    if dlt_red_model and dlt_blue_model:
        dlt_red_predictions = predict(dlt_red_model, dlt_X_red, lottery_model.scaler_dlt_red)
        dlt_blue_predictions = predict(dlt_blue_model, dlt_X_blue, lottery_model.scaler_dlt_blue)

        dlt_red_numbers = lottery_model.convert_to_lottery_numbers(dlt_red_predictions, 'dlt_red')
        dlt_blue_numbers = lottery_model.convert_to_lottery_numbers(dlt_blue_predictions, 'dlt_blue')

        lottery_model.logger.info("DLT Red Ball Predictions: %s", dlt_red_numbers[-1])
        lottery_model.logger.info("DLT Blue Ball Predictions: %s", dlt_blue_numbers[-1])
    else:
        lottery_model.logger.error("DLT models could not be loaded. Please train the models first.")

def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ['ssq', 'dlt']:
        print("Usage: python predict.py [ssq|dlt]")
        sys.exit(1)

    lottery_type = sys.argv[1]
    lottery_model = LotteryModel('config.yaml')
    lottery_model.load_data()
    lottery_model.preprocess_data()

    if lottery_type == 'ssq':
        predict_ssq(lottery_model)
    else:
        predict_dlt(lottery_model)

if __name__ == "__main__":
    main()