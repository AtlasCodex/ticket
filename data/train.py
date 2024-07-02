'''
Author: AtlasCodex wenlin.xie@outlook.com
Date: 2024-07-02 11:09:48
LastEditors: AtlasCodex wenlin.xie@outlook.com
LastEditTime: 2024-07-02 11:31:56
FilePath: /ticket/data/train.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from model import LotteryModel
import sys

def train_model(model, X, Y, model_save_path, config):
    model = model.build_model((config['time_step'], X.shape[2]))
    model.fit(X, Y, epochs=config['epochs'], batch_size=config['batch_size'], validation_split=config['validation_split'], verbose=1)
    model.save(model_save_path)
    return model

def train_ssq(lottery_model):
    ssq_X_red, ssq_Y_red = lottery_model.create_dataset(lottery_model.ssq_red_balls_scaled, lottery_model.ssq_config['red']['time_step'])
    ssq_X_blue, ssq_Y_blue = lottery_model.create_dataset(lottery_model.ssq_blue_ball_scaled, lottery_model.ssq_config['blue']['time_step'])

    ssq_red_model_path = lottery_model.ssq_config['red']['model_path']
    ssq_blue_model_path = lottery_model.ssq_config['blue']['model_path']

    train_model(lottery_model, ssq_X_red, ssq_Y_red, ssq_red_model_path, lottery_model.ssq_config['red'])
    train_model(lottery_model, ssq_X_blue, ssq_Y_blue, ssq_blue_model_path, lottery_model.ssq_config['blue'])

def train_dlt(lottery_model):
    dlt_X_red, dlt_Y_red = lottery_model.create_dataset(lottery_model.dlt_red_balls_scaled, lottery_model.dlt_config['red']['time_step'])
    dlt_X_blue, dlt_Y_blue = lottery_model.create_dataset(lottery_model.dlt_blue_balls_scaled, lottery_model.dlt_config['blue']['time_step'])

    dlt_red_model_path = lottery_model.dlt_config['red']['model_path']
    dlt_blue_model_path = lottery_model.dlt_config['blue']['model_path']

    train_model(lottery_model, dlt_X_red, dlt_Y_red, dlt_red_model_path, lottery_model.dlt_config['red'])
    train_model(lottery_model, dlt_X_blue, dlt_Y_blue, dlt_blue_model_path, lottery_model.dlt_config['blue'])

def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ['ssq', 'dlt']:
        print("Usage: python train.py [ssq|dlt]")
        sys.exit(1)

    lottery_type = sys.argv[1]
    lottery_model = LotteryModel('config.yaml')
    lottery_model.load_data()
    lottery_model.preprocess_data()

    if lottery_type == 'ssq':
        train_ssq(lottery_model)
    else:
        train_dlt(lottery_model)

if __name__ == "__main__":
    main()