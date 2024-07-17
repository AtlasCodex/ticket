'''
Author: AtlasCodex wenlin.xie@outlook.com
Date: 2024-07-02 18:25:40
LastEditors: AtlasCodex wenlin.xie@outlook.com
LastEditTime: 2024-07-17 19:20:14
FilePath: /ticket/data/main.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import sys
import argparse
from logger import Logger
from history import LotteryPredictionStorage
# 初始化日志
logger = Logger("config.yaml").logger

from Spider import LotterySpider
from model import LotteryPredictionModel
from data import create_matrix_from_db
import yaml
from predict import LotteryPredictor

def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main(args,config):

    # 爬取数据
    spider = LotterySpider(config)
    spider.run()
    logger.info("双色球任务...")
    matrix = create_matrix_from_db(args.type)
    logger.info(matrix.shape)
    model = LotteryPredictionModel()
    logger.info("训练模型任务...")
    model.run(matrix, args.type)
    # logger.info("预测任务...")
    # storage = LotteryPredictionStorage()
    # storage.run_predictions(args.type)
    

       

if __name__ == "__main__":
    config_path = 'config.yaml'
    config = load_config(config_path)
    parser = argparse.ArgumentParser(description="Crawl, train, and predict lottery data.")
    parser.add_argument('type', type=str, help="Type of lottery: 'ssq' for Shuang Se Qiu, 'dlt' for Da Le Tou.")
    args = parser.parse_args()
    main(args,config)