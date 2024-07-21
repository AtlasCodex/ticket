'''
Author: AtlasCodex wenlin.xie@outlook.com
Date: 2024-07-21 14:34:13
LastEditors: AtlasCodex wenlin.xie@outlook.com
LastEditTime: 2024-07-21 14:54:02
FilePath: /ticket/main.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import schedule
import time
import yaml
from  logger import Logger
from  history import LotteryPredictionStorage
from  Spider import LotterySpider
from  model import LotteryPredictionModel
from  data import create_matrix_from_db
import yaml
from  predict import LotteryPredictor
from  report import LotteryAnalysis
import pyemail as email

# 初始化日志
logger = Logger("config.yaml").logger

def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def run(config,name):
    spider = LotterySpider(config)
    spider.run()
    logger.info("双色球任务...")
    matrix = create_matrix_from_db(name)
    logger.info(matrix.shape)
    model = LotteryPredictionModel()
    logger.info("训练模型任务...")
    model.run(matrix, name)
    logger.info("预测任务...")
    storage = LotteryPredictionStorage()
    storage.run_predictions(name)

def sendReport(config,name):
    analyzer = LotteryAnalysis(config)
    pre = analyzer.get_latest_issue_prediction(name)
    pred_numbers = pre.front_numbers.split(',') + (pre.back_numbers.split(',') if pre.back_numbers else [])
    result = analyzer.run_analysis(name, pre.issue-1, pre.issue)
    historical_matches = analyzer.check_historical_matches(name, pred_numbers)
    # 邮件配置
    sender_email = "wenlin_x@163.com"
    sender_password = "DFOZWXHXIQFKITDP"
    recipient_email = "wenlin.xie@foxmail.com"
    subject = f"{pre.issue}彩票分析结果"

    email.send_lottery_email(sender_email, sender_password, recipient_email, subject, historical_matches, pred_numbers, result)
    

# 定义每周 1、3、6 运行的任务
def task_136():
    run(load_config('config.yaml'),'dlt')
    sendReport(load_config('config.yaml'),'dlt')

# 定义每周 2、4、7 运行的任务
def task_247():
    run(load_config('config.yaml'),'ssq')
    sendReport(load_config('config.yaml'),'ssq')

# 定义每天都运行的任务
def task_everyday():
    run(load_config('config.yaml'),'kl8')
    sendReport(load_config('config.yaml'),'kl8')

# 设置定时任务
schedule.every().monday.at("15:00").do(task_136)
schedule.every().wednesday.at("15:00").do(task_136)
schedule.every().saturday.at("15:00").do(task_136)

schedule.every().tuesday.at("15:00").do(task_247)
schedule.every().thursday.at("15:00").do(task_247)
schedule.every().sunday.at("15:00").do(task_247)

schedule.every().day.at("16:00").do(task_everyday)

while True:
    schedule.run_pending()
    time.sleep(1)