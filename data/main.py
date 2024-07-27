'''
Author: AtlasCodex wenlin.xie@outlook.com
Date: 2024-07-21 14:34:13
LastEditors: AtlasCodex wenlin.xie@outlook.com
LastEditTime: 2024-07-28 00:27:38
FilePath: /ticket/main.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import base64
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
import schedule
import time
import os
from datetime import datetime, time as dt_time

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
    analyzer = LotteryAnalysis('config.yaml')
    pre = analyzer.get_latest_issue_prediction(name)
    pred_numbers = pre.front_numbers.split(',') + (pre.back_numbers.split(',') if pre.back_numbers else [])
    result = analyzer.run_analysis(name, pre.issue, pre.issue)
    historical_matches = analyzer.check_historical_matches(name, pred_numbers)

    # 读取训练绘制图并转换为 base64
    plot_path = config['model_params'][name]['plot_save_path']
    with open(plot_path, "rb") as image_file:
        training_plot = base64.b64encode(image_file.read()).decode('utf-8')
    # 邮件配置
    sender_email = config['email']['sender']
    sender_password = config['email']['password']
    recipient_email = config['email']['recipient']
    if name == 'dlt':
        subject = f"大乐透{pre.issue} 彩票分析结果"
    elif name == 'ssq':
        subject = f"双色球{pre.issue} 彩票分析结果"
    elif name == 'kl8':
        subject = f"快乐8{pre.issue} 彩票分析结果"

    email.send_lottery_email(sender_email, sender_password, recipient_email, name,subject, historical_matches, pre.front_numbers, pre.back_numbers, result,training_plot)

# 记录任务的最后执行时间
last_run_times = {
    'dlt': None,
    'ssq': None,
    'kl8': None
}

# # 定义每周 1、3、6 运行的任务
def combined_task():
    today = datetime.today().weekday()
    current_time = datetime.now()
    if today in [0, 2, 5]:  # 周一、周三、周六
        if last_run_times['ssq'] is None or (current_time - last_run_times['ssq']).days >= 1:
            run(load_config('config.yaml'),'dlt')
            sendReport(load_config('config.yaml'),'dlt')
            last_run_times['ssq'] = current_time
    elif today in [1, 3, 6]:  # 周二、周四、周日
        if last_run_times['dlt'] is None or (current_time - last_run_times['dlt']).days >= 1:
            run(load_config('config.yaml'),'ssq')
            sendReport(load_config('config.yaml'),'ssq')
            last_run_times['dlt'] = current_time # 每天
    if last_run_times['kl8'] is None or (current_time - last_run_times['kl8']).days >= 1:
            run(load_config('config.yaml'),'kl8')
            sendReport(load_config('config.yaml'),'kl8')
            last_run_times['kl8'] = current_time

# 设置在12:00执行的函数
def run_at_specific_time():
    current_time = datetime.now().time()
    if dt_time(1, 30) <= current_time < dt_time(1,  33):
        combined_task()

# 每分钟检查一次是否到达指定时间
schedule.every().minute.do(run_at_specific_time)

while True:
    schedule.run_pending()
    time.sleep(1)
