'''
Author: AtlasCodex wenlin.xie@outlook.com
Date: 2024-07-17 15:07:26
LastEditors: AtlasCodex wenlin.xie@outlook.com
LastEditTime: 2024-07-17 15:17:03
FilePath: /ticket/data/data.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import yaml
from logger import Logger
from Spider import SSQ, DLT, KL8

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def create_matrix_from_db(name, config_path='config.yaml'):
    config = load_config(config_path)
    db_url = config['database']['db_url']
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Initialize logger
    logger = Logger(config_path)
    logger.info(f"Starting to create matrix for {name}")

    if name == 'ssq':
        query = session.query(SSQ).order_by(SSQ.issue)
        num_rows = 33 + 16  # 33 红球 + 16 蓝球
    elif name == 'dlt':
        query = session.query(DLT).order_by(DLT.issue)
        num_rows = 35 + 12  # 35 红球 + 12 蓝球
    elif name == 'kl8':
        query = session.query(KL8).order_by(KL8.issue)
        num_rows = 80  # 80 红球
    else:
        logger.error(f"Unknown lottery name: {name}")
        raise ValueError(f"Unknown lottery name: {name}")

    results = query.all()
    num_issues = len(results)
    logger.info(f"Retrieved {num_issues} records for {name}")

    matrix = np.zeros((num_issues, num_rows), dtype=int)

    for i, row in enumerate(results):
        if name == 'ssq':
            for j in range(1, 7):
                red_ball = int(getattr(row, f'red{j}'))
                matrix[i, red_ball - 1] = 1
            blue_ball = int(row.blue)
            matrix[i, 33 + blue_ball - 1] = 1
        elif name == 'dlt':
            for j in range(1, 6):
                red_ball = int(getattr(row, f'red{j}'))
                matrix[i, red_ball - 1] = 1
            for j in range(1, 3):
                blue_ball = int(getattr(row, f'blue{j}'))
                matrix[i, 35 + blue_ball - 1] = 1
        elif name == 'kl8':
            for j in range(1, 21):
                red_ball = int(getattr(row, f'red{j}'))
                matrix[i, red_ball - 1] = 1

    session.close()
    logger.info(f"Matrix creation completed for {name}. Matrix shape: {matrix.shape}")
    return matrix

if __name__ == '__main__':
# 使用示例
    matrix = create_matrix_from_db('dlt')