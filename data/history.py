from re import L
from sqlalchemy import create_engine, Column, Integer, String, Float,UniqueConstraint
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import yaml
from logger import Logger
from Spider import SSQ, DLT, KL8
from data import create_matrix_from_db
from predict import LotteryPredictor
Base = declarative_base()

class PredictionResult(Base):
    __tablename__ = 'prediction_results'
    id = Column(Integer, primary_key=True, autoincrement=True)
    lottery_type = Column(String, nullable=False)
    issue = Column(String, nullable=False)
    front_numbers = Column(String, nullable=False)
    back_numbers = Column(String)
    probabilities = Column(String, nullable=False)
    
    # 添加 UNIQUE 约束
    __table_args__ = (
        UniqueConstraint('lottery_type', 'issue', name='unique_lottery_issue'),
    )

class LotteryPredictionStorage:
    def __init__(self, config_path='config.yaml'):
        self.config = self.load_config(config_path)
        self.logger = Logger(config_path)
        self.engine = create_engine(self.config['database']['db_url'])
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.predictor = LotteryPredictor(config_path)

    def load_config(self, config_path):
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

    def save_prediction(self, lottery_type, issue, front_numbers, back_numbers, probabilities):
        session = self.Session()
        try:
            front_numbers_str = ','.join(map(str, front_numbers))
            back_numbers_str = ','.join(map(str, back_numbers))
            probabilities_str = ';'.join([f"{num}:{prob:.6f}" for num, prob in probabilities])

            prediction_data = {
                'lottery_type': lottery_type,
                'issue': issue,
                'front_numbers': front_numbers_str,
                'back_numbers': back_numbers_str,
                'probabilities': probabilities_str
            }

            stmt = insert(PredictionResult).values(prediction_data)
            stmt = stmt.on_conflict_do_update(
                index_elements=['lottery_type', 'issue'],
                set_=prediction_data
            )

            session.execute(stmt)
            session.commit()
            self.logger.info(f"Saved prediction for {lottery_type} issue {issue}, front numbers: {front_numbers_str}, back numbers: {back_numbers_str}, probabilities: {probabilities_str}")
        except Exception as e:
            self.logger.error(f"Error saving prediction: {e}")
            session.rollback()
        finally:
            session.close()

    def get_last_predicted_issue(self, lottery_type):
        session = self.Session()
        try:
            last_prediction = session.query(PredictionResult).filter_by(lottery_type=lottery_type).order_by(PredictionResult.issue.desc()).first()
            return last_prediction.issue if last_prediction else None
        finally:
            session.close()

    def run_predictions(self, lottery_type):
        self.logger.info(f"Starting predictions for {lottery_type}")
        
        # 获取最后预测的期号
        last_predicted_issue = self.get_last_predicted_issue(lottery_type)
        
        # 从数据库获取矩阵数据
        matrix = create_matrix_from_db(lottery_type)
        
        # 获取最新的期号
        if lottery_type == 'ssq':
            latest_issue = self.Session().query(SSQ).order_by(SSQ.issue.desc()).first().issue
        elif lottery_type == 'dlt':
            latest_issue = self.Session().query(DLT).order_by(DLT.issue.desc()).first().issue
        elif lottery_type == 'kl8':
            latest_issue = self.Session().query(KL8).order_by(KL8.issue.desc()).first().issue
        else:
            raise ValueError(f"Unknown lottery type: {lottery_type}")
        
        # # 如果没有最后预测的期号，从最早的期号开始预测
        # if last_predicted_issue is None:
        start_issue = str(int(latest_issue) + 1).zfill(5)
        # else:
        #     start_issue = str(int(latest_issue) + 1).zfill(5)
        current_issue = start_issue
    
        self.logger.info(f"Predicting {lottery_type} issue {current_issue}")
            
            # 进行预测
        self.logger.info(f"Predicting {lottery_type} issue {current_issue} matrix: {matrix}")
        results = self.predictor.run_prediction(lottery_type, matrix)
        self.logger.info(f"Predicted {lottery_type} issue {current_issue} results: {results}")
            
        for _, front_numbers, back_numbers, prob_info in results:
                # 解析概率信息
                probabilities = []
                for line in prob_info[1:]:  # 跳过第一行的标题
                    if ':' in line:
                        num, prob = line.split(': ')
                        num = int(num.split()[-1])
                        prob = float(prob)
                        probabilities.append((num, prob))
                
                # 保存预测结果
                self.save_prediction(lottery_type, current_issue, front_numbers, back_numbers, probabilities)
            
            # 更新期号
        current_issue = str(int(current_issue) + 1).zfill(5)

        self.logger.info(f"Completed predictions for {lottery_type}")

# 使用示例
# storage = LotteryPredictionStorage()
# storage.run_predictions('ssq')
# storage.run_predictions('dlt')
# storage.run_predictions('kl8')