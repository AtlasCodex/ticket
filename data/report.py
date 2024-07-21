import yaml
from sqlalchemy import create_engine, select,event
from sqlalchemy.orm import sessionmaker
from history import PredictionResult
from logger import Logger
from Spider import SSQ, DLT, KL8
import pyemail as email

class LotteryAnalysis:
    def __init__(self, config_path):
        self.logger = Logger(config_path)
        self.config = self.load_config(config_path)
        self.engine = create_engine(self.config['database']['db_url'])
        # 添加事件监听器来捕获 SQL 语句
        event.listen(self.engine, 'before_cursor_execute', self.log_sql)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()

    def load_config(self, config_path):
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    def log_sql(self, conn, cursor, statement, parameters, context, executemany):
        self.logger.debug(f"Executing SQL: {statement}")
        if parameters:
            self.logger.debug(f"Parameters: {parameters}")

    def get_prediction(self, lottery_type, issue):
        stmt = select(PredictionResult).where(
            PredictionResult.lottery_type == lottery_type,
            PredictionResult.issue == issue
        )
        return self.session.execute(stmt).scalar_one_or_none()
    
    def get_latest_issue_prediction(self, lottery_type):
        stmt = select(PredictionResult).where(
            PredictionResult.lottery_type == lottery_type
        ).order_by(PredictionResult.issue.desc()).limit(1)
        return self.session.execute(stmt).scalar_one_or_none()

    def get_actual_result(self, lottery_type, issue):
        table = {'ssq': SSQ, 'dlt': DLT, 'kl8': KL8}[lottery_type]
        stmt = select(table).where(table.issue == issue)
        return self.session.execute(stmt).scalar_one_or_none()

    def compare_numbers(self, predicted, actual, num_front, num_back=None):
        def to_int_set(numbers, count):
            return set(int(num) for num in numbers[:count])

        pred_front = to_int_set(predicted, num_front)
        actual_front = to_int_set(actual, num_front)
        matched_front = pred_front & actual_front
        
        if num_back:
            pred_back = to_int_set(predicted[-num_back:], num_back)
            actual_back = to_int_set(actual[-num_back:], num_back)
            matched_back = pred_back & actual_back
            return len(matched_front), len(matched_back)
        else:
            return len(matched_front)

    def determine_prize(self, lottery_type, matched_front, matched_back=None):
        if lottery_type == 'ssq':  # 双色球
            if matched_front == 6 and matched_back == 1:
                return "一等奖"
            elif matched_front == 6:
                return "二等奖"
            elif matched_front == 5 and matched_back == 1:
                return "三等奖"
            elif matched_front == 5 or (matched_front == 4 and matched_back == 1):
                return "四等奖"
            elif matched_front == 4 or (matched_front == 3 and matched_back == 1):
                return "五等奖"
            elif matched_back == 1:
                return "六等奖"
        
        elif lottery_type == 'dlt':  # 大乐透
            if matched_front == 5 and matched_back == 2:
                return "一等奖"
            elif matched_front == 5 and matched_back == 1:
                return "二等奖"
            elif matched_front == 5:
                return "三等奖"
            elif matched_front == 4 and matched_back == 2:
                return "四等奖"
            elif (matched_front == 4 and matched_back == 1) or (matched_front == 3 and matched_back == 2):
                return "五等奖"
            elif (matched_front == 4) or (matched_front == 3 and matched_back == 1) or (matched_front == 2 and matched_back == 2):
                return "六等奖"
            elif (matched_front == 3) or (matched_front == 2 and matched_back == 1) or (matched_front == 1 and matched_back == 2) or (matched_back == 2):
                return "七等奖"
        
        elif lottery_type == 'kl8':  # 快乐8
            total_matched = matched_front  # 快乐8只有前区
            if 1 <= total_matched <= 10:
                return f"{total_matched}中{total_matched}"
            elif total_matched in [11, 12]:
                return f"任{total_matched}中{total_matched}"
            elif 15 <= total_matched <= 20:
                return f"选{20}中{total_matched}"
        
        return "未中奖"
        
    def analyze_prediction(self, lottery_type, issue):
        prediction = self.get_prediction(lottery_type, issue)
        actual = self.get_actual_result(lottery_type, issue)

        if not prediction or not actual:
            self.logger.error(f"无法获取 {lottery_type} 第 {issue} 期的预测或实际结果")
            return

        pred_numbers = prediction.front_numbers.split(',') + (prediction.back_numbers.split(',') if prediction.back_numbers else [])

        if lottery_type =='ssq':
            actual_numbers = [actual.red1, actual.red2, actual.red3, actual.red4, actual.red5, actual.red6, actual.blue]
            matched_front, matched_back = self.compare_numbers(pred_numbers, actual_numbers, 6, 1)
            prize = self.determine_prize('ssq', matched_front, matched_back)

            result = f"双色球第 {issue} 期分析结果:\n"
            result += f"预测号码: {prediction.front_numbers}, {prediction.back_numbers}\n"
            result += f"实际号码: {','.join(actual_numbers[:-1])}, {actual_numbers[-1]}\n"
            result += f"匹配结果: 红球 {matched_front}/6, 蓝球 {matched_back}/1\n"
            result += f"中奖结果: {prize}\n"

            # 分析前 6、7、8 个红球
            for i in range(6, 9):
                matched = self.compare_numbers(pred_numbers, actual_numbers, i)
                result += f"前{i}个红球匹配: {matched}/{i}\n"

            # 分析前 2、3 个蓝球
            for i in range(2, 4):
                matched = self.compare_numbers(pred_numbers[-i:], actual_numbers[-i:], i)
                result += f"前{i}个蓝球匹配: {matched}/{i}\n"

        elif lottery_type == 'dlt':
            actual_numbers = [actual.red1, actual.red2, actual.red3, actual.red4, actual.red5, actual.blue1, actual.blue2]
            matched_front, matched_back = self.compare_numbers(pred_numbers, actual_numbers, 5, 2)
            prize = self.determine_prize('dlt', matched_front, matched_back)

            result = f"大乐透第 {issue} 期分析结果:\n"
            result += f"预测号码: {prediction.front_numbers}, {prediction.back_numbers}\n"
            result += f"实际号码: {','.join(actual_numbers[:5])}, {','.join(actual_numbers[5:])}\n"
            result += f"匹配结果: 前区 {matched_front}/5, 后区 {matched_back}/2\n"
            result += f"中奖结果: {prize}\n"

            # 分析前 5、6、7 个前区号码
            for i in range(5, 8):
                matched = self.compare_numbers(pred_numbers, actual_numbers, i)
                result += f"前区前{i}个号码匹配: {matched}/{i}\n"

            # 分析前 2、3 个后区号码
            for i in range(2, 4):
                matched = self.compare_numbers(pred_numbers[-i:], actual_numbers[-i:], i)
                result += f"后区前{i}个号码匹配: {matched}/{i}\n"

        elif lottery_type == 'kl8':
            actual_numbers = [getattr(actual, f'red{i}') for i in range(1, 21)]
            matched = self.compare_numbers(pred_numbers, actual_numbers, 20)

            result = f"快乐 8 第 {issue} 期分析结果:\n"
            result += f"预测号码: {prediction.front_numbers}\n"
            result += f"实际号码: {','.join(actual_numbers)}\n"
            result += f"匹配结果: {matched}/20\n"

            # 分析前 10、15、20 个号码
            for i in [5,7,8,10, 15, 20]:
                matched = self.compare_numbers(pred_numbers, actual_numbers, i)
                result += f"前{i}个号码匹配: {matched}/{i}\n"

        historical_matches = self.check_historical_matches(lottery_type, pred_numbers)
        result += "历史匹配情况：\n"
        for match in historical_matches:
            result += str(match) + '\n'

        return result
        
    
    def check_historical_matches(self, lottery_type, predicted_numbers, limit=5):
            table = {'ssq': SSQ, 'dlt': DLT, 'kl8': KL8}[lottery_type]
            
            # 根据彩票类型确定前区和后区号码数量及列名
            if lottery_type == 'ssq':
                front_count, back_count = 6, 1
                front_cols = [f'red{i}' for i in range(1, front_count + 1)]
                back_cols = ['blue']
            elif lottery_type == 'dlt':
                front_count, back_count = 5, 2
                front_cols = [f'red{i}' for i in range(1, front_count + 1)]
                back_cols = [f'blue{i}' for i in range(1, back_count + 1)]
            elif lottery_type == 'kl8':
                front_count, back_count = 20, 0
                front_cols = [f'red{i}' for i in range(1, front_count + 1)]
                back_cols = []
            
            pred_front = set(int(num) for num in predicted_numbers[:front_count])
            pred_back = set(int(num) for num in predicted_numbers[-back_count:]) if back_count > 0 else set()
            
            # 构建查询
            columns = [getattr(table, col) for col in front_cols + back_cols]
            
            stmt = select(table.issue, *columns)
            results = self.session.execute(stmt).fetchall()
            
            matches = []
            max_front_match = 0
            max_back_match = 0
            for result in results:
                issue = result[0]
                actual_front = set(map(int, result[1:front_count + 1]))
                actual_back = set(map(int, result[front_count + 1:])) if back_count > 0 else set()

                front_match = len(pred_front & actual_front)
                back_match = len(pred_back & actual_back) if back_count > 0 else 0

                if front_match > max_front_match or (front_match == max_front_match and back_match > max_back_match):
                    max_front_match = front_match
                    max_back_match = back_match
                    matches = [(issue, front_match, back_match)]
                elif front_match == max_front_match and back_match == max_back_match:
                    matches.append((issue, front_match, back_match))
            
            # 按匹配百分比排序，返回前limit个结果
            return matches

    def run_analysis(self, lottery_type, start_issue, end_issue):
        for issue in range(int(start_issue), int(end_issue) + 1):
            result = self.analyze_prediction(lottery_type, str(issue))
            return result

if __name__ == "__main__":
    analyzer = LotteryAnalysis('config.yaml')
    pre = analyzer.get_latest_issue_prediction('dlt')
    pred_numbers = pre.front_numbers.split(',') + (pre.back_numbers.split(',') if pre.back_numbers else [])
    historical_matches = analyzer.check_historical_matches('dlt', pred_numbers)
    print(historical_matches)
    print(pred_numbers)
    result = analyzer.run_analysis('dlt', '24082', '24083')
    print(result)

    # 邮件配置
    sender_email = "wenlin_x@163.com"
    sender_password = "DFOZWXHXIQFKITDP"
    recipient_email = "wenlin.xie@foxmail.com"
    subject = "彩票分析结果"

    email.send_lottery_email(sender_email, sender_password, recipient_email, subject, historical_matches, pred_numbers, result)
    
    # analyzer.run_analysis('ssq', '24082', '24083')
    # analyzer.run_analysis('kl8', '2024191', '2024192')