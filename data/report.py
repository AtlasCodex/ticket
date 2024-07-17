from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from typing import List, Tuple, Dict
from logger import Logger  # 导入自定义的日志模块
from Spider import SSQ, DLT, KL8
from history import PredictionResult
import yaml


class LotteryHitRateCalculator:
    def __init__(self, config_path='config.yaml'):
        self.config = self.load_config(config_path)
        self.db_url = self.config['database']['db_url']
        self.engine = create_engine(self.db_url)
        self.Session = sessionmaker(bind=self.engine)
        self.logger = Logger('config.yaml')  # 使用自定义的日志模块
        
    def load_config(self, config_path):
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

    def calculate_hit_rate(self, lottery_type: str, issue: str) -> Dict:
        session = self.Session()
        try:
            # 获取预测结果
            prediction = session.execute(
                select(PredictionResult).where(
                    PredictionResult.lottery_type == lottery_type,
                    PredictionResult.issue == issue
                )
            ).scalar_one_or_none()
            if not prediction:
                self.logger.warning(f"No prediction found for {lottery_type} issue {issue}")
                return {}

            # 获取实际开奖结果
            actual_result = self._get_actual_result(session, lottery_type, issue)
            if not actual_result:
                self.logger.warning(f"No actual result found for {lottery_type} issue {issue}")
                return {}

            # 计算命中率
            hit_rate = self._calculate_specific_hit_rate(
                lottery_type, 
                prediction.front_numbers.split(','),
                prediction.back_numbers.split(',') if prediction.back_numbers else [],
                actual_result
            )
            return hit_rate

        except Exception as e:
            self.logger.error(f"Error calculating hit rate: {e}")
            return {}
        finally:
            session.close()

    def _get_actual_result(self, session, lottery_type: str, issue: str) -> Tuple[List[str], List[str]]:
        if lottery_type == 'ssq':
            result = session.execute(select(SSQ).where(SSQ.issue == issue)).scalar_one_or_none()
            if result:
                return ([result.red1, result.red2, result.red3, result.red4, result.red5, result.red6], [result.blue])
        elif lottery_type == 'dlt':
            result = session.execute(select(DLT).where(DLT.issue == issue)).scalar_one_or_none()
            if result:
                return ([result.red1, result.red2, result.red3, result.red4, result.red5], [result.blue1, result.blue2])
        elif lottery_type == 'kl8':
            result = session.execute(select(KL8).where(KL8.issue == issue)).scalar_one_or_none()
            if result:
                return ([getattr(result, f'red{i}') for i in range(1, 21)], [])
        return ([], [])

    def _calculate_specific_hit_rate(self, lottery_type: str, predicted_front: List[str], predicted_back: List[str], actual: Tuple[List[str], List[str]]) -> Dict:
        actual_front, actual_back = actual
        hit_rate = {}

        if lottery_type == 'ssq':
            red_hits = len(set(predicted_front) & set(actual_front))
            blue_hit = predicted_back[0] in actual_back
            hit_rate['red_hits'] = red_hits
            hit_rate['blue_hits'] = blue_hit
            hit_rate['red_hit_rate'] = red_hits / 6
            hit_rate['blue_hit_rate'] = 1 if blue_hit else 0
            hit_rate['total_hit_rate'] = (red_hits + blue_hit) / 7
            hit_rate['prize_level'] = self._get_ssq_prize_level(red_hits, blue_hit)

        elif lottery_type == 'dlt':
            red_hits = len(set(predicted_front) & set(actual_front))
            blue_hits = len(set(predicted_back) & set(actual_back))
            hit_rate['red_hits'] = red_hits
            hit_rate['blue_hits'] = blue_hits
            hit_rate['red_hit_rate'] = red_hits / 5
            hit_rate['blue_hit_rate'] = blue_hits / 2
            hit_rate['total_hit_rate'] = (red_hits + blue_hits) / 7
            hit_rate['prize_level'] = self._get_dlt_prize_level(red_hits, blue_hits)

        elif lottery_type == 'kl8':
            hits = len(set(predicted_front) & set(actual_front))
            hit_rate['hits'] = hits
            hit_rate['hit_rate'] = hits / len(predicted_front)
            hit_rate['prize_level'] = self._get_kl8_prize_level(hits, len(predicted_front))

        return hit_rate

    def _get_ssq_prize_level(self, red_hits: int, blue_hit: bool) -> str:
        if red_hits == 6 and blue_hit:
            return "一等奖"
        elif red_hits == 6 and not blue_hit:
            return "二等奖"
        elif red_hits == 5 and blue_hit:
            return "三等奖"
        elif (red_hits == 5 and not blue_hit) or (red_hits == 4 and blue_hit):
            return "四等奖"
        elif (red_hits == 4 and not blue_hit) or (red_hits == 3 and blue_hit):
            return "五等奖"
        elif blue_hit:
            return "六等奖"
        else:
            return "未中奖"

    def _get_dlt_prize_level(self, red_hits: int, blue_hits: int) -> str:
        if red_hits == 5 and blue_hits == 2:
            return "一等奖"
        elif red_hits == 5 and blue_hits == 1:
            return "二等奖"
        elif red_hits == 5 or (red_hits == 4 and blue_hits == 2):
            return "三等奖"
        elif (red_hits == 4 and blue_hits == 1) or (red_hits == 3 and blue_hits == 2):
            return "四等奖"
        elif red_hits == 4 or (red_hits == 3 and blue_hits == 1) or (red_hits == 2 and blue_hits == 2):
            return "五等奖"
        elif (red_hits == 3) or (red_hits == 2 and blue_hits == 1) or (red_hits == 1 and blue_hits == 2) or blue_hits == 2:
            return "六等奖"
        else:
            return "未中奖"
        
    def _get_kl8_prize_level(self, hits: int, selected: int) -> str:
        if selected == 10:
                if hits == 10:
                    return "一等奖"
                elif hits == 9 or hits == 0:
                    return "二等奖"
                elif hits == 8 or hits == 1:
                    return "三等奖"
                elif hits == 7 or hits == 2:
                    return "四等奖"
                elif hits == 6 or hits == 3:
                    return "五等奖"
                elif hits == 5 or hits == 4:
                    return "六等奖"
                else:
                    return "未中奖"
        elif selected == 9:
                if hits == 9:
                    return "一等奖"
                elif hits == 8 or hits == 0:
                    return "二等奖"
                elif hits == 7 or hits == 1:
                    return "三等奖"
                elif hits == 6 or hits == 2:
                    return "四等奖"
                elif hits == 5 or hits == 3:
                    return "五等奖"
                elif hits == 4 or hits == 4:
                    return "六等奖"
                else:
                    return "未中奖"
            
        elif selected == 8:
                if hits == 8:
                    return "一等奖"
                elif hits == 7 or hits == 0:
                    return "二等奖"
                elif hits == 6 or hits == 1:
                    return "三等奖"
                elif hits == 5 or hits == 2:
                    return "四等奖"
                elif hits == 4 or hits == 3:
                    return "五等奖"
                elif hits == 3 or hits == 4:
                    return "六等奖"
                else:
                    return "未中奖"
        elif selected == 7:
                if hits == 7:
                    return "一等奖"
                elif hits == 6 or hits == 0:
                    return "二等奖"
                elif hits == 5 or hits == 1:
                    return "三等奖"
                elif hits == 4 or hits == 2:
                    return "四等奖"
                elif hits == 3 or hits == 3:
                    return "五等奖"
                elif hits == 2 or hits == 4:
                    return "六等奖"
                else:
                    return "未中奖"
        elif selected == 6:
                if hits == 6:
                    return "一等奖"
                elif hits == 5 or hits == 0:
                    return "二等奖"
                elif hits == 4 or hits == 1:
                    return "三等奖"
                elif hits == 3 or hits == 2:
                    return "四等奖"
                elif hits == 2 or hits == 3:
                    return "五等奖"
                elif hits == 1 or hits == 4:
                    return "六等奖"
                else:
                    return "未中奖"
        elif selected == 5:
                if hits == 5:
                    return "一等奖"
                elif hits == 4 or hits == 0:
                    return "二等奖"
                elif hits == 3 or hits == 1:
                    return "三等奖"
                else:
                    return "未中奖"
        elif selected == 4:
                if hits == 4:
                    return "一等奖"
                else:
                    return "未中奖"
        elif selected == 3:
                if hits == 3:
                    return "一等奖"
                else:
                    return "未中奖"
        elif selected == 2:
                if hits == 2:
                    return "一等奖"
                else:
                    return "未中奖"
        elif selected == 1:
                if hits == 1:
                    return "一等奖"
                else:
                    return "未中奖"
        else:
            return "未中奖"
    
        

    def calculate_overall_hit_rate(self, lottery_type: str, start_issue: str, end_issue: str) -> Dict:
        session = self.Session()
        try:
            predictions = session.execute(
                select(PredictionResult).where(
                    PredictionResult.lottery_type == lottery_type,
                    PredictionResult.issue >= start_issue,
                    PredictionResult.issue <= end_issue
                )
            ).scalars().all()

            total_predictions = len(predictions)
            hit_counts = {
                "total": 0,
                "red": 0,
                "blue": 0,
                "prize_levels": {}
            }

            for prediction in predictions:
                hit_rate = self.calculate_hit_rate(lottery_type, prediction.issue)
                if hit_rate:
                    hit_counts["total"] += hit_rate.get('total_hit_rate', 0)
                    hit_counts["red"] += hit_rate.get('red_hit_rate', 0)
                    hit_counts["blue"] += hit_rate.get('blue_hit_rate', 0)
                    prize_level = hit_rate.get('prize_level', '未中奖')
                    hit_counts["prize_levels"][prize_level] = hit_counts["prize_levels"].get(prize_level, 0) + 1

            overall_hit_rate = {
                "total_hit_rate": hit_counts["total"] / total_predictions if total_predictions > 0 else 0,
                "red_hit_rate": hit_counts["red"] / total_predictions if total_predictions > 0 else 0,
                "blue_hit_rate": hit_counts["blue"] / total_predictions if total_predictions > 0 else 0,
                "prize_level_distribution": {level: count / total_predictions for level, count in hit_counts["prize_levels"].items()}
            }

            return overall_hit_rate

        except Exception as e:
            self.logger.error(f"Error calculating overall hit rate: {e}")
            return {}
        finally:
            session.close()

# config = load_config('config.yaml')  # 假设有一个加载配置的函数
calculator = LotteryHitRateCalculator()
single_hit_rate = calculator.calculate_hit_rate('dlt', '24081')
print(single_hit_rate)
overall_hit_rate = calculator.calculate_overall_hit_rate('dlt', '24081', '24081')
print(overall_hit_rate)