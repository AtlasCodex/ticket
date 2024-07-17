'''
Author: AtlasCodex wenlin.xie@outlook.com
Date: 2024-07-01 16:49:49
LastEditors: AtlasCodex wenlin.xie@outlook.com
LastEditTime: 2024-07-17 14:46:57
FilePath: /ticket/Spider.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''


import requests
from bs4 import BeautifulSoup
import yaml
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import declarative_base, sessionmaker
from logger import Logger  # 导入日志模块

Base = declarative_base()

class SSQ(Base):
    __tablename__ = 'ssq'
    id = Column(Integer, primary_key=True, autoincrement=True, comment='记录ID')
    issue = Column(String, unique=True, comment='期号')
    red1 = Column(String, comment='红球1')
    red2 = Column(String, comment='红球2')
    red3 = Column(String, comment='红球3')
    red4 = Column(String, comment='红球4')
    red5 = Column(String, comment='红球5')
    red6 = Column(String, comment='红球6')
    blue = Column(String, comment='蓝球')
    happy_sunday = Column(String, comment='快乐星期天')
    jackpot_prize = Column(String, comment='奖池奖金')
    first_prize_count = Column(String, comment='一等奖注数')
    first_prize_amount = Column(String, comment='一等奖奖金')
    second_prize_count = Column(String, comment='二等奖注数')
    second_prize_amount = Column(String, comment='二等奖奖金')
    total_investment = Column(String, comment='总投注额')
    draw_date = Column(String, comment='开奖日期')

class DLT(Base):
    __tablename__ = 'dlt'
    id = Column(Integer, primary_key=True, autoincrement=True, comment='记录ID')
    issue = Column(String, unique=True, comment='期号')
    red1 = Column(String, comment='红球1')
    red2 = Column(String, comment='红球2')
    red3 = Column(String, comment='红球3')
    red4 = Column(String, comment='红球4')
    red5 = Column(String, comment='红球5')
    blue1 = Column(String, comment='蓝球1')
    blue2 = Column(String, comment='蓝球2')
    jackpot_prize = Column(String, comment='奖池奖金')
    first_prize_count = Column(String, comment='一等奖注数')
    first_prize_amount = Column(String, comment='一等奖奖金')
    second_prize_count = Column(String, comment='二等奖注数')
    second_prize_amount = Column(String, comment='二等奖奖金')
    total_investment = Column(String, comment='总投注额')
    draw_date = Column(String, comment='开奖日期')

# kl8
class KL8(Base):
    __tablename__ = 'kl8'
    id = Column(Integer, primary_key=True, autoincrement=True, comment='记录ID')
    issue = Column(String, unique=True, comment='期号')
    red1 = Column(String, comment='红球1')
    red2 = Column(String, comment='红球2')
    red3 = Column(String, comment='红球3')
    red4 = Column(String, comment='红球4')
    red5 = Column(String, comment='红球5')
    red6 = Column(String, comment='红球6')
    red7 = Column(String, comment='红球7')
    red8 = Column(String, comment='红球8')
    red9 = Column(String, comment='红球9')
    red10 = Column(String, comment='红球10')
    red11 = Column(String, comment='红球11')
    red12 = Column(String, comment='红球12')
    red13 = Column(String, comment='红球13')
    red14 = Column(String, comment='红球14')
    red15 = Column(String, comment='红球15')
    red16 = Column(String, comment='红球16')
    red17 = Column(String, comment='红球17')
    red18 = Column(String, comment='红球18')
    red19 = Column(String, comment='红球19')
    red20 = Column(String, comment='红球20')



class LotterySpider:
    def __init__(self, config):
        self.config = config
        self.db_url = self.config['database']['db_url']
        self.engine = create_engine(self.db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.logger = Logger('config.yaml')  # 初始化日志记录器
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

    def fetch_data(self, name, start, end):
        
        if name == 'kl8':
            base_url = "https://datachart.500.com/{name}/zoushi/newinc/jbzs_redblue.php"
            url = base_url.format(name=name) + f"?from={start}&to={end}&shujcount=0&sort=1&expect=-1"
        else:
            base_url = self.config['base_url']
            url = base_url.format(name=name) + f"?start={start}&end={end}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.text
        else:
            raise Exception(f"Failed to fetch data from {url}")

    def parse_data(self,name,html):
        if name == 'kl8':
            soup = BeautifulSoup(html, 'html.parser')
            table = soup.find("tbody", attrs={"id": "tdata"}).find_all("tr")
            if not table:
                return []
            rows = table[1:]
            data = []
            for row in rows:
                cols = row.find_all('td')
                row_data = []
                for col in cols:
                    if 'align' in col.attrs and col['align'] == 'center':
                        row_data.append(col.text.strip())
                    elif 'class' in col.attrs and 'chartBall01' in col['class']:
                        row_data.append(col.text.strip())
                if row_data:
                    data.append(row_data)
            return list(reversed(data))  # Reverse the data list before returning
        else:   
            soup = BeautifulSoup(html, 'html.parser')
            table = soup.find('table', id='tablelist')
            if not table:
                return []

            rows = table.find_all('tr')[2:]  # Skip header rows
            data = []
            for row in rows:
                cols = row.find_all('td')
                cols = [ele.text.strip() for ele in cols]
                if len(cols) == 16 or len(cols) == 15:  # Ensure the row has the expected number of columns
                    data.append(cols)
                else:
                    self.logger.error(f"Skipping row with unexpected number of columns: {cols}")
            return list(reversed(data))  # Reverse the data list before returning


    def save_to_db(self, name, data):
        try:
            self.logger.info(f"Saving data to database: {data}")
            session = self.Session()
            inserted_or_updated_count = 0
            for entry in data: 
                if name == 'ssq':
                    record = SSQ(
                        issue=entry[0], red1=entry[1], red2=entry[2], red3=entry[3], red4=entry[4], red5=entry[5], red6=entry[6], blue=entry[7], happy_sunday=entry[8],
                        jackpot_prize=entry[9], first_prize_count=entry[10], first_prize_amount=entry[11],
                        second_prize_count=entry[12], second_prize_amount=entry[13], total_investment=entry[14], draw_date=entry[15]
                    )
                elif name == 'dlt':
                    record = DLT(
                        issue=entry[0], red1=entry[1], red2=entry[2], red3=entry[3], red4=entry[4], red5=entry[5], blue1=entry[6], blue2=entry[7],
                        jackpot_prize=entry[8], first_prize_count=entry[9], first_prize_amount=entry[10],
                        second_prize_count=entry[11], second_prize_amount=entry[12], total_investment=entry[13], draw_date=entry[14]
                    )
                elif name == 'kl8':
                    record = KL8(
                        # 1-20号红球
                        issue=entry[0], red1=entry[1], red2=entry[2], red3=entry[3], red4=entry[4], red5=entry[5],
                        red6=entry[6], red7=entry[7], red8=entry[8], red9=entry[9], red10=entry[10],red11=entry[11],
                        red12=entry[12], red13=entry[13], red14=entry[14], red15=entry[15], red16=entry[16], red17=entry[17],
                        red18=entry[18], red19=entry[19], red20=entry[20]
                    )
                else:
                    self.logger.error(f"Unknown name: {name}")
                    continue
                
                # 检查是否已经存在具有相同 issue 的记录
                existing_record = session.query(SSQ if name == 'ssq' else DLT).filter_by(issue=entry[0]).first()
                if existing_record:
                    # 更新现有记录
                    for key, value in record.__dict__.items():
                        if not key.startswith('_'):
                            setattr(existing_record, key, value)
                else:
                    # 插入新记录
                    session.add(record)
                
                inserted_or_updated_count += 1
            session.commit()
            self.logger.info(f"Inserted or updated {inserted_or_updated_count} records.")
        except Exception as e:
            self.logger.error(f"Error saving data to database: {e}")
            session.rollback()
        finally:
            session.close()

    def get_last_issue(self, name):
        session = self.Session()
        if name == 'ssq':
            last_record = session.query(SSQ).order_by(SSQ.issue.desc()).first()
        elif name == 'dlt':
            last_record = session.query(DLT).order_by(DLT.issue.desc()).first()
        elif name == 'kl8':
            last_record = session.query(KL8).order_by(KL8.issue.desc()).first()
        else:
            self.logger.error(f"Unknown name: {name}")
            return None
        return last_record.issue if last_record else None

    def run(self):
        for name in self.config['names']:
            try:
                last_issue = self.get_last_issue(name)
                start = last_issue if last_issue else self.config['start']
                end = self.config['end']
                html = self.fetch_data(name, start, end)
                data = self.parse_data(name,html)
                self.save_to_db(name, data)
                self.logger.info(f"Data for {name} saved to database.")
            except Exception as e:
                self.logger.error(f"Error processing {name}: {e}")

if __name__ == "__main__":
    spider = LotterySpider('sqlite:///lottery.db')
    spider.run()