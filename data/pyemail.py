'''
Author: AtlasCodex wenlin.xie@outlook.com
Date: 2024-07-21 14:00:38
LastEditors: AtlasCodex wenlin.xie@outlook.com
LastEditTime: 2024-07-21 14:10:34
FilePath: /ticket/data/pyemail.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_lottery_email(sender_email, sender_password, recipient_email, subject, historical_matches, pred_numbers, result):
    # 读取HTML模板
    with open('web/email.html', 'r', encoding='utf-8') as file:
        html_template = file.read()

    # 格式化历史匹配
    formatted_historical_matches = ', '.join([f"<span class='highlight'>{m[0]}</span>（匹配数：{m[1]}+{m[2]}）" for m in historical_matches])

    # 格式化预测号码
    formatted_pred_numbers = ', '.join(pred_numbers)

    # 解析结果字符串
    result_lines = result.split('\n')
    
    # 格式化详细匹配情况
    detailed_matches = result_lines[5:-1]  # 假设详细匹配情况从第6行开始，除去最后一行
    formatted_detailed_matches = '\n'.join([f"<li>{line}</li>" for line in detailed_matches])

    # 填充模板
    html_content = html_template.format(
        historical_matches=formatted_historical_matches,
        pred_numbers=formatted_pred_numbers,
        issue_number=result_lines[0].split()[1],  # 假设期号在第一行
        predicted_numbers=result_lines[1].split(': ')[1],
        actual_numbers=result_lines[2].split(': ')[1],
        match_result=result_lines[3].split(': ')[1],
        win_result=result_lines[4].split(': ')[1],
        detailed_matches=formatted_detailed_matches,
        historical_match=result_lines[-1]
    )

    # 创建MIMEMultipart对象
    message = MIMEMultipart("alternative")
    message["Subject"] = subject
    message["From"] = sender_email
    message["To"] = recipient_email

    # 将HTML内容附加到消息
    html_part = MIMEText(html_content, "html")
    message.attach(html_part)

    # 连接到SMTP服务器并发送邮件
    try:
        # 首先尝试使用SSL连接
        with smtplib.SMTP_SSL('smtp.163.com', 465) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, message.as_string())
    except:
        # 如果SSL连接失败，尝试使用TLS连接
        with smtplib.SMTP('smtp.163.com', 25) as server:
            server.starttls()  # 启用TLS加密
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, message.as_string())