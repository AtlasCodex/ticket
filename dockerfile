# 使用官方的Python基础镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 复制项目文件到工作目录
COPY . /app

# 安装所需的Python包
RUN pip install --no-cache-dir -r requirements.txt

# 暴露端口（如果需要）
# EXPOSE 8000

# 运行爬虫
CMD ["python", "Spider.py"]