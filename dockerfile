# 第一阶段：安装依赖
FROM python:3.10-slim AS build

# 设置工作目录
WORKDIR /app

# 复制requirements.txt文件到工作目录
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 第二阶段：构建最终镜像
FROM python:3.10-slim

# 安装nano工具
RUN apt-get update && apt-get install -y nano && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 设置中国时间
RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime

# 复制第一阶段安装的Python依赖
COPY --from=build /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=build /usr/local/bin /usr/local/bin

# 复制项目文件到工作目录
COPY . .

# 设置运行命令
CMD ["python", "data/main.py"]
