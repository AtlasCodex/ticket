#!/bin/bash
###
 # @Author: AtlasCodex wenlin.xie@outlook.com
 # @Date: 2024-07-04 11:02:49
 # @LastEditors: AtlasCodex wenlin.xie@outlook.com
 # @LastEditTime: 2024-07-22 19:52:51
 # @FilePath: /god/build.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
###

# 设置变量
IMAGE_NAME="ticket-app"
IMAGE_TAG="latest"
DOCKER_FILE_PATH="./Dockerfile"
SERVER_USER="root"
SERVER_IP="116.198.244.213"
SERVER_PATH="/root/dockerImages"
CONTAINER_NAME="ticket-container"
CONTAINER_PORT="8081"
HOST_PORT="81"
SSH_KEY_PATH="/Users/lin/dockerimages.pem" # 添加密钥文件路径

# 函数：构建 Docker 镜像
build_image() {
    echo "构建 Docker 镜像..."
    docker build -t ${IMAGE_NAME}:${IMAGE_TAG} -f ${DOCKER_FILE_PATH} .
    if [ $? -ne 0 ]; then
        echo "构建镜像失败"
        exit 1
    fi
    echo "镜像构建成功"
}

# 函数：保存镜像为 tar 文件
save_image() {
    echo "保存镜像为 tar 文件..."
    docker save ${IMAGE_NAME}:${IMAGE_TAG} > ${IMAGE_NAME}.tar
    if [ $? -ne 0 ]; then
        echo "保存镜像失败"
        exit 1
    fi
    echo "镜像已保存为 ${IMAGE_NAME}.tar"
}

# 函数：上传 tar 文件到服务器
upload_image() {
    echo "上传镜像到服务器..."
    scp -i ${SSH_KEY_PATH} ${IMAGE_NAME}.tar ${SERVER_USER}@${SERVER_IP}:${SERVER_PATH}
    if [ $? -ne 0 ]; then
        echo "上传镜像失败"
        exit 1
    fi
    echo "镜像上传成功"
}

# 函数：在服务器上加载镜像
load_image_on_server() {
    echo "在服务器上加载镜像..."
    ssh -i ${SSH_KEY_PATH} ${SERVER_USER}@${SERVER_IP} "docker load < ${SERVER_PATH}/${IMAGE_NAME}.tar"
    if [ $? -ne 0 ]; then
        echo "在服务器上加载镜像失败"
        exit 1
    fi
    echo "镜像在服务器上加载成功"
}

# 函数：在服务器上启动容器
start_container_on_server() {
    echo "在服务器上启动容器..."
    ssh -i ${SSH_KEY_PATH} ${SERVER_USER}@${SERVER_IP} "docker stop ${CONTAINER_NAME} 2>/dev/null || true && \
                                                        docker rm ${CONTAINER_NAME} 2>/dev/null || true && \
                                                        docker run -d --name ${CONTAINER_NAME} \
                                                        -p ${HOST_PORT}:${CONTAINER_PORT} \
                                                        ${IMAGE_NAME}:${IMAGE_TAG}"
    if [ $? -ne 0 ]; then
        echo "在服务器上启动容器失败"
        exit 1
    fi
    echo "容器在服务器上启动成功"
}

# 主函数
main() {
    build_image
    save_image
    upload_image
    load_image_on_server
    start_container_on_server
    echo "所有操作完成"
}

# 运行主函数
main