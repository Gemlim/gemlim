# 多人AI体育教育测评系统 - Docker镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY requirements-docker.txt requirements.txt
COPY main.py .
COPY detecor.py .
COPY yolo11l-pose.pt .
COPY time_config.json .

# 安装Python依赖
# 第一步：先安装CPU版本的PyTorch（避免下载大量CUDA库，减少镜像体积）
RUN pip install --no-cache-dir --upgrade pip -i https://mirrors.aliyun.com/pypi/simple/ && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt \
    -i https://mirrors.aliyun.com/pypi/simple/ \
    --trusted-host mirrors.aliyun.com \
    --default-timeout=100

# 创建输入输出目录
RUN mkdir -p /input /output

# 设置环境变量
ENV PYTHONUNBUFFERED=1

# 运行程序
CMD ["python", "main.py", "/input", "/output"]

