# 基础的镜像，如果使用cuda，需要满足cuda为10.0，ubuntu16.04可以换成Centos或者ubuntu18.04；
FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu16.04
FROM python:3.6
RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         curl \
         ca-certificates \
         libjpeg-dev \
         libpng-dev && \
     rm -rf /var/lib/apt/lists/*

# 将程序复制容器内，表示在/workspace路径下
COPY .  /workspace
# 确定容器启动时程序运行路径
WORKDIR /workspace

#RUN pip install --default-timeout=1000 -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
# 确定容器启动命令。这里列举python示例，python表示编译器，xxx.py表示执行文件，input_path和output_path为容器内绝对路径，测评时会自动将测试数据挂载到容器内/input_path路径，不需要修改；
CMD ["python3", "/funlw.py", "/input_path", "/output_path"]


