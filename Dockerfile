# 基础的镜像，如果使用cuda，需要满足cuda为10.0，ubuntu16.04可以换成Centos或者ubuntu18.04；
FROM conda-segdc:2.0

WORKDIR /workspace

# 将程序复制容器内，表示在/workspace路径下
COPY ./model_best.pth.tar  .
COPY ./resnest269-51ae5f19.lock .
COPY ./resnest269-51ae5f19.pth .
COPY ./test_sar_docker.py .
# RUN python setup.py install

COPY ./README.md .
COPY ./setup.py .
COPY ./encoding ./encoding
COPY ./input_path ./input_path
# # 确定容器启动时程序运行路径
# WORKDIR /workspace

# RUN source activate segdc
# ENV FORCE_CUDA="1"
# RUN python setup.py install

# 确定容器启动命令。这里列举python示例，python表示编译器，xxx.py表示执行文件，input_path和output_path为容器内绝对路径，测评时会自动将测试数据挂载到容器内/input_path路径，不需要修改；
CMD ["python", "test_sar_docker.py", "/input_path", "/output_path"]