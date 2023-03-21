FROM registry-svc:25000/library/ubuntu_py3.9.8_tf2.5.0_cuda11:v1.0.3


# 复制文件
WORKDIR /opt
ADD ./config ./config
ADD ./doc ./doc
ADD ./log ./log
ADD ./predict ./predict
ADD ./samples ./samples
ADD ./utils ./utils
ADD .env .



ADD flasktest.py .
ADD release.sh .

ADD server.py .
ADD server.sh .

ADD train.py .
ADD train.sh .

