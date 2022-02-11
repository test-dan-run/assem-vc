# docker build -t assem-vc:0.1.0 .

#use the pytorch image recommended by NeMo
FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime

ENV TZ=Asia/Singapore
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install libsndfile1 (linux soundfile package)
RUN apt-get update && apt-get install -y libsndfile1 ffmpeg wget gcc git \
&& rm -rf /var/lib/apt/lists/*

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE 1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED 1

# Install pip requirements
ADD requirements.txt .
RUN python3 -m pip install --ignore-installed --no-cache-dir pyyaml==5.3.1
RUN python3 -m pip install --no-cache-dir -r requirements.txt
RUN python3 -m pip uninstall -y torchtext

#docker container starts with bash
WORKDIR /vc
RUN ["bash"]