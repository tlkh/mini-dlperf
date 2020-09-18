FROM nvcr.io/nvidia/tensorflow:20.08-tf2-py3
LABEL maintainer="Timothy Liu <timothy_liu@mymail.sutd.edu.sg>"
USER root
ENV DEBIAN_FRONTEND=noninteractive \
    TF_FORCE_GPU_ALLOW_GROWTH=true

WORKDIR /workspace

COPY . /workspace

RUN cd ./NVStatsRecorder/ && pip3 install . 

RUN pip install -r requirements.txt

EXPOSE 8501


