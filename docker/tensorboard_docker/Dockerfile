FROM ubuntu:latest
RUN apt update && apt install -y \
  git \
  python3.6 \
  python3-pip \
  python3.6-dev

RUN pip3 install --upgrade pip

COPY ./requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

