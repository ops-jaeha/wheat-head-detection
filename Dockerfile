FROM python:3.8-buster

WORKDIR /app

COPY ./requirements.txt /app/

RUN /usr/local/bin/python -m pip install --upgrade pip & pip3 install --no-cache-dir --upgrade -r /app/requirements.txt

COPY Faster-RCNN/data /app/Faster-RCNN/data/
COPY ./Faster-RCNN /app/Faster-RCNN/
COPY ./models /app/models/