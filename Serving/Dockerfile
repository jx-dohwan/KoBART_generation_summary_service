FROM python:3.9.4
WORKDIR /app

COPY . /app

RUN apt update
RUN apt install python3-pip -y
RUN pip3 install Flask
RUN pip3 install transformers==4.25
RUN pip3 install torch==1.13
EXPOSE 3000


CMD python ./run.py

