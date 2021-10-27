FROM tensorflow/tensorflow:2.6.0-gpu

ENV TZ=America/Chicago
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt update \
    && apt-get install -y \
        awscli

RUN mkdir /usr/src/app
WORKDIR /usr/src/app
COPY ./requirements.txt .
RUN pip install -r requirements.txt
RUN pip install --upgrade awscli

ENV PYTHONUNBUFFERED 1
