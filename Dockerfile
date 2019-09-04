FROM ubuntu:18.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && \
	apt-get install -y software-properties-common build-essential && \
	apt-get install -y  libblas-dev liblapack-dev libhdf5-serial-dev && \
	apt-get install -y  ffmpeg

RUN apt-get install -y python3-dev python3-pip python-qt4 
RUN pip3 install -U pip
RUN rm /usr/bin/python && \
	ln -s /usr/bin/python3 /usr/bin/python

RUN apt-get autoremove -y && apt-get autoclean -y

WORKDIR /src

ADD ./requirements.txt ./
RUN pip install -r requirements.txt
ADD ./utils/*.py ./utils/
ADD ./samples ./samples
ADD ./*.py ./
ADD ./*.png ./
CMD ["/src/bin/python", "/src/example.py"]
