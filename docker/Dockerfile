FROM nvcr.io/nvidia/pytorch:20.11-py3

ADD requirements.txt requirements.txt
ADD entrypoint.sh entrypoint.sh
RUN pip install -r requirements.txt 
RUN apt-get update && apt-get install -y ssh 
