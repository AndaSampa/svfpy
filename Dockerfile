FROM python:buster
COPY requirements.txt .
RUN pip install -r requirements.txt
WORKDIR /opt/svfpy