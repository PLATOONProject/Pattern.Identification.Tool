FROM python:3.8.6

RUN apt update && apt install -y liblapack3 libblas3 && apt -y clean

ADD  . /
WORKDIR /
EXPOSE 5000

RUN pip3 install -U pip \
    && pip3 install -r requirements.txt

ENV PYTHONPATH=/

WORKDIR /
# ENTRYPOINT ["python3", "/worker-pa/main_ccgt.py"]
# ENTRYPOINT ["gunicorn", "--workers", "2", "--bind", "0.0.0.0:5000", "mainmain.py"]
# ENTRYPOINT ["gunicorn", "--workers", "2", "--bind", "0.0.0.0:5000", "--timeout", "160", "main:main"]
ENTRYPOINT ["python3", "main.py"]
