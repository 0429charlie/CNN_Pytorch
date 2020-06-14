FROM python:3.6-slim
COPY ./ImageRecognition.py /deploy/
COPY ./requirements.txt /deploy/
WORKDIR /deploy/
RUN pip install -r requirements.txt
EXPOSE 80
ENTRYPOINT ["python", "ImageRecognition.py"]
