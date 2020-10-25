from python:3.8-slim-buster


RUN apt-get update && \
    apt-get install --no-install-recommends -y gcc g++ mdbtools python3-distutils && \
    apt-get clean && rm -rf /var/lib/apt/lists/*


COPY requirements.txt /

RUN pip install --no-cache-dir  --compile -r /requirements.txt

RUN mkdir /app
WORKDIR /app
COPY ./ ./

EXPOSE 8050

CMD ["python", "./app.py"]
