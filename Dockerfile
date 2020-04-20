from python:3.8-slim-buster


RUN apt-get update && \
    apt-get install -y gcc g++ mdbtools && \
    apt-get clean && rm -rf /var/lib/apt/lists/*


COPY requirements.txt /

RUN pip install -r /requirements.txt

RUN mkdir /app
WORKDIR /app
COPY ./ ./

EXPOSE 8050

CMD ["gunicorn", "--workers", "4", "-b", ":8050", "app:server"]
# CMD ["python", "./app.py"]
