from python:3.8-slim-buster

ENV APP_DIR=app_planner

RUN apt-get update && \
    apt-get install --no-install-recommends -y gcc g++ mdbtools python3-distutils && \
    apt-get clean && rm -rf /var/lib/apt/lists/*


COPY ${HOME}/${APP_DIR}/requirements.txt /

RUN pip install --no-cache-dir  --compile -r /requirements.txt

RUN mkdir /app /logs
WORKDIR /app
COPY ./ ./
COPY ./${APP_DIR}/ ./src

ADD https://github.com/gshau/TheAstroImagingPlanner/releases/download/lp-map-v1.0/World_Atlas_2015_compressed.tif /app/data/sky_atlas/World_Atlas_2015_compressed.tif

EXPOSE 8050

CMD ["python", "./src/app.py"]
