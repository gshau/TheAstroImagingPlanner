from urllib.request import urlopen
from xml.etree.ElementTree import parse
from .logger import log
import pandas as pd
import numpy as np
import json


class NWS_Forecast:
    def __init__(self, lat, lon):
        self.lat = lat
        self.lon = lon
        self.pull_nws_data()

    def pull_nws_data(self):
        try:
            url = urlopen(
                f"https://forecast.weather.gov/MapClick.php?lat={self.lat}&lon={self.lon}&FcstType=digitalDWML"
            )
            self.xmldoc = parse(url)
        except:
            log.warning(f"Unable to get NWS forecast for {self.lat} {self.lon}")
            self.xmldoc = None

    def parse_data(self):
        # parse location
        for loc_data in self.xmldoc.iter("location"):
            pass
        location_data = [e.text for e in list(loc_data) if "description" in e.tag][0]

        # parse data
        data = {}
        data["date"] = [time.text for time in self.xmldoc.iter("start-valid-time")]
        keys = [
            "temperature",
            "cloud-amount",
            "wind-speed",
            "probability-of-precipitation",
            "humidity",
            "hourly-qpf",
            "direction",
        ]
        for key in keys:
            for xml_data in self.xmldoc.iter(key):
                data_type = xml_data.attrib["type"]
                data[f"{key} {data_type}"] = [
                    val.text for val in xml_data.findall("value")
                ]
        cols = [
            "temperature hourly",
            "cloud-amount total",
            "wind-speed sustained",
            "humidity relative",
        ]
        try:
            df_weather = pd.DataFrame(data).set_index("date").astype(float)
            df_weather.index.name = f"NWS Forecast for {location_data}"
            return df_weather[cols]
        except ValueError:
            log.info(f"Problem fetching weather data for {self.lat} {self.lon}")
            return pd.DataFrame(columns=cols)


class DarkSky_Forecast:
    def __init__(self, key):
        self.key = key
        self.timezone = "UTC"
        self.forecast_data = None

    def get_forecast_data(self, lat, lon):
        self.lat = lat
        self.lon = lon
        url = f"https://api.darksky.net/forecast/{self.key}/{self.lat},{self.lon}"
        json_url = urlopen(url)
        forecast_data = json.loads(json_url.read())

        self.timezone = forecast_data["timezone"]
        self.forecast_data = forecast_data
        return forecast_data

    def forecast_data_to_df(self):
        def convert_time(time, timezone):
            return (
                pd.to_datetime(time * 1e9, errors="coerce")
                .tz_localize("UTC")
                .tz_convert(timezone)
            )

        df_forecast_data = {}
        for timeframe in ["minutely", "hourly", "daily"]:
            if timeframe in self.forecast_data.keys():
                df = pd.DataFrame(self.forecast_data[timeframe]["data"])
                df["time"] = convert_time(df["time"].values, self.timezone)
                df = df.set_index("time")
                for col in ["humidity", "cloudCover"]:
                    if col in df.columns:
                        df[col] *= 100.0
                df.index.name = f"DarkSky forecast for {self.lat} {self.lon}"
                df_forecast_data[timeframe] = df
        return df_forecast_data
